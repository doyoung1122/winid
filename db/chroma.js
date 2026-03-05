import { ChromaClient, IncludeEnum } from "chromadb";
import { CHROMA_URL, EMB_URL, EMB_MODEL } from "../server/src/config/env.js";
import { l2Normalize } from "./repo.js";

// ---- ChromaDB Client ----
// chromadb 3.x는 baseUrl 대신 { host, port, ssl } 방식 사용
const _chromaUrl = new URL(CHROMA_URL);
const client = new ChromaClient({
  host: _chromaUrl.hostname,
  port: Number(_chromaUrl.port) || (_chromaUrl.protocol === "https:" ? 443 : 80),
  ssl: _chromaUrl.protocol === "https:",
});

// Custom embedding function that calls vLLM bge-m3
const vllmEmbedder = {
  name: "vllm-bge-m3",

  async generate(texts) {
    const r = await fetch(`${EMB_URL}/v1/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: EMB_MODEL, input: texts }),
    });
    if (!r.ok) {
      const errText = await r.text();
      throw new Error(`embedding failed ${r.status}: ${errText}`);
    }
    const j = await r.json();
    return j.data.map((d) => d.embedding);
  },

  defaultSpace() {
    return "cosine";
  },

  supportedSpaces() {
    return ["cosine"];
  },
};

// ---- Collection cache ----
// Promise-deduplication prevents concurrent getOrCreateCollection races
let _collection = null;
let _collectionPromise = null;
let _statsCollection = null;
let _statsCollectionPromise = null;

async function getCollection() {
  if (_collection) return _collection;
  if (!_collectionPromise) {
    _collectionPromise = client.getOrCreateCollection({
      name: "vfims_documents",
      embeddingFunction: null,
      metadata: { "hnsw:space": "cosine" },
    }).then((col) => {
      _collection = col;
      _collectionPromise = null;
      return col;
    }).catch((err) => {
      _collectionPromise = null;
      throw err;
    });
  }
  return _collectionPromise;
}

async function getStatsCollection() {
  if (_statsCollection) return _statsCollection;
  if (!_statsCollectionPromise) {
    _statsCollectionPromise = client.getOrCreateCollection({
      name: "vfims_stats",
      embeddingFunction: null,
      metadata: { "hnsw:space": "cosine" },
    }).then((col) => {
      _statsCollection = col;
      _statsCollectionPromise = null;
      return col;
    }).catch((err) => {
      _statsCollectionPromise = null;
      throw err;
    });
  }
  return _statsCollectionPromise;
}

// Invalidate collection caches (e.g. after external deletion)
function resetCollectionCache() {
  _collection = null;
  _collectionPromise = null;
  _statsCollection = null;
  _statsCollectionPromise = null;
}

// ---- counter for unique IDs ----
let idCounter = 0;

// ---- inserts ----

/**
 * Insert a document with its pre-computed embedding into ChromaDB.
 * Same interface as repo.js insertDocumentWithEmbedding.
 */
async function insertDocumentWithEmbedding(content, metadata, embedding) {
  if (embedding.length !== 1024) throw new Error("Dimension mismatch");

  const embNorm = Array.from(l2Normalize(embedding));
  const docId = `doc_${Date.now()}_${idCounter++}`;
  const flatMeta = flattenMetadata(metadata ?? {});

  // Retry up to 3 times to handle transient ChromaDB errors
  for (let attempt = 0; attempt < 3; attempt++) {
    const col = await getCollection();
    try {
      await col.upsert({
        ids: [docId],
        embeddings: [embNorm],
        documents: [content],
        metadatas: [flatMeta],
      });
      return docId;
    } catch (err) {
      const msg = err.message || "";
      // Stale collection reference (collection deleted externally) → reset cache & retry
      if (attempt < 2 && (msg.includes("does not exist") || (msg.includes("Bad request") && msg.includes(docId.split("_")[1])))) {
        console.warn("[chroma] Collection stale, resetting cache and retrying...");
        resetCollectionCache();
        continue;
      }
      // "already exists" on upsert = concurrent insert won, treat as success
      if (msg.includes("already exists")) {
        return docId;
      }
      // Transient error on new collection → wait & retry once
      if (attempt < 2 && (msg.includes("resource") || msg.includes("timeout"))) {
        await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
        continue;
      }
      throw err;
    }
  }
}

/**
 * Flatten nested metadata object for ChromaDB storage.
 * ChromaDB only supports flat key-value pairs with string/number/bool values.
 */
function flattenMetadata(meta) {
  const flat = {};
  for (const [k, v] of Object.entries(meta)) {
    if (v == null) continue;
    if (typeof v === "string" || typeof v === "number" || typeof v === "boolean") {
      flat[k] = v;
    } else {
      // Serialize complex values as JSON strings
      flat[k] = JSON.stringify(v);
    }
  }
  return flat;
}

/**
 * Unflatten metadata by parsing JSON string values back to objects.
 */
function unflattenMetadata(flat) {
  if (!flat) return {};
  const meta = {};
  for (const [k, v] of Object.entries(flat)) {
    if (typeof v === "string" && (v.startsWith("{") || v.startsWith("["))) {
      try {
        meta[k] = JSON.parse(v);
      } catch {
        meta[k] = v;
      }
    } else {
      meta[k] = v;
    }
  }
  return meta;
}

/**
 * Match documents by query embedding.
 * Same interface as repo.js matchDocuments.
 * ChromaDB cosine distance = 1 - cosine_similarity, so we convert.
 */
async function matchDocuments(queryEmbedding, options = {}) {
  const { k = 8, threshold = 0.7, types = null, sha256 = null, doc_type = null, metaFilter = null } = options;

  // stats 문서는 전용 컬렉션 사용 (대형 컬렉션 HNSW 오류 방지)
  const col = doc_type === "stats" ? await getStatsCollection() : await getCollection();
  const qNorm = Array.from(l2Normalize(queryEmbedding));

  // Build ChromaDB where filter
  const whereConditions = [];
  if (types && types.length > 0) {
    whereConditions.push({ type: { $in: types } });
  }
  if (sha256) {
    whereConditions.push({ sha256: sha256 });
  }
  // doc_type="stats"는 전용 컬렉션을 쓰므로 where 필터 불필요
  if (doc_type && doc_type !== "stats") {
    whereConditions.push({ doc_type: { $eq: doc_type } });
  }
  // 커스텀 메타데이터 필터 (예: 날짜 특정)
  if (metaFilter) {
    for (const [key, val] of Object.entries(metaFilter)) {
      whereConditions.push({ [key]: { $eq: val } });
    }
  }

  const queryArgs = {
    queryEmbeddings: [qNorm],
    nResults: k,
    include: [IncludeEnum.documents, IncludeEnum.metadatas, IncludeEnum.distances],
  };

  if (whereConditions.length === 1) {
    queryArgs.where = whereConditions[0];
  } else if (whereConditions.length > 1) {
    queryArgs.where = { $and: whereConditions };
  }

  let result;
  try {
    result = await col.query(queryArgs);
  } catch (err) {
    // 필터 결과가 k보다 적거나 컬렉션이 비어있을 때 ChromaDB가 던지는 오류들
    if (
      err.message?.includes("not enough") ||
      err.message?.includes("empty") ||
      err.message?.includes("contigious") ||
      err.message?.includes("ef or M")
    ) {
      return [];
    }
    throw err;
  }

  // ChromaDB returns nested arrays: result.ids[0], result.distances[0], etc.
  const ids = result.ids[0] || [];
  const distances = result.distances[0] || [];
  const documents = result.documents[0] || [];
  const metadatas = result.metadatas[0] || [];

  const results = [];
  for (let i = 0; i < ids.length; i++) {
    // Convert cosine distance to cosine similarity
    const similarity = 1 - (distances[i] ?? 1);
    if (similarity < threshold) continue;

    results.push({
      id: ids[i],
      content: documents[i] || "",
      metadata: unflattenMetadata(metadatas[i]),
      similarity,
    });
  }

  // Already sorted by distance (ascending) from ChromaDB, which means sorted by similarity (descending)
  return results;
}

/**
 * 날짜 기반 통계 문서 직접 조회 (벡터 검색 없이 메타데이터 필터만 사용)
 * @param {{year: number, month: number, day: number}} date
 * @returns {Promise<Array<{id, content, metadata, similarity}>>}
 */
export async function getStatsByDate({ year, month, day }) {
  const col = await getStatsCollection();
  try {
    // day가 null이면 연/월만 필터 (지난달 등 월 단위 조회)
    const conditions = [{ year: { $eq: year } }, { month: { $eq: month } }];
    if (day != null) conditions.push({ day: { $eq: day } });
    const result = await col.get({
      where: { $and: conditions },
      include: [IncludeEnum.documents, IncludeEnum.metadatas],
    });
    const ids = result.ids || [];
    const documents = result.documents || [];
    const metadatas = result.metadatas || [];
    return ids.map((id, i) => ({
      id,
      content: documents[i] || "",
      metadata: unflattenMetadata(metadatas[i]),
      similarity: 1.0, // 직접 조회이므로 유사도 1로 설정
    }));
  } catch (err) {
    console.warn("getStatsByDate 오류:", err.message);
    return [];
  }
}

/**
 * 날짜 범위 기반 통계 문서 조회 (일주일 등)
 * @param {string} startDate "YYYY-MM-DD"
 * @param {string} endDate   "YYYY-MM-DD"
 */
export async function getStatsByDateRange(startDate, endDate) {
  const [sy, sm, sd] = startDate.split("-").map(Number);
  const [ey, em, ed] = endDate.split("-").map(Number);
  const start = new Date(sy, sm - 1, sd);
  const end   = new Date(ey, em - 1, ed);

  const results = [];
  const cur = new Date(start);
  while (cur <= end) {
    const year  = cur.getFullYear();
    const month = cur.getMonth() + 1;
    const day   = cur.getDate();
    const dayResults = await getStatsByDate({ year, month, day });
    results.push(...dayResults);
    cur.setDate(cur.getDate() + 1);
  }
  return results;
}

/**
 * 지역명·연도 기반 사례 문서 직접 조회 (벡터 검색 없이 메타데이터 필터 사용)
 * region은 포함 검색 (JS 포스트 필터), year는 ChromaDB where 필터
 * @param {{year?: number, region?: string}} meta
 * @returns {Promise<Array<{id, content, metadata, similarity}>>}
 */
export async function getCasesByMeta({ year, region, building } = {}) {
  const col = await getCollection();
  const conditions = [{ doc_type: { $eq: "case" } }];
  if (year) conditions.push({ year: { $eq: year } });

  const where = conditions.length === 1 ? conditions[0] : { $and: conditions };

  try {
    const result = await col.get({
      where,
      include: [IncludeEnum.documents, IncludeEnum.metadatas],
      limit: 50,
    });
    let items = (result.ids || []).map((id, i) => ({
      id,
      content: result.documents[i] || "",
      metadata: unflattenMetadata(result.metadatas[i]),
      similarity: 1.0,
    }));
    // region/building은 ChromaDB 부분 문자열 미지원 → JS 포스트 필터
    if (region) {
      items = items.filter((m) => (m.metadata.region || "").includes(region));
    }
    if (building) {
      items = items.filter((m) => (m.content || "").includes(building));
    }
    return items;
  } catch (err) {
    console.warn("getCasesByMeta 오류:", err.message);
    return [];
  }
}

/**
 * Get all text documents from the main collection.
 * Used for building the BM25 keyword search index.
 * @param {{textOnly?: boolean, limit?: number}} options
 * @returns {Promise<Array<{id, content, metadata}>>}
 */
export async function getAllDocuments({ textOnly = false, limit = 10000 } = {}) {
  const col = await getCollection();
  const queryArgs = {
    include: [IncludeEnum.documents, IncludeEnum.metadatas],
    limit,
  };
  if (textOnly) {
    queryArgs.where = { type: { $in: ["pdf", "text", "office", "hwpx", "hwp"] } };
  }
  try {
    const result = await col.get(queryArgs);
    return (result.ids || []).map((id, i) => ({
      id,
      content: result.documents[i] || "",
      metadata: unflattenMetadata(result.metadatas[i]),
    }));
  } catch (err) {
    console.warn("[chroma] getAllDocuments 오류:", err.message);
    return [];
  }
}

/**
 * No-op: ChromaDB manages its own storage; no cache needed.
 */
async function loadCache() {
  // Verify connection is alive
  try {
    await getCollection();
    console.log("ChromaDB collection ready.");
  } catch (err) {
    console.warn("ChromaDB connection check failed:", err.message);
  }
}

export {
  insertDocumentWithEmbedding,
  matchDocuments,
  loadCache,
};
