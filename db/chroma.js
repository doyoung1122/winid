import { ChromaClient, IncludeEnum } from "chromadb";
import { CHROMA_URL, EMB_URL, EMB_MODEL } from "../server/src/config/env.js";

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
let _collection = null;

async function getCollection() {
  if (_collection) return _collection;
  _collection = await client.getOrCreateCollection({
    name: "vfims_documents",
    embeddingFunction: null, // 항상 pre-computed embedding 직접 주입, EF 등록 불필요
    metadata: { "hnsw:space": "cosine" },
  });
  return _collection;
}

// ---- helpers ----
function l2Normalize(v) {
  if (!v || v.length === 0) throw new Error("l2Normalize: invalid vector");
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
  const n = Math.sqrt(sum) || 1;
  if (n === 1) return Array.from(v);
  const out = new Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
  return out;
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

  const col = await getCollection();
  const embNorm = l2Normalize(embedding);
  const docId = `doc_${Date.now()}_${idCounter++}`;

  // ChromaDB metadata must be flat (string/number/bool only)
  const flatMeta = flattenMetadata(metadata ?? {});

  await col.add({
    ids: [docId],
    embeddings: [embNorm],
    documents: [content],
    metadatas: [flatMeta],
  });

  return docId;
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
  const { k = 8, threshold = 0.7, types = null, sha256 = null, doc_type = null } = options;

  const col = await getCollection();
  const qNorm = l2Normalize(queryEmbedding);

  // Build ChromaDB where filter
  const whereConditions = [];
  if (types && types.length > 0) {
    whereConditions.push({ type: { $in: types } });
  }
  if (sha256) {
    whereConditions.push({ sha256: sha256 });
  }
  if (doc_type) {
    whereConditions.push({ doc_type: { $eq: doc_type } });
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
    // If collection is empty, ChromaDB may throw
    if (err.message?.includes("not enough") || err.message?.includes("empty")) {
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
  l2Normalize,
};
