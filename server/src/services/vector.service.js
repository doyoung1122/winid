import * as mysqlRepo from "../../../db/repo.js";
import * as chromaRepo from "../../../db/chroma.js";
import { getEmbedding } from "./embedding.service.js";
import { rerank } from "./reranker.service.js";
import { searchBM25, addToBM25Index } from "./bm25.service.js";
import { RETRIEVE_MIN, TEXT_K, TABLE_K, IMAGE_K, VECTOR_BACKEND,
         RERANK_ENABLED, RERANK_CANDIDATE_K, RERANK_TOP_K } from "../config/env.js";

// MySQL-only exports (doc_assets / doc_tables are MySQL-only features)
export const { insertDocAsset, insertDocTable } = mysqlRepo;

/**
 * Get the repo module for the given backend name.
 */
function getRepo(backend) {
  if (backend === "chroma") return chromaRepo;
  return mysqlRepo;
}

/**
 * Reciprocal Rank Fusion: merges vector results and BM25 results.
 * RRF score = Σ 1/(rank + k), k=60 (standard default)
 * BM25-only results include their stored content.
 * @param {Array} vectorDocs - vector search results (with .id, .content, .metadata, .similarity)
 * @param {Array} bm25Docs   - BM25 search results (with .id, .content, .metadata, .score)
 * @param {number} rrfK      - RRF constant (default 60)
 * @returns {Array} merged results sorted by rrf_score desc
 */
function rrfMerge(vectorDocs, bm25Docs, rrfK = 60) {
  const map = new Map(); // id → { rrf_score, doc }

  vectorDocs.forEach((doc, i) => {
    map.set(doc.id, { rrf_score: 1 / (i + 1 + rrfK), doc });
  });

  bm25Docs.forEach((bm25Doc, i) => {
    const rrf = 1 / (i + 1 + rrfK);
    if (map.has(bm25Doc.id)) {
      const cur = map.get(bm25Doc.id);
      map.set(bm25Doc.id, { ...cur, rrf_score: cur.rrf_score + rrf });
    } else {
      // BM25-only result: include with stored content, similarity=0
      map.set(bm25Doc.id, {
        rrf_score: rrf,
        doc: { id: bm25Doc.id, content: bm25Doc.content, metadata: bm25Doc.metadata, similarity: 0 },
      });
    }
  });

  return [...map.values()]
    .sort((a, b) => b.rrf_score - a.rrf_score)
    .map(({ rrf_score, doc }) => ({ ...doc, rrf_score }));
}

/**
 * Insert a document+embedding into the specified backend(s).
 * When VECTOR_BACKEND is "both", inserts into MySQL and ChromaDB.
 */
export async function insertDocumentWithEmbedding(content, metadata, embedding, backendOverride) {
  const backend = backendOverride || VECTOR_BACKEND;
  const TEXT_TYPES = new Set(["pdf", "text", "office", "hwpx", "hwp"]);

  if (backend === "both") {
    const [mysqlId, chromaId] = await Promise.all([
      mysqlRepo.insertDocumentWithEmbedding(content, metadata, embedding),
      chromaRepo.insertDocumentWithEmbedding(content, metadata, embedding),
    ]);
    return mysqlId; // Return MySQL ID as primary
  }

  const repo = getRepo(backend);
  const docId = await repo.insertDocumentWithEmbedding(content, metadata, embedding);

  // Keep BM25 index up-to-date for text-type ChromaDB docs
  if (backend === "chroma" && TEXT_TYPES.has(metadata?.type)) {
    addToBM25Index(String(docId), content, metadata);
  }

  return docId;
}

/**
 * Load vector cache for the specified backend(s).
 */
export async function loadCache(backendOverride) {
  const backend = backendOverride || VECTOR_BACKEND;

  if (backend === "both") {
    await Promise.all([mysqlRepo.loadCache(), chromaRepo.loadCache()]);
    return;
  }

  const repo = getRepo(backend);
  await repo.loadCache();
}

/**
 * Match documents using the specified backend.
 */
export async function matchDocuments(queryEmbedding, options = {}) {
  const { backend = VECTOR_BACKEND, ...rest } = options;
  const repo = getRepo(backend === "both" ? "mysql" : backend);
  return repo.matchDocuments(queryEmbedding, rest);
}

/**
 * Search documents by query text
 * @param {string} queryText - Query text
 * @param {Object} options - Search options
 * @returns {Promise<Object>} Search results with matches
 */
export async function searchByQuery(queryText, options = {}) {
  const { match_count = TEXT_K, doc_sha = null, backend, rerank: useRerank = RERANK_ENABLED } = options;

  const effectiveBackend = backend || (VECTOR_BACKEND === "both" ? "mysql" : VECTOR_BACKEND);
  const repo = getRepo(effectiveBackend);

  // Get query embedding
  const qVec = await getEmbedding(queryText, "query");

  const baseOpts = {
    threshold: RETRIEVE_MIN,
    ...(doc_sha ? { sha256: doc_sha } : {}),
  };

  // rerank 활성화 시 텍스트 후보를 RERANK_CANDIDATE_K(20)으로 확장
  const textCandidateK = useRerank ? RERANK_CANDIDATE_K : (match_count || TEXT_K);

  // 병렬: 벡터 검색 (text/table/image) + BM25 키워드 검색 (text 전용)
  const [textMatches, tableMatches, imageMatches, bm25Results] = await Promise.all([
    repo.matchDocuments(qVec, {
      ...baseOpts,
      k: textCandidateK,
      types: ["pdf", "text", "office", "hwpx", "hwp"],
    }),
    repo.matchDocuments(qVec, { ...baseOpts, k: TABLE_K, types: ["table"] }),
    repo.matchDocuments(qVec, { ...baseOpts, k: IMAGE_K, types: ["image_caption"] }),
    // BM25는 ChromaDB 백엔드 전용 (MySQL 백엔드는 자체 in-memory 캐시로 충분)
    effectiveBackend === "chroma"
      ? searchBM25(queryText, RERANK_CANDIDATE_K).catch((e) => {
          console.warn("[BM25] search error:", e.message);
          return [];
        })
      : Promise.resolve([]),
  ]);

  // RRF 병합: 벡터 + BM25 결과 → 중복 제거 + 순위 합산
  let textCandidates;
  if (bm25Results.length > 0) {
    const merged = rrfMerge(textMatches, bm25Results);
    textCandidates = merged.slice(0, RERANK_CANDIDATE_K);
    console.log(`[BM25] vector=${textMatches.length} bm25=${bm25Results.length} merged=${textCandidates.length}`);
  } else {
    textCandidates = textMatches;
  }

  // 텍스트 청크만 rerank (표/이미지는 이미 별도 임베딩으로 정밀)
  const rerankedText = useRerank && textCandidates.length > RERANK_TOP_K
    ? await rerank(queryText, textCandidates, RERANK_TOP_K)
    : textCandidates.slice(0, match_count || TEXT_K);

  const matches = [...rerankedText, ...tableMatches, ...imageMatches];
  // maxSim: 코사인 유사도 기준 (USE_AS_CTX_MIN 임계값과 스케일 일치)
  // 리랭커/RRF 점수는 정렬용으로만 사용하고 임계값 비교엔 코사인 유사도 사용
  const maxSim = matches.reduce((m, r) => Math.max(m, r.similarity ?? 0), 0);

  return { qVec, matches, maxSim };
}

/**
 * Calculate top-3 average similarity
 * @param {Array} matches - Match results
 * @returns {number} Top-3 average similarity
 */
export function calculateTop3Avg(matches) {
  const sims = matches.map((m) => m.similarity ?? 0);
  const sorted = [...sims].sort((a, b) => b - a);
  const top3 = sorted.slice(0, 3);
  return top3.length > 0 ? top3.reduce((a, b) => a + b, 0) / top3.length : 0;
}

/**
 * Build context string from matches
 * @param {Array} matches - Match results
 * @param {number} maxCtx - Maximum context length
 * @returns {Object} Context string and sources
 */
export function buildContext(matches, maxCtx = 4000) {
  let ctx = "";
  let length = 0;
  const sources = [];

  for (const m of matches) {
    let t = m.content?.trim();
    if (!t) continue;
    if (t.length > 1600) t = t.slice(0, 800) + "\n...\n" + t.slice(-800);
    if (length + t.length > maxCtx) break;

    const meta = typeof m.metadata === "string" ? JSON.parse(m.metadata) : m.metadata || {};
    const filename = (meta.source || meta.filepath || "").split(/[\\/]/).pop();
    const type = meta.type || "unknown";
    const page = meta.page || "N/A";

    ctx += `<document source="${filename}" page="${page}" type="${type}">
${t}
</document>\n\n`;

    length += t.length;
    sources.push({ filename, similarity: m.similarity, page: meta.page, type });
  }

  return { context: ctx.trim(), sources };
}

