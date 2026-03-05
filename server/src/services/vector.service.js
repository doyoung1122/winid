import * as mysqlRepo from "../../../db/repo.js";
import * as chromaRepo from "../../../db/chroma.js";
import { getEmbedding } from "./embedding.service.js";
import { rerank } from "./reranker.service.js";
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
 * Insert a document+embedding into the specified backend(s).
 * When VECTOR_BACKEND is "both", inserts into MySQL and ChromaDB.
 */
export async function insertDocumentWithEmbedding(content, metadata, embedding, backendOverride) {
  const backend = backendOverride || VECTOR_BACKEND;

  if (backend === "both") {
    const [mysqlId, chromaId] = await Promise.all([
      mysqlRepo.insertDocumentWithEmbedding(content, metadata, embedding),
      chromaRepo.insertDocumentWithEmbedding(content, metadata, embedding),
    ]);
    return mysqlId; // Return MySQL ID as primary
  }

  const repo = getRepo(backend);
  return repo.insertDocumentWithEmbedding(content, metadata, embedding);
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

  // Search across different content types
  const textMatches = await repo.matchDocuments(qVec, {
    ...baseOpts,
    k: textCandidateK,
    types: ["pdf", "text", "office", "hwpx", "hwp"],
  });

  const tableMatches = await repo.matchDocuments(qVec, {
    ...baseOpts,
    k: TABLE_K,
    types: ["table"],
  });

  const imageMatches = await repo.matchDocuments(qVec, {
    ...baseOpts,
    k: IMAGE_K,
    types: ["image_caption"],
  });

  // 텍스트 청크만 rerank (표/이미지는 이미 별도 임베딩으로 정밀)
  const rerankedText = useRerank && textMatches.length > RERANK_TOP_K
    ? await rerank(queryText, textMatches, RERANK_TOP_K)
    : textMatches.slice(0, match_count || TEXT_K);

  const matches = [...rerankedText, ...tableMatches, ...imageMatches];
  const sims = matches.map((m) => m.rerank_score ?? m.similarity ?? 0);
  const maxSim = sims.length ? Math.max(...sims) : 0;

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

