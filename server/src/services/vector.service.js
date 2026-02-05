import {
  matchDocuments,
  insertDocumentWithEmbedding,
  insertDocAsset,
  insertDocTable,
  loadCache,
} from "../../../db/repo.js";
import { getEmbedding } from "./embedding.service.js";
import { RETRIEVE_MIN, TEXT_K, TABLE_K, IMAGE_K } from "../config/env.js";

/**
 * Search documents by query text
 * @param {string} queryText - Query text
 * @param {Object} options - Search options
 * @returns {Promise<Object>} Search results with matches
 */
export async function searchByQuery(queryText, options = {}) {
  const { match_count = TEXT_K, doc_sha = null } = options;

  // Get query embedding
  const qVec = await getEmbedding(queryText, "query");

  const baseOpts = {
    threshold: RETRIEVE_MIN,
    ...(doc_sha ? { sha256: doc_sha } : {}),
  };

  // Search across different content types
  const textMatches = await matchDocuments(qVec, {
    ...baseOpts,
    k: match_count || TEXT_K,
    types: ["pdf", "text", "office", "hwpx", "hwp"],
  });

  const tableMatches = await matchDocuments(qVec, {
    ...baseOpts,
    k: TABLE_K,
    types: ["table_row"],
  });

  const imageMatches = await matchDocuments(qVec, {
    ...baseOpts,
    k: IMAGE_K,
    types: ["image_caption"],
  });

  const matches = [...textMatches, ...tableMatches, ...imageMatches];
  const sims = matches.map((m) => m.similarity ?? 0);
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
    const filename = (meta.filepath || "").split(/[\\/]/).pop();
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

// Re-export db functions for convenience
export {
  matchDocuments,
  insertDocumentWithEmbedding,
  insertDocAsset,
  insertDocTable,
  loadCache,
};

export default {
  searchByQuery,
  calculateTop3Avg,
  buildContext,
  matchDocuments,
  insertDocumentWithEmbedding,
  insertDocAsset,
  insertDocTable,
  loadCache,
};
