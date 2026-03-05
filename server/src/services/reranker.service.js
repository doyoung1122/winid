import { RERANKER_URL, RERANKER_MODEL } from "../config/env.js";

/**
 * Re-rank documents using cross-encoder model.
 * Falls back to original order if reranker service is unavailable.
 *
 * @param {string} query - Search query
 * @param {Array<{content: string, [key: string]: any}>} documents - Documents to rerank
 * @param {number} topK - Number of top documents to return
 * @returns {Promise<Array>} Reranked documents (topK or fewer)
 */
export async function rerank(query, documents, topK = 5) {
  if (!documents || documents.length === 0) return [];
  if (documents.length <= topK) return documents;

  const texts = documents.map((d) => d.content?.slice(0, 2000) ?? "");

  try {
    const res = await fetch(`${RERANKER_URL}/v1/rerank`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: RERANKER_MODEL,
        query,
        documents: texts,
      }),
      signal: AbortSignal.timeout(8000),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`rerank failed ${res.status}: ${errText}`);
    }

    const j = await res.json();
    // 응답: { results: [{ index: number, relevance_score: number }, ...] }
    const ranked = j.results
      .sort((a, b) => b.relevance_score - a.relevance_score)
      .slice(0, topK)
      .map((r) => ({
        ...documents[r.index],
        rerank_score: r.relevance_score,
      }));

    console.log(
      `[rerank] ${documents.length} → ${ranked.length} | top score: ${ranked[0]?.rerank_score?.toFixed(4)}`
    );
    return ranked;
  } catch (e) {
    // reranker 서비스 다운 시 원본 cosine sim 순서 유지
    console.warn("[rerank] fallback to original order:", e.message);
    return documents.slice(0, topK);
  }
}

/**
 * Health check for reranker service
 */
export async function checkRerankerHealth() {
  try {
    const res = await fetch(`${RERANKER_URL}/health`, {
      signal: AbortSignal.timeout(3000),
    });
    return { ok: res.ok, status: res.ok ? "healthy" : `status ${res.status}` };
  } catch (e) {
    return { ok: false, status: `error: ${e.message}` };
  }
}

export default { rerank, checkRerankerHealth };
