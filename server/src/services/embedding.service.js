import { EMB_URL, EMB_MODEL } from "../config/env.js";

/**
 * Get embedding for a single text
 * @param {string} text - Text to embed
 * @param {string} mode - Embedding mode (passage/query)
 * @returns {Promise<number[]>} Embedding vector
 */
export async function getEmbedding(text, mode = "passage") {
  const r = await fetch(`${EMB_URL}/v1/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: EMB_MODEL,
      input: text,
    }),
  });

  if (!r.ok) {
    const errText = await r.text();
    throw new Error(`embedding failed ${r.status}: ${errText}`);
  }

  const j = await r.json();
  // OpenAI 응답 포맷: { data: [{ embedding: [...] }] }
  return j.data[0].embedding;
}

/**
 * Get embeddings for multiple texts in batch
 * @param {string[]} texts - Array of texts to embed
 * @param {string} mode - Embedding mode (passage/query)
 * @returns {Promise<number[][]>} Array of embedding vectors
 */
export async function getEmbeddingsBatch(texts, mode = "passage") {
  if (!Array.isArray(texts) || texts.length === 0) return [];

  const r = await fetch(`${EMB_URL}/v1/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: EMB_MODEL,
      input: texts,
    }),
  });

  if (!r.ok) {
    const errText = await r.text();
    throw new Error(`embedding(batch) failed ${r.status}: ${errText}`);
  }

  const j = await r.json();
  // 순서대로 벡터 추출
  return j.data.map((d) => d.embedding);
}

/**
 * Health check for embedding service
 * @returns {Promise<{ok: boolean, status?: string}>}
 */
export async function checkEmbeddingHealth() {
  try {
    const res = await fetch(`${EMB_URL}/health`, { method: "GET" });
    return {
      ok: res.ok,
      status: res.ok ? "healthy" : `status ${res.status}`,
    };
  } catch (e) {
    return {
      ok: false,
      status: `error: ${e.message}`,
    };
  }
}

export default {
  getEmbedding,
  getEmbeddingsBatch,
  checkEmbeddingHealth,
};
