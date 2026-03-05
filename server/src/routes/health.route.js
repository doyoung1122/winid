import { Router } from "express";
import {
  EMB_URL,
  LLM_URL,
  RERANKER_URL,
  ALWAYS_UNSTRUCTURED,
  RENDER_PAGES,
  MAX_TABLE_ROWS_EMB,
  FAST_MODE,
  RERANK_ENABLED,
  RERANK_CANDIDATE_K,
  RERANK_TOP_K,
} from "../config/env.js";
import { checkRerankerHealth } from "../services/reranker.service.js";

const router = Router();

/**
 * GET /health
 * Health check endpoint
 */
router.get("/health", async (_, res) => {
  const rerankerHealth = RERANK_ENABLED ? await checkRerankerHealth() : { ok: false, status: "disabled" };
  res.json({
    ok: true,
    emb_url: EMB_URL,
    llm_url: LLM_URL,
    reranker_url: RERANKER_URL,
    rerank_enabled: RERANK_ENABLED,
    rerank_candidate_k: RERANK_CANDIDATE_K,
    rerank_top_k: RERANK_TOP_K,
    reranker_status: rerankerHealth.status,
    storage: "local:/assets",
    always_unstructured: ALWAYS_UNSTRUCTURED,
    render_pages: RENDER_PAGES,
    max_table_rows_emb: MAX_TABLE_ROWS_EMB,
    fast_mode: FAST_MODE,
  });
});

export default router;
