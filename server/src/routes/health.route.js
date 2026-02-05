import { Router } from "express";
import {
  EMB_URL,
  LLM_URL,
  ALWAYS_UNSTRUCTURED,
  RENDER_PAGES,
  MAX_TABLE_ROWS_EMB,
  FAST_MODE,
} from "../config/env.js";

const router = Router();

/**
 * GET /health
 * Health check endpoint
 */
router.get("/health", (_, res) => {
  res.json({
    ok: true,
    emb_url: EMB_URL,
    llm_url: LLM_URL,
    storage: "local:/assets",
    always_unstructured: ALWAYS_UNSTRUCTURED,
    render_pages: RENDER_PAGES,
    max_table_rows_emb: MAX_TABLE_ROWS_EMB,
    fast_mode: FAST_MODE,
  });
});

export default router;
