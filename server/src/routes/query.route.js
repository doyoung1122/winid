import { Router } from "express";
import { handleQuery, handleQueryGet, handleQueryCompare, handleQueryStream } from "../handlers/query.handler.js";

const router = Router();

/**
 * POST /query
 * RAG query endpoint
 */
router.post("/query", handleQuery);

/**
 * POST /query/compare
 * Compare MySQL vs ChromaDB search results
 */
router.post("/query/compare", handleQueryCompare);

/**
 * POST /query/stream
 * SSE streaming query endpoint
 */
router.post("/query/stream", handleQueryStream);

/**
 * GET /query/:question
 * Simple GET query endpoint
 */
router.get("/query/:question", handleQueryGet);

export default router;
