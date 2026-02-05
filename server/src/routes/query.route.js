import { Router } from "express";
import { handleQuery, handleQueryGet } from "../handlers/query.handler.js";

const router = Router();

/**
 * POST /query
 * RAG query endpoint
 */
router.post("/query", handleQuery);

/**
 * GET /query/:question
 * Simple GET query endpoint
 */
router.get("/query/:question", handleQueryGet);

export default router;
