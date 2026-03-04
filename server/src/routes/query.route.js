import { Router } from "express";
import { handleQuery, handleQueryGet, handleQueryCompare, handleQueryStream, handleGetHistory } from "../handlers/query.handler.js";

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
 * GET /history?room_id=X&mem_id=Y
 * 채팅방 AI 대화 히스토리 조회 (화면 표시용)
 */
router.get("/history", handleGetHistory);

/**
 * GET /query/:question
 * Simple GET query endpoint
 */
router.get("/query/:question", handleQueryGet);

export default router;
