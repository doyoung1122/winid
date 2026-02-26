import { runRag, runRagStream } from "../rag/chain.js";
import { searchByQuery, calculateTop3Avg } from "../services/vector.service.js";
import {
  isOwner, getAiHistory, saveAiMessages,
  countAiTurns, getHistoryForCompression, saveSummaryAndPrune,
  COMPRESS_THRESHOLD,
} from "../../../db/chat.js";
import { callLLMStream } from "../services/llm.service.js";

/**
 * 백그라운드 히스토리 압축
 * 응답 후 비동기로 실행 — 응답 속도에 영향 없음
 */
async function compressHistoryIfNeeded(roomId) {
  try {
    const total = await countAiTurns(roomId);
    if (total <= COMPRESS_THRESHOLD) return;

    const { turns, ids } = await getHistoryForCompression(roomId);
    if (!turns.length) return;

    const historyText = turns
      .map((t, i) => `[${i + 1}턴]\n사용자: ${t.user}\nAI: ${t.assistant}`)
      .join("\n\n");

    const summary = await callLLMStream(
      [
        {
          role: "system",
          content: "당신은 화재감식 전문 AI입니다. 아래 대화를 핵심 정보 중심으로 간결하게 요약하세요. 감식 결론, 주요 원인, 중요 수치는 반드시 포함하세요.",
        },
        {
          role: "user",
          content: `다음 화재감식 대화를 요약해주세요:\n\n${historyText}`,
        },
      ],
      { maxTokens: 800, temperature: 0.1 }
    );

    await saveSummaryAndPrune(roomId, summary, ids);
    console.log(`[chat] room ${roomId} 히스토리 압축 완료 (${ids.length}행 → 요약)`);
  } catch (e) {
    console.warn("[chat] compressHistory failed:", e?.message);
  }
}

/**
 * POST /query handler
 * @param {import('express').Request} req
 * @param {import('express').Response} res
 */
export async function handleQuery(req, res) {
  try {
    const {
      question,
      room_id,
      mem_id,
      match_count = 5,
      history = [],
      max_new_tokens = 3000,
      temperature = 0.2,
      top_p = 0.9,
    } = req.body || {};

    if (!question) {
      return res.status(400).json({ ok: false, error: "question required" });
    }
    if (String(question).length > 8000) {
      return res.status(413).json({ ok: false, error: "question too long" });
    }

    // 채팅방 모드: room_id + mem_id 제공 시 DB 히스토리 사용
    let hist;
    if (room_id && mem_id) {
      const owner = await isOwner(room_id, mem_id);
      if (!owner) return res.status(403).json({ ok: false, error: "권한 없음" });
      hist = await getAiHistory(room_id);
    } else {
      hist = Array.isArray(history) ? [...history] : [];
      if (hist.length > 50) hist.splice(0, hist.length - 50);
    }

    const result = await runRag({
      question,
      history: hist,
      match_count,
      max_new_tokens,
      temperature,
      top_p,
    });

    if (room_id && mem_id) {
      await saveAiMessages(room_id, mem_id, question, result.answer).catch((e) =>
        console.warn("[chat] saveAiMessages failed:", e?.message)
      );
      // 백그라운드 압축 (응답 후 비동기)
      compressHistoryIfNeeded(room_id);
    }

    return res.json({
      ok: true,
      mode: "json",
      ...result,
    });
  } catch (e) {
    console.error("/query error:", e);
    if (!res.headersSent) {
      return res.status(500).json({ ok: false, error: e?.message || "query failed" });
    }
  }
}

/**
 * GET /query/:question handler
 * @param {import('express').Request} req
 * @param {import('express').Response} res
 */
export async function handleQueryGet(req, res) {
  try {
    const question = req.params.question || "";

    if (!question) {
      return res.status(400).json({ ok: false, error: "question required" });
    }

    const result = await runRag({
      question,
      history: [],
      match_count: 5,
      max_new_tokens: 700,
      temperature: 0.2,
      top_p: 0.9,
    });

    return res.json({
      ok: true,
      mode: "json",
      ...result,
    });
  } catch (e) {
    console.error("/query/:question error:", e);
    if (!res.headersSent) {
      return res.status(500).json({ ok: false, error: e?.message || "query failed" });
    }
  }
}

/**
 * POST /query/compare handler
 * Compares MySQL and ChromaDB search results for the same query.
 */
export async function handleQueryCompare(req, res) {
  try {
    const { question, match_count = 5 } = req.body || {};

    if (!question) {
      return res.status(400).json({ ok: false, error: "question required" });
    }

    const t0 = performance.now();
    const mysqlResult = await searchByQuery(question, { backend: "mysql", match_count });
    const mysqlTime = performance.now() - t0;

    const t1 = performance.now();
    const chromaResult = await searchByQuery(question, { backend: "chroma", match_count });
    const chromaTime = performance.now() - t1;

    const mysqlTop3Avg = calculateTop3Avg(mysqlResult.matches);
    const chromaTop3Avg = calculateTop3Avg(chromaResult.matches);

    // Find overlapping document IDs (by content match since IDs differ)
    const mysqlContents = new Set(mysqlResult.matches.map((m) => m.content?.slice(0, 100)));
    const overlap = chromaResult.matches.filter((m) =>
      mysqlContents.has(m.content?.slice(0, 100))
    ).length;

    return res.json({
      ok: true,
      question,
      mysql: {
        count: mysqlResult.matches.length,
        maxSim: mysqlResult.maxSim,
        top3Avg: mysqlTop3Avg,
        latency_ms: Math.round(mysqlTime * 100) / 100,
        matches: mysqlResult.matches.map((m) => ({
          id: m.id,
          similarity: m.similarity,
          type: m.metadata?.type,
          content_preview: m.content?.slice(0, 120),
        })),
      },
      chroma: {
        count: chromaResult.matches.length,
        maxSim: chromaResult.maxSim,
        top3Avg: chromaTop3Avg,
        latency_ms: Math.round(chromaTime * 100) / 100,
        matches: chromaResult.matches.map((m) => ({
          id: m.id,
          similarity: m.similarity,
          type: m.metadata?.type,
          content_preview: m.content?.slice(0, 120),
        })),
      },
      analysis: {
        overlap_count: overlap,
        sim_diff: Math.round(Math.abs(mysqlResult.maxSim - chromaResult.maxSim) * 10000) / 10000,
        faster: mysqlTime < chromaTime ? "mysql" : "chroma",
      },
    });
  } catch (e) {
    console.error("/query/compare error:", e);
    if (!res.headersSent) {
      return res.status(500).json({ ok: false, error: e?.message || "compare failed" });
    }
  }
}

/**
 * POST /query/stream handler - SSE streaming
 */
export async function handleQueryStream(req, res) {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  try {
    const {
      question,
      room_id,
      mem_id,
      history = [],
      max_new_tokens = 3000,
      temperature = 0.2,
      top_p = 0.9,
    } = req.body || {};

    if (!question) {
      res.write(`event: error\ndata: ${JSON.stringify({ error: "question required" })}\n\n`);
      return res.end();
    }
    if (String(question).length > 8000) {
      res.write(`event: error\ndata: ${JSON.stringify({ error: "question too long" })}\n\n`);
      return res.end();
    }

    // 채팅방 모드: room_id + mem_id 제공 시 DB 히스토리 사용
    let hist;
    if (room_id && mem_id) {
      const owner = await isOwner(room_id, mem_id);
      if (!owner) {
        res.write(`event: error\ndata: ${JSON.stringify({ error: "권한 없음" })}\n\n`);
        return res.end();
      }
      hist = await getAiHistory(room_id);
    } else {
      hist = Array.isArray(history) ? [...history] : [];
      if (hist.length > 50) hist.splice(0, hist.length - 50);
    }

    const heartbeat = setInterval(() => {
      if (!res.writableEnded) res.write(": ping\n\n");
    }, 10000);

    try {
      let fullAnswer = "";
      const { sources, rag_mode } = await runRagStream({
        question,
        history: hist,
        max_new_tokens,
        temperature,
        top_p,
        onToken: (token) => {
          fullAnswer += token;
          res.write(`data: ${JSON.stringify({ token })}\n\n`);
        },
      });

      if (room_id && mem_id) {
        await saveAiMessages(room_id, mem_id, question, fullAnswer).catch((e) =>
          console.warn("[chat] saveAiMessages failed:", e?.message)
        );
        // 백그라운드 압축 (응답 후 비동기)
        compressHistoryIfNeeded(room_id);
      }

      res.write(`event: done\ndata: ${JSON.stringify({ sources, rag_mode })}\n\n`);
      res.end();
    } finally {
      clearInterval(heartbeat);
    }
  } catch (e) {
    console.error("/query/stream error:", e);
    if (!res.writableEnded) {
      res.write(`event: error\ndata: ${JSON.stringify({ error: e?.message || "stream failed" })}\n\n`);
      res.end();
    }
  }
}

export default { handleQuery, handleQueryGet, handleQueryCompare, handleQueryStream };
