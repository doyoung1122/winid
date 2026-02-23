import { runRag, runRagStream } from "../rag/chain.js";
import { searchByQuery, calculateTop3Avg } from "../services/vector.service.js";

/**
 * POST /query handler
 * @param {import('express').Request} req
 * @param {import('express').Response} res
 */
export async function handleQuery(req, res) {
  try {
    const {
      question,
      match_count = 5,
      history = [],
      max_new_tokens = 700,
      temperature = 0.2,
      top_p = 0.9,
    } = req.body || {};

    if (!question) {
      return res.status(400).json({ ok: false, error: "question required" });
    }
    if (String(question).length > 8000) {
      return res.status(413).json({ ok: false, error: "question too long" });
    }

    const hist = Array.isArray(history) ? [...history] : [];
    if (hist.length > 50) {
      hist.splice(0, hist.length - 50);
    }

    const result = await runRag({
      question,
      history: hist,
      match_count,
      max_new_tokens,
      temperature,
      top_p,
    });

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
      history = [],
      max_new_tokens = 700,
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

    const hist = Array.isArray(history) ? [...history] : [];
    if (hist.length > 50) hist.splice(0, hist.length - 50);

    const { sources, rag_mode } = await runRagStream({
      question,
      history: hist,
      max_new_tokens,
      temperature,
      top_p,
      onToken: (token) => {
        res.write(`data: ${JSON.stringify({ token })}\n\n`);
      },
    });

    res.write(`event: done\ndata: ${JSON.stringify({ sources, rag_mode })}\n\n`);
    res.end();
  } catch (e) {
    console.error("/query/stream error:", e);
    if (!res.writableEnded) {
      res.write(`event: error\ndata: ${JSON.stringify({ error: e?.message || "stream failed" })}\n\n`);
      res.end();
    }
  }
}

export default { handleQuery, handleQueryGet, handleQueryCompare, handleQueryStream };
