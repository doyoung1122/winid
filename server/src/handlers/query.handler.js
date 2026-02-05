import { runRag } from "../rag/chain.js";

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

export default { handleQuery, handleQueryGet };
