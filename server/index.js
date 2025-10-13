require("dotenv").config();
const express = require("express");
const cors = require("cors");
const multer = require("multer");
const pdfParse = require("pdf-parse");
const fetch = require("node-fetch");
const { createClient } = require("@supabase/supabase-js");
const { chunkTextTokens } = require("./chunk.js");

// =========================
// 환경변수
// =========================
const PORT = Number(process.env.PORT || 8000);
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE = process.env.SUPABASE_SERVICE_ROLE;
const EMB_URL = (process.env.EMB_URL || "http://127.0.0.1:8001").replace(/\/$/, "");
const LLM_URL = (process.env.LLM_URL || "http://127.0.0.1:8002").replace(/\/$/, "");

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE);
const app = express();

// CORS
const allowedOrigins = [
  "http://localhost:5173",
  "http://127.0.0.1:5173",
  "http://192.168.0.7:5173",
  "http://218.158.62.155:5173",
];

app.use(cors({
  origin(origin, cb) {
    if (!origin || allowedOrigins.includes(origin)) return cb(null, true);
    console.log("❌ CORS blocked:", origin);
    cb(new Error("Not allowed by CORS"));
  },
  credentials: true,
}));

app.use(express.json({ limit: "2mb" }));

// 업로드(메모리 저장, 100MB 제한)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 100 * 1024 * 1024 },
});

// =========================
// 유틸: 간단 동시성 리미터 (p-limit 대체)
// =========================
function createLimiter(concurrency = 4) {
  let active = 0;
  const q = [];
  const run = () => {
    if (active >= concurrency || q.length === 0) return;
    active++;
    const { fn, res, rej } = q.shift();
    Promise.resolve().then(fn).then(
      (v) => { active--; res(v); run(); },
      (e) => { active--; rej(e); run(); }
    );
  };
  return (fn) => new Promise((res, rej) => { q.push({ fn, res, rej }); process.nextTick(run); });
}

// =========================
// 유틸: 텍스트 후처리 (한글/합자/하이픈/널문자 정리)
// =========================
function cleanText(s) {
  if (!s) return "";
  return s
    .replace(/\u0000/g, "")                 
    .replace(/\r/g, "\n")
    .normalize("NFKC")
    .replace(/ﬁ/g, "fi")
    .replace(/ﬂ/g, "fl")
    .replace(/ﬃ/g, "ffi")
    .replace(/ﬄ/g, "ffl")
    .replace(/([A-Za-z0-9])-\n([A-Za-z0-9])/g, "$1$2")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

// =========================
// 헬스체크
// =========================
app.get("/health", (_, res) => {
  res.json({
    ok: true,
    emb_url: EMB_URL,
    llm_url: LLM_URL,
    supabase: !!SUPABASE_URL && !!SUPABASE_SERVICE_ROLE,
  });
});

// =========================
// 업로드: PDF/TXT/MD → 텍스트 정제 → 청킹 → 임베딩 → Supabase 저장
// =========================
app.post("/upload", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "file required" });

    const mimetype = req.file.mimetype || "";
    const filepath = req.file.originalname || "unknown";

    let text = "";

    if (mimetype.includes("pdf") || filepath.toLowerCase().endsWith(".pdf")) {
      // 1) pdf-parse로 추출
      const pdf = await pdfParse(req.file.buffer);
      // 2) 널문자 제거/정규화 포함 후처리
      const raw = cleanText(pdf.text || "");
      // (페이지 구분이 꼭 필요하면 아래 같이 간단히 마크업)
      // const pages = raw.split(/\f/g); // pdf-parse는 form-feed가 없을 때도 많음
      // text = pages.map((t,i)=>`# Page ${i+1}\n${t}`).join("\n\n--- page ---\n\n");
      text = raw;
    } else if (mimetype.includes("text") || /\.(txt|md)$/i.test(filepath)) {
      text = cleanText(req.file.buffer.toString("utf8"));
    } else {
      return res.status(400).json({ error: "Only PDF/TXT/MD supported" });
    }

    if (!text) return res.status(400).json({ error: "empty text" });

    // 텍스트 → 청킹
    const chunks = chunkTextTokens(text, 800, 120); // [{text, startTok, endTok}, ...]
    if (!chunks.length) return res.status(400).json({ error: "no chunks" });

    // 동시성 제한
    const limit = createLimiter(4);
    let inserted = 0;

    await Promise.all(
      chunks.map((c, idx) =>
        limit(async () => {
          // ====== 임베딩 (로컬 BGE-M3 서비스) ======
          // 임베딩 서비스 응답 형태에 따라 아래 파싱 조정:
          // - { embedding: number[] } 또는
          // - { data: [{ embedding: number[] }] }
          const embRes = await fetch(`${EMB_URL}/embed`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ input: c.text }),
          });
          if (!embRes.ok) {
            const t = await embRes.text().catch(() => "");
            throw new Error(`embedding failed: HTTP ${embRes.status} ${t}`);
          }
          const embJson = await embRes.json();
          const vector =
            embJson?.embedding ||
            embJson?.data?.[0]?.embedding ||
            embJson?.data ||
            null;
          if (!Array.isArray(vector)) throw new Error("invalid embedding payload");

          // ====== DB 저장 (문서 + 벡터) ======
          const { error } = await supabase.rpc("insert_document_with_embedding_v2", {
            p_content: c.text,
            p_metadata: { filepath, chunk_index: idx, startTok: c.startTok, endTok: c.endTok },
            p_embedding: vector,
          });
          if (error) throw error;

          inserted++;
        })
      )
    );

    res.json({ ok: true, chunks: inserted });
  } catch (e) {
    console.error("/upload error:", e);
    res.status(500).json({ error: e?.message || "server error" });
  }
});

// =========================
// 질의: 임베딩 검색 → 컨텍스트 구성 → LLM(SSE) 프록시
// =========================
app.post("/query", async (req, res) => {
  try {
    const {
      question,
      match_count = 3,
      threshold = 0.65,
      history = [], // [{role:'user'|'assistant', content:string}] 옵션
    } = req.body || {};
    if (!question) return res.status(400).json({ error: "question required" });

    // 1) 질문 임베딩
    const embRes = await fetch(`${EMB_URL}/embed`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: question }),
    });
    if (!embRes.ok) throw new Error(`embedding(q) failed: HTTP ${embRes.status}`);
    const embJson = await embRes.json();
    const qVec = embJson?.embedding || embJson?.data?.[0]?.embedding || embJson?.data;
    if (!Array.isArray(qVec)) throw new Error("invalid embedding payload for question");

    // 2) 유사도 검색
    const { data: matches, error: matchErr } = await supabase.rpc("match_documents_v2", {
      query_embedding: qVec,
      match_count,
      similarity_threshold: threshold,
    });
    if (matchErr) throw matchErr;

    // 3) 컨텍스트 구성
    const MAX_CTX = 4000;
    let total = 0;
    const ctxParts = [];
    for (const m of matches || []) {
      let t = (m.content || "").replace(/\u0000/g, "").trim();
      if (t.length > 1500) t = t.slice(0, 700) + "\n...\n" + t.slice(-700);
      if (total + t.length > MAX_CTX) break;
      ctxParts.push(t);
      total += t.length;
    }
    const context = ctxParts.join("\n\n---\n\n");

    // 4) 클라이언트로 SSE 헤더
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");

    // 5) LLM 서비스(/chat)로 SSE 프록시
    const llmRes = await fetch(`${LLM_URL.replace(/\/$/, "")}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
      body: JSON.stringify({
        context,
        question,
        history,
        system: "",
        max_new_tokens: 128,
      }),
    });

    if (!llmRes.ok || !llmRes.body) {
      const t = await llmRes.text().catch(() => "");
      throw new Error(`LLM failed: HTTP ${llmRes.status} ${t}`);
    }

    // llm_service가 보내는 SSE("data: ...\n\n")를 그대로 릴레이
    llmRes.body.on("data", (chunk) => {
      res.write(chunk);
    });
    llmRes.body.on("end", () => {
      res.end();
    });
    llmRes.body.on("error", (err) => {
      console.error("🔥 LLM stream error:", err);
      try { res.end(); } catch {}
    });

    // 클라이언트가 연결 끊으면 LLM 스트림 정리
    req.on("close", () => {
      try { llmRes.body?.destroy?.(); } catch {}
    });
  } catch (e) {
    console.error("🔥 /query error:", e);
    if (!res.headersSent) res.status(500).json({ error: e?.message || "query failed" });
    else res.end();
  }
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server running on http://0.0.0.0:${PORT}`);
  console.log(`   - EMB_URL = ${EMB_URL}`);
  console.log(`   - LLM_URL = ${LLM_URL}`);
  console.log(`   - CORS allow = ${allowedOrigins.join(", ")}`);
});