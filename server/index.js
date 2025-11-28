require("dotenv").config();

const express = require("express");
const cors = require("cors");
const multer = require("multer");
const pdfParse = require("pdf-parse");
const fetch = require("node-fetch");
const { fromBuffer } = require("pdf2pic");
const path = require("path");
const fs = require("fs");
const fsp = require("fs/promises");
const crypto = require("crypto");
const iconv = require("iconv-lite");
const chardet = require("chardet");
const JSZip = require("jszip");
const { parseStringPromise } = require("xml2js");
const { spawn } = require("child_process");
const { JSDOM } = require("jsdom");

const {
  insertDocumentWithEmbedding,
  insertDocAsset,
  insertDocTable,
  matchDocuments,
} = require("../db/repo.js");


const { chunkTextTokens } = require("./chunk.js");

// ===== 임베딩/정규화 헬퍼 =====
async function getEmbedding(text, mode = "passage") {
  const r = await fetch(`${EMB_URL}/embed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input: text, mode }),
  });
  if (!r.ok) throw new Error(`embedding failed ${r.status}`);
  const j = await r.json();
  const vec =
    Array.isArray(j.embedding) ? j.embedding :
    (j?.data?.[0]?.embedding && Array.isArray(j.data[0].embedding)) ? j.data[0].embedding :
    (Array.isArray(j.data) ? j.data : null);
  if (!Array.isArray(vec)) throw new Error("invalid embedding payload");
  return vec;
}

// 숫자/단위 정규화(필터/정렬 대비)
function normalizeNumber(v) {
  if (v == null) return null;
  const s = String(v);
  const m = s.match(/^([\d.,+-]+)\s*([A-Za-z%]*)$/);
  if (!m) return { raw: s };
  const num = parseFloat(m[1].replace(/,/g, ""));
  const unit = m[2] || "";
  return { value: isNaN(num) ? null : num, unit, raw: s };
}

// 행(Row) 내용을 자연어로 문장화 → 검색 친화
function rowToSentence(tableTitle, headers, row) {
  const parts = headers.map((h, i) => `${h}=${row[i] ?? ""}`);
  return `Table: ${tableTitle} | ${parts.join("; ")}`;
}

// =========================
// 환경변수
// =========================
const PORT = Number(process.env.PORT || 8000);
const EMB_URL = (process.env.EMB_URL || "http://127.0.0.1:8001").replace(/\/$/, "");
const LLM_URL = (process.env.LLM_URL || "http://127.0.0.1:8002").replace(/\/$/, "");
//onst VISION_URL = (process.env.VISION_URL || "http://127.0.0.1:8003").replace(/\/$/, ""); // 🔒 llama3.2 vision 비활성
const HWP2TXT_EXE = process.env.HWP2TXT_EXE || "";
const PUBLIC_BASE = (process.env.PUBLIC_BASE || `http://127.0.0.1:${PORT}`).replace(/\/$/, "");

// === 옵션 A 플래그 ===
// true면 PDF/Office는 무조건 Unstructured로 텍스트+표를 일괄 추출
const ALWAYS_UNSTRUCTURED = String(process.env.ALWAYS_UNSTRUCTURED || "true") === "true";

// 표/캡션 처리 한도(운영비/성능 조절) — 필요 시 사용
const MAX_TABLE_ROWS_EMB = Number(process.env.MAX_TABLE_ROWS_EMB || 500);
const MAX_CAPTION_PAGES = Number(process.env.MAX_CAPTION_PAGES || 10);

const app = express();

app.use(cors());
app.use(express.json({ limit: "10mb" }));

const PUBLIC_DIR = path.join(__dirname, "public");
const ASSET_DIR = path.join(PUBLIC_DIR, "doc-assets");
if (!fs.existsSync(ASSET_DIR)) fs.mkdirSync(ASSET_DIR, { recursive: true });
app.use("/assets", express.static(PUBLIC_DIR, { maxAge: "1y" }));

// =========================
// 업로드(메모리 저장, 100MB 제한)
// =========================
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 100 * 1024 * 1024 },
});

// =========================
// 동시성 리미터
// =========================
function createLimiter(concurrency = 4) {
  let active = 0;
  const q = [];
  const run = () => {
    if (active >= concurrency || q.length === 0) return;
    active++;
    const { fn, res, rej } = q.shift();
    Promise.resolve()
      .then(fn)
      .then(
        (v) => {
          active--;
          res(v);
          run();
        },
        (e) => {
          active--;
          rej(e);
          run();
        }
      );
  };
  return (fn) =>
    new Promise((res, rej) => {
      q.push({ fn, res, rej });
      process.nextTick(run);
    });
}

// =========================
// 파일 저장 유틸
// =========================
const UPLOAD_DIR = path.join(__dirname, "uploads");
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

function safeBasename(name = "file") {
  return name.replace(/[^\w.\-가-힣]+/g, "_").slice(0, 100);
}
function yyyymmdd(d = new Date()) {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${y}/${m}/${dd}`;
}
async function saveOriginalFile(buffer, originalName) {
  const dayDir = path.join(UPLOAD_DIR, yyyymmdd());
  await fsp.mkdir(dayDir, { recursive: true });
  const sha = crypto.createHash("sha256").update(buffer).digest("hex");
  const ext = path.extname(originalName) || ".bin";
  const base = safeBasename(path.basename(originalName, ext)) || "file";
  const storedName = `${sha.slice(0, 8)}_${Date.now()}_${base}${ext}`;
  const absPath = path.join(dayDir, storedName);
  await fsp.writeFile(absPath, buffer);
  const relPath = path.relative(__dirname, absPath).replace(/\\/g, "/");
  return { absPath, relPath, sha };
}

// =========================
// 텍스트 정리
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
// Unstructured: ocr_once.py
// =========================
async function extractWithUnstructuredOnce(fileBuffer, filename) {
  const TMP_DIR = path.join(UPLOAD_DIR, "tmp");
  await fsp.mkdir(TMP_DIR, { recursive: true });
  const ext = path.extname(filename) || ".bin";
  const base = safeBasename(path.basename(filename, ext)) || "file";
  const tmpPath = path.join(TMP_DIR, `${Date.now()}_${base}${ext}`);
  await fsp.writeFile(tmpPath, fileBuffer);

  const pyPath = path.join(__dirname, "ocr_once.py");
  const out = await new Promise((resolve, reject) => {
    const PY = process.env.UNSTRUCT_PY
      || (process.platform === "win32"
          ? "C:\\Users\\user\\anaconda3\\envs\\unstruct\\python.exe"
          : "python3");

    const p = spawn(PY, [pyPath, tmpPath], {
      stdio: ["ignore", "pipe", "pipe"],
      env: {
        ...process.env,
        PYTHONUTF8: "1",
        LANG: "ko_KR.UTF-8",
        LC_ALL: "ko_KR.UTF-8",
      },
    });

    let stdout = "",
      stderr = "";
    p.stdout.on("data", (d) => (stdout += d.toString()));
    p.stderr.on("data", (d) => (stderr += d.toString()));
    p.on("error", reject);
    p.on("close", (code) => {
      if (code !== 0) return reject(new Error(`unstructured exit ${code}: ${stderr || stdout}`));
      resolve(stdout);
    });
  }).finally(async () => {
    try {
      await fsp.unlink(tmpPath);
    } catch {}
  });

  let parsed;
  try {
    parsed = JSON.parse(out);
  } catch (e) {
    throw new Error(`unstructured output parse error: ${e.message}`);
  }
  if (!parsed?.ok) throw new Error(parsed?.error || "unstructured returned ok=false");
  return parsed; // { ok: true, text, tables }
}

// =========================
// HWPX 추출 (ZIP+XML)
// =========================
async function extractHwpx(buffer) {
  const zip = await JSZip.loadAsync(buffer);
  let sectionEntries = Object.keys(zip.files)
    .filter((n) => /^Contents\/section\d+\.xml$/i.test(n))
    .sort();

  if (sectionEntries.length === 0) {
    sectionEntries = Object.keys(zip.files).filter((n) => n.toLowerCase().endsWith(".xml"));
  }

  const texts = [];
  const tables = [];

  const crawl = (node) => {
    if (node == null || typeof node !== "object") return;
    if (typeof node._ === "string") texts.push(node._);
    for (const k of Object.keys(node)) {
      const v = node[k];
      if (Array.isArray(v)) v.forEach(crawl);
      else if (typeof v === "object") crawl(v);
    }
  };
  const findTables = (node, entry) => {
    if (!node || typeof node !== "object") return;
    for (const k of Object.keys(node)) {
      const v = node[k];
      if (/tbl$/i.test(k) && typeof v === "object") tables.push({ entry, type: "table" });
      if (Array.isArray(v)) v.forEach((x) => findTables(x, entry));
      else if (typeof v === "object") findTables(v, entry);
    }
  };

  for (const entry of sectionEntries) {
    const xmlStr = await zip.files[entry].async("string");
    const xml = await parseStringPromise(xmlStr, { explicitArray: true, preserveChildrenOrder: true });
    crawl(xml);
    findTables(xml, entry);
  }

  const text = cleanText(texts.join("\n"));
  return { text, tables };
}

// =========================
// HWP 변환기 훅(옵션)
// =========================
async function convertHwpToTxtViaCli(buffer, filename) {
  if (!HWP2TXT_EXE || !fs.existsSync(HWP2TXT_EXE)) {
    throw new Error("HWP converter not configured (set HWP2TXT_EXE).");
  }
  const TMP_DIR = path.join(UPLOAD_DIR, "tmp");
  await fsp.mkdir(TMP_DIR, { recursive: true });
  const tmpIn = path.join(TMP_DIR, `${Date.now()}_${safeBasename(filename)}`);
  const tmpOut = tmpIn + ".txt";
  await fsp.writeFile(tmpIn, buffer);

  try {
    await new Promise((resolve, reject) => {
      const p = spawn(HWP2TXT_EXE, [tmpIn, tmpOut], { stdio: ["ignore", "pipe", "pipe"] });
      let stderr = "";
      p.stderr.on("data", (d) => (stderr += d.toString()));
      p.on("error", reject);
      p.on("close", (code) => {
        if (code !== 0) return reject(new Error(`hwp2txt exit ${code}: ${stderr}`));
        resolve();
      });
    });
    const txt = await fsp.readFile(tmpOut, "utf8");
    return cleanText(txt);
  } finally {
    try {
      await fsp.unlink(tmpIn);
    } catch {}
    try {
      await fsp.unlink(tmpOut);
    } catch {}
  }
}

// =========================
// 표 정규화 유틸 (unstructured table meta → header/rows/tsv/md/html)
// =========================
function sanitizeCell(s = "") {
  return String(s || "").replace(/\s+/g, " ").trim();
}
function toMarkdownTable(header, rows) {
  if (!header?.length) return "";
  const sep = header.map(() => "---");
  const lines = [
    `| ${header.join(" | ")} |`,
    `| ${sep.join(" | ")} |`,
    ...rows.slice(0, 30).map((r) => `| ${r.map(sanitizeCell).join(" | ")} |`),
  ];
  return lines.join("\n");
}
function normalizeTableMeta(t) {
  const html = t?.html || t?.metadata?.text_as_html || null;
  let header = [],
    rows = [];
  if (html) {
    const dom = new JSDOM(html);
    const $rows = [...dom.window.document.querySelectorAll("tr")];
    const grid = $rows.map((tr) =>
      [...tr.querySelectorAll("th,td")].map((td) => (td.textContent || "").trim())
    );
    header = grid[0] || [];
    rows = grid.slice(1);
  } else if (Array.isArray(t?.rows) || Array.isArray(t?.header)) {
    const arr = Array.isArray(t?.rows) ? t.rows : [];
    header = Array.isArray(t?.header) ? t.header : (arr[0] || []).map((_, i) => `col_${i + 1}`);
    rows = arr.length ? arr : [];
  } else if (Array.isArray(t?.preview_rows)) {
    const grid = t.preview_rows.map(r => (Array.isArray(r) ? r.map(sanitizeCell) : [sanitizeCell(String(r))]));
    header = grid[0] || [];
    rows   = grid.slice(1);
   }
  const tsv = [header, ...rows].map((r) => r.map(sanitizeCell).join("\t")).join("\n");
  const md = toMarkdownTable(header, rows);
  const n_rows_hint = Number.isFinite(t?.n_rows) ? t.n_rows : undefined;
  const n_cols_hint = Number.isFinite(t?.n_cols) ? t.n_cols : undefined;
  return { header, rows, tsv, md, html, n_rows_hint, n_cols_hint };
}

// =========================
// 헬스체크
// =========================
app.get("/health", (_, res) => {
  res.json({
    ok: true,
    emb_url: EMB_URL,
    llm_url: LLM_URL,
    //vision_url: VISION_URL, // 🔒 비활성
    storage: "local:/assets",
    always_unstructured: ALWAYS_UNSTRUCTURED,
  });
});

// =========================
// 업로드
// =========================
app.post("/upload", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "file required" });

    const mimetype = req.file.mimetype || "";
    const filepath = req.file.originalname || "unknown";

    const saved = await saveOriginalFile(req.file.buffer, filepath);
    let text = "",
      tablesForMeta = [],
      pageImageUrls = [];

    if (mimetype.includes("pdf") || filepath.toLowerCase().endsWith(".pdf")) {
      const fileBuffer = await fsp.readFile(saved.absPath);

      if (ALWAYS_UNSTRUCTURED) {
        // 항상 Unstructured로 처리 (텍스트+표)
        const outJson = await extractWithUnstructuredOnce(fileBuffer, filepath);
        text = cleanText(outJson.text || "");
        tablesForMeta = Array.isArray(outJson.tables) ? outJson.tables : [];
      } else {
        // Fast path: pdf-parse → 실패 시 Unstructured
        const parsed = await pdfParse(fileBuffer).catch(() => ({ text: "" }));
        let raw = cleanText(parsed.text || "");
        if (!raw) {
          const outJson = await extractWithUnstructuredOnce(fileBuffer, filepath);
          raw = cleanText(outJson.text || "");
          tablesForMeta = Array.isArray(outJson.tables) ? outJson.tables : [];
        }
        text = raw;
      }

      // 페이지 이미지 렌더(그림/표 미리보기용) → 로컬 정적 저장
      try {
        const converter = await fromBuffer(fileBuffer, { format: "png", density: 180 });
        const pages = await converter.bulk(-1, false);
        for (const p of pages) {
          const buf = p.buffer || Buffer.from((p.base64 || "").split(",")[1] || "", "base64");
          if (!buf?.length) continue;
          const storagePath = path.join(ASSET_DIR, saved.sha, "pages");
          await fsp.mkdir(storagePath, { recursive: true });
          const filename = `page-${String(p.page).padStart(4, "0")}.png`;
          await fsp.writeFile(path.join(storagePath, filename), buf);
          pageImageUrls.push(`${PUBLIC_BASE}/assets/doc-assets/${saved.sha}/pages/${filename}`);
        }
      } catch (e) {
        console.warn("⚠️ page image render skipped:", e?.message || e);
      }
    } else if (/\.(jpe?g|png|webp)$/i.test(filepath) || mimetype.startsWith("image/")) {
      return res
        .status(415)
        .json({ error: "Image captioning disabled (vision off). Upload PDF/Office/Text/HWPX/HWP instead." });
    } else if (mimetype.includes("text") || /\.(txt|md)$/i.test(filepath)) {
      const encoding = chardet.detect(req.file.buffer) || "utf8";
      const decoded = iconv.decode(req.file.buffer, encoding);
      text = cleanText(decoded);
    } else if (/\.(hwpx)$/i.test(filepath) || mimetype.includes("hwpx")) {
      console.log("📄 HWPX detected: parsing XML...");
      const out = await extractHwpx(req.file.buffer);
      text = out.text;
      tablesForMeta = Array.isArray(out.tables) ? out.tables : [];
    } else if (/\.(hwp)$/i.test(filepath) || mimetype.includes("x-hwp")) {
      console.log("📄 HWP detected: converting via external tool...");
      try {
        text = await convertHwpToTxtViaCli(req.file.buffer, filepath);
      } catch (e) {
        return res.status(415).json({
          error: "HWP not supported on this server",
          detail: "서버에 HWP 변환기가 구성되어 있지 않습니다. HWPX 또는 PDF/DOCX로 변환하여 업로드하세요.",
        });
      }
    } else if (
      /\.(pptx?|docx?)$/i.test(filepath) ||
      mimetype.includes("presentation") ||
      mimetype.includes("word") ||
      mimetype.includes("officedocument")
    ) {
      // Office 문서 → Unstructured 한 번 호출
      const outJson = await extractWithUnstructuredOnce(req.file.buffer, filepath);
      text = cleanText(outJson.text || "");
      tablesForMeta = Array.isArray(outJson.tables) ? outJson.tables : [];
    } else {
      return res.status(400).json({ error: "Unsupported file type (PDF/TXT/MD/PPT/DOC/DOCX/PPTX/HWPX/HWP only)" });
    }

    if (!text) return res.status(400).json({ error: "empty text" });

    // ========= [표] 자산/표 저장 (로컬+MySQL) =========
    try {
      if (tablesForMeta?.length) {
        for (let i = 0; i < tablesForMeta.length; i++) {
          const t = tablesForMeta[i];
          const { header, rows, tsv, md, html, n_rows_hint, n_cols_hint } = normalizeTableMeta(t);

          // (1) 캡션 임베딩(너무 많은 페이지면 건너뜀)
          const pageNo = t?.page || 1;
          const caption = t?.caption || "Table";
          const doCaptionEmb = pageNo <= MAX_CAPTION_PAGES && !!caption;
          const captionEmb = doCaptionEmb ? await getEmbedding(caption, "passage") : null;

          // (2) 자산 생성 (table)
          const assetId = await insertDocAsset({
            sha256: saved.sha,
            filepath,
            page: pageNo,
            type: "table",
            image_url: pageImageUrls[pageNo - 1] || null,
            caption_text: caption,
            caption_emb: captionEmb, // 캡션 임베딩 저장
            meta: { source: "unstructured", headers: header },
          });

          // (3) 표 본문 저장
          await insertDocTable({
            asset_id: assetId,
            n_rows: n_rows_hint ?? rows.length,
            n_cols: n_cols_hint ?? (header.length || (rows[0]?.length || 0)),
            tsv,
            md,
            html: html || null,
          });

          // (4) 행(Row) → 문서(document) + 임베딩 (상한 적용)
          const limitRows = Math.min(rows.length, MAX_TABLE_ROWS_EMB);
          for (let rIdx = 0; rIdx < limitRows; rIdx++) {
            const row = rows[rIdx] || [];
            const sentence = rowToSentence(caption, header, row);
            const rowEmb = await getEmbedding(sentence, "passage");

            // 숫자 필드 정규화(옵션)
            const normalized = {};
            header.forEach((h, colIdx) => {
              const n = normalizeNumber(row[colIdx]);
              if (n && (n.value != null || n.unit || n.raw)) normalized[h] = n;
            });

            const metadata = {
              type: "table_row",
              asset_id: assetId,
              row_index: rIdx,
              headers: header,
              normalized,
              source: filepath,
              stored_path: saved.relPath,
              sha256: saved.sha,
              page: pageNo,
              caption,
              mimetype,
            };

            await insertDocumentWithEmbedding(sentence, metadata, rowEmb);
          }
        }
      }
    } catch (e) {
      console.warn("⚠️ table asset save skipped:", e?.message || e);
    }

    // ========= 청킹 + 임베딩 =========
    const chunks = chunkTextTokens(text, 800, 120);
    const limit = createLimiter(4);
    let inserted = 0;

    await Promise.all(
      chunks.map((c, idx) =>
        limit(async () => {
          const embRes = await fetch(`${EMB_URL}/embed`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ input: c.text, mode: "passage" }),
          });
          if (!embRes.ok) throw new Error(`embedding failed ${embRes.status}`);
          const embJson = await embRes.json();
          const vector = Array.isArray(embJson.embedding)
            ? embJson.embedding
            : (embJson.data && embJson.data[0] && Array.isArray(embJson.data[0].embedding))
            ? embJson.data[0].embedding
            : Array.isArray(embJson.data)
            ? embJson.data
            : null;
          if (!Array.isArray(vector)) throw new Error("invalid embedding payload");

          const metadata = {
            filepath,
            stored_path: saved.relPath,
            sha256: saved.sha,
            size: req.file.size,
            mimetype,
            chunk_index: idx,
            startTok: c.startTok,
            endTok: c.endTok,
            type: /\.(hwpx)$/i.test(filepath)
              ? "hwpx"
              : /\.(hwp)$/i.test(filepath)
              ? "hwp"
              : /\.(pptx?|docx?)$/i.test(filepath)
              ? "office"
              : /\.(txt|md)$/i.test(filepath)
              ? "text"
              : "pdf",
            ...(idx === 0 ? { assets: { pages: pageImageUrls, tables: tablesForMeta } } : {}),
          };

          await insertDocumentWithEmbedding(c.text, metadata, vector);
          inserted++;
        })
      )
    );

    res.json({
      ok: true,
      chunks: inserted,
      stored: saved.relPath,
      tables: tablesForMeta.length,
      pages: pageImageUrls.length,
      image_caption_chunks: 0, // 비전 비활성
    });
  } catch (e) {
    console.error("/upload error:", e);
    res.status(500).json({ error: e?.message || "server error" });
  }
});

// =========================
// 공용 유틸
// =========================
function withTimeout(promise, ms = 30000, tag = "req") {
  return Promise.race([
    promise,
    new Promise((_, rej) =>
      setTimeout(() => rej(new Error(`${tag} timeout ${ms}ms`)), ms)
    ),
  ]);
}

function readSSEToText(readable) {
  return new Promise((resolve, reject) => {
    let acc = "";
    const onData = (buf) => {
      const chunk = buf.toString();
      const lines = chunk.split(/\r?\n/);
      for (const line of lines) {
        if (!line) continue;
        if (line.startsWith("data: ")) {
          const payload = line.slice(6);
          if (payload === "[DONE]") continue;
          acc += payload;
        }
      }
    };
    const onEnd = () => resolve(acc);
    const onErr = (e) => reject(e);
    readable.on("data", onData);
    readable.on("end", onEnd);
    readable.on("error", onErr);
  });
}

// =========================
// 질문 모드 분류: "general" | "document"
// =========================
async function classifyQuestionMode(question) {
  const CLASSIFIER_SYSTEM = `
당신은 사용자의 질문이 어떤 종류인지 분류하는 조교입니다.

- "general": LLM 일반 지식, 기술 용어(예: LLM, RAG, 임베딩, 벡터DB, Transformer 등)의 뜻/개념/원리를 설명해 달라는 질문
- "document": 업로드된 문서 내용(논문, 보고서 등)에 기반해 답해야 하는 질문
  (예: "이 논문에서 RAG는 무엇을 의미하나요?", "p.23 표 해석해줘")

다음 중 하나만 단독으로 출력하십시오 (기타 텍스트 금지):
general
document
`.trim();

  const clsRes = await withTimeout(
    fetch(`${LLM_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
      },
      body: JSON.stringify({
        context: "",
        question: `사용자 질문: "${question}"`,
        history: [],
        system: CLASSIFIER_SYSTEM,
        max_new_tokens: 8,
        temperature: 0.0,
        top_p: 1.0,
      }),
    }),
    12000,
    "llm(classify)"
  );

  if (!clsRes.ok || !clsRes.body) {
    const t = await clsRes.text().catch(() => "");
    throw new Error(`classify failed: HTTP ${clsRes.status} ${t}`);
  }

  const raw = (await readSSEToText(clsRes.body)) || "";
  const ans = raw.trim().toLowerCase();

  if (ans.startsWith("general")) return "general";
  if (ans.startsWith("document")) return "document";

  // 애매하면 기본은 문서 기반
  return "document";
}

// =========================
// 일반 지식 모드: LLM으로만 답변 (출처 없음)
// =========================
async function answerWithGeneralKnowledge({ question, history, max_new_tokens, temperature, top_p }) {
  const GENERAL_SYSTEM = `
당신은 인공지능, 머신러닝, 소프트웨어 공학, 수학 등 일반 기술 지식을 설명하는 조교입니다.

- 말투: 한국어 존댓말, 간결하고 단계적으로 설명합니다.
- LLM, RAG(검색 증강 생성), 임베딩, 벡터DB, Transformer, CUDA, WebSocket 등은 일반 지식으로 자유롭게 설명하십시오.
- 1) 한 줄 정의, 2) 2~4줄 정도의 부연 설명으로 구성하되, 필요 이상으로 길게 설명하지 마십시오.
- 업로드된 문서(CONTEXT)는 사용하지 않으며, 출처 목록도 적지 마십시오.
`.trim();

  const llmRes = await withTimeout(
    fetch(`${LLM_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
      },
      body: JSON.stringify({
        context: "",
        question,
        history,
        system: GENERAL_SYSTEM,
        max_new_tokens: Math.min(Math.max(96, max_new_tokens || 384), 256),
        temperature: Math.max(0.15, temperature || 0.2),
        top_p: top_p ?? 0.9,
      }),
    }),
    30000,
    "llm(general)"
  );

  if (!llmRes.ok || !llmRes.body) {
    const t = await llmRes.text().catch(() => "");
    throw new Error(`LLM(general) failed: HTTP ${llmRes.status} ${t}`);
  }

  const answerRaw = await readSSEToText(llmRes.body);
  const answer = (answerRaw || "").trim();
  return answer;
}

// =========================
// /query
// =========================
async function handleQueryJSON(req, res) {
  try {
    const {
      question,
      match_count = 5,
      threshold = 0.6,
      history = [],
      strict = true,
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

    // 최근 문맥 50턴 제한
    if (Array.isArray(history) && history.length > 50) {
      history.splice(0, history.length - 50);
    }

    // 0. 스몰토크 fast-path
    const smalltalkRe =
      /^(?:\s*)(hi|hello|hey|yo|sup|안녕|안녕하세요|하이|헬로|ㅎ+|ㅋㅋ+|감사|고마워|땡큐|수고|잘가|bye|바이|누구야|너 누구|자기소개|소개|도움|help|지원)(?:[\s!?.…]*)$/i;
    const isSmalltalk = smalltalkRe.test(question);

    if (isSmalltalk) {
      const SMALLTALK_SYSTEM = `
당신은 기업 환경의 공식 AI 어시스턴트입니다.
- 말투: 격식체, 간결·친절.
- 인사/감사/작별/자기소개/도움요청은 문서 근거 없이 1~2문장으로만 응답.
- "근거:", "출처:" 같은 접두 라벨은 사용하지 않습니다.
`.trim();

      const llmRes = await withTimeout(
        fetch(`${LLM_URL}/chat`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
          },
          body: JSON.stringify({
            context: "",
            question,
            history,
            system: SMALLTALK_SYSTEM,
            max_new_tokens: Math.min(Math.max(48, max_new_tokens), 160),
            temperature: Math.max(0.3, temperature || 0.3),
            top_p,
          }),
        }),
        30000,
        "llm(smalltalk)"
      );
      if (!llmRes.ok || !llmRes.body) {
        const t = await llmRes.text().catch(() => "");
        throw new Error(`LLM(smalltalk) failed: HTTP ${llmRes.status} ${t}`);
      }
      const answerRaw = await readSSEToText(llmRes.body);
      const answer = (answerRaw || "").trim();
      return res.json({ ok: true, mode: "json", sources: [], answer });
    }

    // =========================
    // 1. RAG 검색을 항상 먼저 수행
    // =========================

    // 1) 임베딩
    const embRes = await withTimeout(
      fetch(`${EMB_URL}/embed`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input: question, mode: "query" }),
      }),
      20000,
      "embed(q)"
    );
    if (!embRes.ok) {
      throw new Error(`embedding(q) failed: HTTP ${embRes.status}`);
    }
    const embJson = await embRes.json();
    const qVec =
      (embJson && Array.isArray(embJson.embedding)) ? embJson.embedding :
      (embJson?.data?.[0]?.embedding && Array.isArray(embJson.data[0].embedding)) ? embJson.data[0].embedding :
      (Array.isArray(embJson.data) ? embJson.data : null);
    if (!Array.isArray(qVec)) {
      throw new Error("invalid embedding payload for question");
    }

    // 2) 검색용 threshold와 답변용 threshold 분리
    const RETRIEVE_MIN = 0.4;

    const BASE_ANSWER_THRESHOLD = 0.6;
    let answerThreshold = BASE_ANSWER_THRESHOLD;

    const wordCount = question.trim().split(/\s+/).filter(Boolean).length;

    if (wordCount <= 3) {
      answerThreshold -= 0.03;   // 0.57
    } else if (wordCount <= 7) {
      answerThreshold -= 0.02;   // 0.58
    }

    if (answerThreshold < 0.55) answerThreshold = 0.55;

    // 3) 매칭 (검색은 RETRIEVE_MIN 기준으로)
    const matches = await matchDocuments(qVec, {
      k: match_count,
      threshold: RETRIEVE_MIN,
    });

    const sims = matches.map((m) => m.similarity ?? 0);
    const maxSim = sims.length ? Math.max(...sims) : 0;
    const NOT_FOUND_MSG = "모릅니다.";

    // =========================
    // 2. 문서 매치가 충분히 좋으면 → RAG 답변
    // =========================
    if (maxSim >= answerThreshold) {
      const MAX_TOKENS = 1400;
      let approxTok = 0;
      const ctxParts = [];
      const srcPayloadRaw = [];

      for (const m of matches) {
        const meta = typeof m.metadata === "string" ? JSON.parse(m.metadata) : (m.metadata || {});
        const rawPath = meta.filepath || meta.source || "";
        const filename = rawPath ? rawPath.split(/[\\/]/).pop() : "";

        let t = (m.content || "").trim();
        if (t.length > 1400) {
          t = t.slice(0, 700) + "\n...\n" + t.slice(-700);
        }

        const est = Math.ceil(t.length / 3.5);
        if (approxTok + est > MAX_TOKENS) break;
        approxTok += est;

        const labelForCtx = filename || rawPath || "source";
        ctxParts.push(`【source:${labelForCtx}】\n${t}`);

        srcPayloadRaw.push({
          key: `${meta.doc_id || m.id || ""}|${rawPath}|${meta.page ?? ""}`,
          label: meta.chunk_index != null
            ? `chunk${meta.chunk_index}`
            : (m.id ? `id:${m.id}` : "chunk"),
          filepath: rawPath,
          filename,
          page: meta.page,
          similarity: Math.round((m.similarity ?? 0) * 1000) / 1000,
        });
      }

      // 출처 리스트 구성
      const seen = new Set();
      const sources = [];
      let sourceIdx = 1;
      for (const s of srcPayloadRaw) {
        const idKey = s.key || `${s.filepath}|${s.page || ""}`;
        if (!idKey) continue;
        if (seen.has(idKey)) continue;
        seen.add(idKey);

        const displayName =
          s.filename && s.filename.trim()
            ? s.filename
            : (s.filepath && s.filepath.trim())
            ? s.filepath
            : `문서${sourceIdx}`;

        sources.push({
          label: s.label,
          filepath: s.filepath || null,
          filename: displayName,
          page: s.page,
          similarity: s.similarity,
        });
        sourceIdx += 1;
        if (sources.length >= 3) break;
      }

      const sourcesList =
        sources.length > 0
          ? "\n\n---\n출처 목록:\n" +
            sources.map((s) => `【source:${s.filename}】`).join(" ")
          : "";
      const context = ctxParts.join("\n\n---\n\n") + sourcesList;

      const STRICT_SYSTEM_KO = `
당신은 문서 기반 AI 어시스턴트입니다.
- 말투: 항상 격식체. 이모지·감탄사·추측·감정표현 금지.
- CONTEXT(문서 발췌)만을 근거로 답변하십시오.
- 외부 지식, 추측, 상상은 절대 금지입니다.
- 답변 본문에 "근거:" 같은 꼬리표를 붙이지 마십시오.
- 답변의 마지막에만 아래 형식으로 출처를 명시하십시오:
  ---
  출처: 【source:파일명1】 【source:파일명2】 (가능하면 최대 3개)
- 답변은 500자 이내로 간결하게 대답하십시오:
- CONTEXT가 부족하거나 관련이 없으면 아래 문장만 답하십시오:
  "모릅니다."
`.trim();

      const llmRes = await withTimeout(
        fetch(`${LLM_URL}/chat`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
          },
          body: JSON.stringify({
            context,
            question,
            history,
            system: STRICT_SYSTEM_KO,
            max_new_tokens: Math.max(128, Math.min(max_new_tokens, 1024)),
            temperature,
            top_p,
          }),
        }),
        60000,
        "llm(rag)"
      );
      if (!llmRes.ok || !llmRes.body) {
        const t = await llmRes.text().catch(() => "");
        throw new Error(`LLM(rag) failed: HTTP ${llmRes.status} ${t}`);
      }
      const answerRaw = await readSSEToText(llmRes.body);
      const answer = (answerRaw || "").trim();

      const normalized = answer.replace(/\s+/g, "");
      const isUnknownAnswer =
        normalized === "모릅니다." ||
        normalized === "모릅니다" ||
        normalized.startsWith("모릅니다.");

      return res.json({
        ok: true,
        mode: "json",
        sources: isUnknownAnswer ? [] : sources,
        answer,
      });
    }

    // =========================
    // 3. 문서 매치가 약한 경우 → 분류 후 fallback
    // =========================
    let mode = "document";
    try {
      mode = await classifyQuestionMode(question); // "general" | "document"
    } catch (e) {
      console.warn("⚠️ classifyQuestionMode failed, fallback to document:", e?.message || e);
      mode = "document";
    }

    if (mode === "document") {
      // 문서에서 찾고 싶은 질문인데, 근거가 없음 → 모른다 (출처 없음)
      return res.json({
        ok: true,
        mode: "json",
        sources: [],
        answer: NOT_FOUND_MSG,
      });
    }

    // mode === "general" → 일반 기술 지식 모드 (LLM 지식, 출처 없음)
    const generalAnswer = await answerWithGeneralKnowledge({
      question,
      history,
      max_new_tokens,
      temperature,
      top_p,
    });

    return res.json({
      ok: true,
      mode: "json",
      sources: [],
      answer: generalAnswer,
    });

  } catch (e) {
    console.error("🔥 /query error:", e);
    if (!res.headersSent) {
      return res.status(500).json({ ok: false, error: e?.message || "query failed" });
    }
  }
}

// =========================
// 라우터 바인딩
// =========================
app.post("/query", handleQueryJSON);

app.get("/query/:question", (req, res) => {
  const q = decodeURIComponent(req.params.question || "");
  req.body = { question: q, history: [], match_count: 5, threshold: 0.6, strict: true };
  return handleQueryJSON(req, res);
});


// =========================
/** 서버 시작 */
// =========================
app.listen(PORT, "0.0.0.0", () => {
  console.log(`✅ Server running on http://0.0.0.0:${PORT}`);
  console.log(`   - EMB_URL = ${EMB_URL}`);
  console.log(`   - LLM_URL = ${LLM_URL}`);
  //console.log(`   - VISION_URL = ${VISION_URL}`); // 🔒 비활성
  if (HWP2TXT_EXE) console.log(`   - HWP2TXT_EXE = ${HWP2TXT_EXE}`);
});