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

const { runRag } = require("./rag_langchain");

const {
  insertDocumentWithEmbedding,
  insertDocAsset,
  insertDocTable,
  matchDocuments,
} = require("../db/repo.js");

const { chunkTextTokens } = require("./chunk.js");

// =========================
// 환경변수
// =========================
const PORT = Number(process.env.PORT || 8000);
const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || "")
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);

const EMB_URL = (process.env.EMB_URL || "http://127.0.0.1:8001").replace(/\/$/, "");
const LLM_URL = (process.env.LLM_URL || "http://127.0.0.1:8002").replace(/\/$/, "");

// Docling/파이썬 실행용
const DOC_PY =
  process.env.UNSTRUCT_PY ||
  (process.platform === "win32"
    ? "C:\\Users\\user\\anaconda3\\envs\\unstruct\\python.exe"
    : "python3");

const HWP2TXT_EXE = process.env.HWP2TXT_EXE || "";
const PUBLIC_BASE = (process.env.PUBLIC_BASE || `http://127.0.0.1:${PORT}`).replace(
  /\/$/,
  ""
);

// === 옵션 플래그 ===
const ALWAYS_UNSTRUCTURED = String(process.env.ALWAYS_UNSTRUCTURED || "false") === "true"; // 지금은 안 쓰지만 남겨둠
const FAST_MODE = String(process.env.FAST_MODE || "false") === "true"; // 속도 튜닝용 플래그 추가

const MAX_TABLE_ROWS_EMB = Number(process.env.MAX_TABLE_ROWS_EMB || 50);
const MAX_CAPTION_PAGES = Number(process.env.MAX_CAPTION_PAGES || 10);
const RENDER_PAGES =
  String(
    process.env.RENDER_PAGES ??
      (process.env.ENABLE_PAGE_IMAGES === "1" ? "true" : "false")
  ) === "true";
const ENABLE_TABLE_INDEX = String(process.env.ENABLE_TABLE_INDEX || "true") === "true";

const CHUNK_SIZE_TOKENS = Number(process.env.CHUNK_SIZE_TOKENS || 800);
const CHUNK_OVERLAP_TOKENS = Number(process.env.CHUNK_OVERLAP_TOKENS || 120);
const MAX_CHUNKS_EMB = Number(process.env.MAX_CHUNKS_EMB || 0);

// =========================
// 임베딩 헬퍼
// =========================
async function getEmbedding(text, mode = "passage") {
  const r = await fetch(`${EMB_URL}/v1/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "BAAI/bge-m3",
      input: text
    }),
  });

  if (!r.ok) {
    const errText = await r.text();
    throw new Error(`embedding failed ${r.status}: ${errText}`);
  }

  const j = await r.json();
  // OpenAI 응답 포맷: { data: [{ embedding: [...] }] }
  return j.data[0].embedding;
}

// 여러 개 텍스트를 한 번에 임베딩
async function getEmbeddingsBatch(texts, mode = "passage") {
  if (!Array.isArray(texts) || texts.length === 0) return [];

  const r = await fetch(`${EMB_URL}/v1/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "BAAI/bge-m3",
      input: texts
    }),
  });

  if (!r.ok) {
    const errText = await r.text();
    throw new Error(`embedding(batch) failed ${r.status}: ${errText}`);
  }
  
  const j = await r.json();
  // 순서대로 벡터 추출
  return j.data.map(d => d.embedding);
}

// 숫자/단위 정규화
function normalizeNumber(v) {
  if (v == null) return null;
  const s = String(v);
  const m = s.match(/^([\d.,+-]+)\s*([A-Za-z%]*)$/);
  if (!m) return { raw: s };
  const num = parseFloat(m[1].replace(/,/g, ""));
  const unit = m[2] || "";
  return { value: isNaN(num) ? null : num, unit, raw: s };
}

// 행(Row) 내용을 자연어 문장으로
function rowToSentence(tableTitle, headers, row) {
  const parts = headers.map((h, i) => `${h}=${row[i] ?? ""}`);
  return `Table: ${tableTitle} | ${parts.join("; ")}`;
}

const app = express();

// =========================
// CORS
// =========================
if (ALLOWED_ORIGINS.length > 0) {
  app.use(
    cors({
      origin: ALLOWED_ORIGINS,
      credentials: true,
    })
  );
} else {
  app.use(cors());
}

app.use(express.json({ limit: "10mb" }));

// =========================
// 디렉터리/Static
// =========================
const PUBLIC_DIR = path.join(__dirname, "public");
const ASSET_DIR = path.join(PUBLIC_DIR, "doc-assets");
if (!fs.existsSync(ASSET_DIR)) fs.mkdirSync(ASSET_DIR, { recursive: true });
app.use("/assets", express.static(PUBLIC_DIR, { maxAge: "1y" }));

const UPLOAD_DIR = path.join(__dirname, "uploads");
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });
// uploads 도 정적 서빙 (이미지/원본 파일 URL용)
app.use("/uploads", express.static(UPLOAD_DIR, { maxAge: "1y" }));

// =========================
// 업로드(메모리, 100MB 제한)
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
  const relPath = path.relative(UPLOAD_DIR, absPath).replace(/\\/g, "/");
  return { absPath, relPath, sha };
}

// 파생 이미지 저장 (페이지 썸네일/표/그림 등)
async function saveDerivedImage(buffer, saved, kind, filenameHint, ext = ".jpg") {
  const dayRel = path.dirname(saved.relPath); // 예: 2025/12/10
  const dayDir = path.join(UPLOAD_DIR, dayRel, saved.sha, kind);
  await fsp.mkdir(dayDir, { recursive: true });

  const safeName = safeBasename(filenameHint || kind);
  const storedName = `${safeName}${ext}`;
  const absPath = path.join(dayDir, storedName);
  await fsp.writeFile(absPath, buffer);

  const relPath = path.relative(UPLOAD_DIR, absPath).replace(/\\/g, "/");
  const url = `${PUBLIC_BASE}/uploads/${relPath}`;
  return { absPath, relPath, url };
}

// Docling이 생성한 파일을 uploads로 복사
async function moveDoclingImageToUploads(localPath, saved, kind, index) {
  if (!localPath) return null;
  try {
    const buf = await fsp.readFile(localPath);
    const filenameHint = `${kind}-${String(index).padStart(3, "0")}`;
    const { url } = await saveDerivedImage(buf, saved, kind, filenameHint, ".jpg");
    return url;
  } catch (e) {
    console.warn("⚠️ moveDoclingImageToUploads failed:", e?.message || e);
    return null;
  }
}

// =========================
/* 텍스트 정리 */
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
// Docling: ocr_once.py 호출
// =========================
async function extractWithDoclingOnce(saved, filename) {
  const pyPath = path.join(__dirname, "ocr_once.py");
  const outDir = path.join(UPLOAD_DIR, "docling_tmp", saved.sha);
  await fsp.mkdir(outDir, { recursive: true });

  const stdout = await new Promise((resolve, reject) => {
    const p = spawn(DOC_PY, [pyPath, saved.absPath, outDir], {
      stdio: ["ignore", "pipe", "pipe"],
      env: {
        ...process.env,
        PYTHONUTF8: "1",
        LANG: "ko_KR.UTF-8",
        LC_ALL: "ko_KR.UTF-8",
      },
    });

    let out = "";
    let err = "";
    p.stdout.on("data", (d) => (out += d.toString()));
    p.stderr.on("data", (d) => (err += d.toString()));
    p.on("error", reject);
    p.on("close", (code) => {
      if (code !== 0) {
        return reject(new Error(`docling exit ${code}: ${err || out}`));
      }
      resolve(out);
    });
  });

  let parsed;
  try {
    parsed = JSON.parse(stdout);
  } catch (e) {
    throw new Error(`docling output parse error: ${e.message}`);
  }
  if (!parsed?.ok) throw new Error(parsed?.error || "docling returned ok=false");
  return parsed; // { ok, text, tables, pictures, engine }
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
    sectionEntries = Object.keys(zip.files).filter((n) =>
      n.toLowerCase().endsWith(".xml")
    );
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
      if (/tbl$/i.test(k) && typeof v === "object")
        tables.push({ entry, type: "table" });
      if (Array.isArray(v)) v.forEach((x) => findTables(x, entry));
      else if (typeof v === "object") findTables(v, entry);
    }
  };

  for (const entry of sectionEntries) {
    const xmlStr = await zip.files[entry].async("string");
    const xml = await parseStringPromise(xmlStr, {
      explicitArray: true,
      preserveChildrenOrder: true,
    });
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
      const p = spawn(HWP2TXT_EXE, [tmpIn, tmpOut], {
        stdio: ["ignore", "pipe", "pipe"],
      });
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
// 표 정규화 유틸 (unstructured/docling table meta → header/rows/tsv/md/html)
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
  // 1) unstructured 쪽: text_as_html 있으면 그거 먼저 우선
  let html = t?.html || t?.metadata?.text_as_html || null;
  let header = [];
  let rows = [];

  if (html) {
    const dom = new JSDOM(html);
    const $rows = [...dom.window.document.querySelectorAll("tr")];
    const grid = $rows.map((tr) =>
      [...tr.querySelectorAll("th,td")].map((td) => (td.textContent || "").trim())
    );
    header = grid[0] || [];
    rows = grid.slice(1);
  } else if (Array.isArray(t?.rows) || Array.isArray(t?.header)) {
    // 2) docling 쪽: header/rows만 있는 경우
    const arr = Array.isArray(t.rows) ? t.rows : [];
    header = Array.isArray(t.header)
      ? t.header
      : (arr[0] || []).map((_, i) => `col_${i + 1}`);
    rows = arr.length ? arr : [];

    // docling용 HTML 직접 생성
    const headHtml =
      "<thead><tr>" +
      header.map((h) => `<th>${sanitizeCell(h)}</th>`).join("") +
      "</tr></thead>";

    const bodyHtml =
      "<tbody>" +
      rows
        .map(
          (r) =>
            "<tr>" +
            (r || [])
              .map((c) => `<td>${sanitizeCell(c)}</td>`)
              .join("") +
            "</tr>"
        )
        .join("") +
      "</tbody>";

    html = `<table>${headHtml}${bodyHtml}</table>`;
  } else if (Array.isArray(t?.preview_rows)) {
    const grid = t.preview_rows.map((r) =>
      Array.isArray(r) ? r.map(sanitizeCell) : [sanitizeCell(String(r))]
    );
    header = grid[0] || [];
    rows = grid.slice(1);
  }

  const tsv = [header, ...rows]
    .map((r) => r.map(sanitizeCell).join("\t"))
    .join("\n");
  const md = toMarkdownTable(header, rows);

  const n_rows_hint = Number.isFinite(t?.n_rows) ? t.n_rows : rows.length;
  const n_cols_hint = Number.isFinite(t?.n_cols)
    ? t.n_cols
    : header.length || (rows[0]?.length || 0);

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
    storage: "local:/assets",
    always_unstructured: ALWAYS_UNSTRUCTURED,
    render_pages: RENDER_PAGES,
    max_table_rows_emb: MAX_TABLE_ROWS_EMB,
    fast_mode: FAST_MODE, // 상태 표시
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
      picturesForMeta = [],
      pageImageUrls = []; // 필요하면 pdf2pic으로 페이지 썸네일 생성

    // =========================
    // PDF / Office → Docling
    // =========================
    if (
      mimetype.includes("pdf") ||
      filepath.toLowerCase().endsWith(".pdf") ||
      /\.(pptx?|docx?)$/i.test(filepath) ||
      mimetype.includes("presentation") ||
      mimetype.includes("word") ||
      mimetype.includes("officedocument")
    ) {
      // 1) Docling으로 텍스트/표/이미지 메타 추출
      const outJson = await extractWithDoclingOnce(saved, filepath);
      text = cleanText(outJson.text || "");
      tablesForMeta = Array.isArray(outJson.tables) ? outJson.tables : [];
      picturesForMeta = Array.isArray(outJson.pictures) ? outJson.pictures : [];

      console.log(
        `[docling] engine=${outJson.engine}, tables=${tablesForMeta.length}, pictures=${picturesForMeta.length}`
      );

      // 2) 페이지 이미지 썸네일 (옵션)
      if (RENDER_PAGES && filepath.toLowerCase().endsWith(".pdf")) {
        try {
          const fileBuffer = await fsp.readFile(saved.absPath);
          const converter = await fromBuffer(fileBuffer, {
            format: "jpg",
            density: 180,
          });
          const pages = await converter.bulk(-1, false);
          for (const p of pages) {
            const buf =
              p.buffer ||
              Buffer.from((p.base64 || "").split(",")[1] || "", "base64");
            if (!buf?.length) continue;

            const { url } = await saveDerivedImage(
              buf,
              saved,
              "pages",
              `page-${String(p.page).padStart(4, "0")}`,
              ".jpg"
            );
            pageImageUrls.push(url);
          }
        } catch (e) {
          console.warn("page image render failed:", e?.message || e);
        }
      } else if (RENDER_PAGES) {
        console.log("page image render skipped (not a PDF)");
      }
    }
    // =========================
    // 이미지 (Vision 비활성)
    // =========================
    else if (
      /\.(jpe?g|png|webp)$/i.test(filepath) ||
      mimetype.startsWith("image/")
    ) {
      return res.status(415).json({
        error:
          "Image captioning disabled (vision off). Upload PDF/Office/Text/HWPX/HWP instead.",
      });
    }
    // =========================
    // TXT / MD
    // =========================
    else if (mimetype.includes("text") || /\.(txt|md)$/i.test(filepath)) {
      const encoding = chardet.detect(req.file.buffer) || "utf8";
      const decoded = iconv.decode(req.file.buffer, encoding);
      text = cleanText(decoded);
    }
    // =========================
    // HWPX
    // =========================
    else if (/\.(hwpx)$/i.test(filepath) || mimetype.includes("hwpx")) {
      console.log("HWPX detected: parsing XML...");
      const out = await extractHwpx(req.file.buffer);
      text = out.text;
      tablesForMeta = Array.isArray(out.tables) ? out.tables : [];
    }
    // =========================
    // HWP
    // =========================
    else if (/\.(hwp)$/i.test(filepath) || mimetype.includes("x-hwp")) {
      console.log("HWP detected: converting via external tool...");
      try {
        text = await convertHwpToTxtViaCli(req.file.buffer, filepath);
      } catch (e) {
        return res.status(415).json({
          error: "HWP not supported on this server",
          detail:
            "서버에 HWP 변환기가 구성되어 있지 않습니다. HWPX 또는 PDF/DOCX로 변환하여 업로드하세요.",
        });
      }
    } else {
      return res.status(400).json({
        error:
          "Unsupported file type (PDF/TXT/MD/PPT/DOC/DOCX/PPTX/HWPX/HWP only)",
      });
    }

    if (!text) return res.status(400).json({ error: "empty text" });

    // ========= [표] 자산/표 저장 (로컬+MySQL) =========
    try {
      const effectiveMaxTableRowsEmb = FAST_MODE ? 0 : MAX_TABLE_ROWS_EMB; // ★ FAST_MODE일 때 행 임베딩 강제 OFF

      if (ENABLE_TABLE_INDEX && tablesForMeta?.length) {
        const rowInsertLimiter = createLimiter(8);

        for (let i = 0; i < tablesForMeta.length; i++) {
          const t = tablesForMeta[i];
          const {
            header,
            rows,
            tsv,
            md,
            html,
            n_rows_hint,
            n_cols_hint,
          } = normalizeTableMeta(t);

          const pageNo = t?.page || 1;
          const caption = t?.caption || "Table";

          // 이미지 URL (Docling이 만든 이미지 파일 → uploads로 이동)
          let tableImageUrl = null;
          if (t.image_path) {
            tableImageUrl = await moveDoclingImageToUploads(
              t.image_path,
              saved,
              "tables",
              i + 1
            );
          }

          // (1) 캡션 임베딩
          const doCaptionEmb =
            !FAST_MODE && // ★ FAST_MODE에서는 캡션 임베딩도 스킵
            MAX_CAPTION_PAGES > 0 &&
            pageNo <= MAX_CAPTION_PAGES &&
            !!caption;
          const captionEmb = doCaptionEmb
            ? await getEmbedding(caption, "passage")
            : null;

          // (2) 자산 생성
          const assetId = await insertDocAsset({
            sha256: saved.sha,
            filepath,
            page: pageNo,
            type: "table",
            image_url: tableImageUrl,
            caption_text: caption,
            caption_emb: captionEmb,
            meta: { source: t?.source || "docling", headers: header },
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

          // (4) 행(Row) → 문서(document) + 임베딩
          if (effectiveMaxTableRowsEmb > 0 && rows.length > 0) {
            const limitRows = Math.min(rows.length, effectiveMaxTableRowsEmb);

            const sentences = [];
            const rowMeta = [];

            for (let rIdx = 0; rIdx < limitRows; rIdx++) {
              const row = rows[rIdx] || [];
              const sentence = rowToSentence(caption, header, row);
              sentences.push(sentence);

              const normalized = {};
              header.forEach((h, colIdx) => {
                const n = normalizeNumber(row[colIdx]);
                if (n && (n.value != null || n.unit || n.raw)) {
                  normalized[h] = n;
                }
              });

              rowMeta.push({
                rowIndex: rIdx,
                normalized,
              });
            }

            const rowVecs = await getEmbeddingsBatch(sentences, "passage");
            if (!Array.isArray(rowVecs) || rowVecs.length !== sentences.length) {
              throw new Error("table row embedding batch size mismatch");
            }

            await Promise.all(
              rowVecs.map((vec, idx2) =>
                rowInsertLimiter(async () => {
                  const { rowIndex, normalized } = rowMeta[idx2];

                  const metadata = {
                    type: "table_row",
                    asset_id: assetId,
                    row_index: rowIndex,
                    headers: header,
                    normalized,
                    source: filepath,
                    stored_path: saved.relPath,
                    sha256: saved.sha,
                    page: pageNo,
                    caption,
                    mimetype,
                  };

                  await insertDocumentWithEmbedding(sentences[idx2], metadata, vec);
                })
              )
            );
          }
        }
      } else if (!ENABLE_TABLE_INDEX && tablesForMeta?.length) {
        console.log(
          `table indexing skipped (ENABLE_TABLE_INDEX=false, tables=${tablesForMeta.length})`
        );
      }
    } catch (e) {
      console.warn("table asset save skipped:", e?.message || e);
    }

    // ========= [이미지] 자산 저장 (Docling 그림) =========
    try {
      if (picturesForMeta?.length) {
        for (let i = 0; i < picturesForMeta.length; i++) {
          const p = picturesForMeta[i] || {};
          const pageNo = p.page || 1;
          const caption = p.caption || `Image ${i + 1}`;

          const imageUrl = p.image_path
            ? await moveDoclingImageToUploads(p.image_path, saved, "pictures", i + 1)
            : null;

          const doCaptionEmb =
            !FAST_MODE && // ★ FAST_MODE에서는 이미지 캡션 임베딩도 스킵
            MAX_CAPTION_PAGES > 0 &&
            pageNo <= MAX_CAPTION_PAGES &&
            !!caption;
          const captionEmb = doCaptionEmb
            ? await getEmbedding(caption, "passage")
            : null;

          const assetId = await insertDocAsset({
            sha256: saved.sha,
            filepath,
            page: pageNo,
            type: "image",
            image_url: imageUrl,
            caption_text: caption,
            caption_emb: captionEmb,
            meta: {
              source: p?.source || "docling",
            },
          });

          if (captionEmb) {
            await insertDocumentWithEmbedding(
              caption,
              {
                type: "image_caption",
                asset_id: assetId,
                source: filepath,
                stored_path: saved.relPath,
                sha256: saved.sha,
                page: pageNo,
                mimetype,
              },
              captionEmb
            );
          }
        }
      }
    } catch (e) {
      console.warn("image asset save skipped:", e?.message || e);
    }

    // ========= 청킹 + 임베딩 =========
    let chunks = chunkTextTokens(text, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS);

    // ★ FAST_MODE일 때 청크 개수 강제 제한
    if (FAST_MODE && chunks.length > 24) {
      console.log(`FAST_MODE: chunks truncated for speed: ${chunks.length} -> 24`);
      chunks = chunks.slice(0, 24);
    } else if (MAX_CHUNKS_EMB > 0 && chunks.length > MAX_CHUNKS_EMB) {
      console.log(`chunks truncated for speed: ${chunks.length} -> ${MAX_CHUNKS_EMB}`);
      chunks = chunks.slice(0, MAX_CHUNKS_EMB);
    }

    const textsForEmb = chunks.map((c) => c.text);
    const vectors = await getEmbeddingsBatch(textsForEmb, "passage");
    if (!Array.isArray(vectors) || vectors.length !== chunks.length) {
      throw new Error("embedding batch size mismatch");
    }

    const limit = createLimiter(8);
    let inserted = 0;

    await Promise.all(
      chunks.map((c, idx) =>
        limit(async () => {
          const vector = vectors[idx];
          if (!Array.isArray(vector)) {
            throw new Error(`invalid vector at index ${idx}`);
          }

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
            ...(idx === 0
              ? {
                  assets: {
                    pages: pageImageUrls,
                    tables: tablesForMeta,
                    pictures: picturesForMeta,
                  },
                }
              : {}),
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
      image_caption_chunks: 0,
    });
  } catch (e) {
    console.error("/upload error:", e);
    res.status(500).json({ error: e?.message || "server error" });
  }
});

// =========================
// /query
// =========================
app.post("/query", async (req, res) => {
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
    console.error("/query error (langchain):", e);
    if (!res.headersSent) {
      return res.status(500).json({ ok: false, error: e?.message || "query failed" });
    }
  }
});

// =========================
// /query/:question (GET)
// =========================
app.get("/query/:question", async (req, res) => {
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
});

// =========================
// 서버 시작
// =========================
app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server running on http://0.0.0.0:${PORT}`);
  console.log(`   - EMB_URL = ${EMB_URL}`);
  console.log(`   - LLM_URL = ${LLM_URL}`);
  console.log(`   - ALWAYS_UNSTRUCTURED = ${ALWAYS_UNSTRUCTURED}`);
  console.log(`   - FAST_MODE = ${FAST_MODE}`); // 로그 추가
  console.log(`   - RENDER_PAGES = ${RENDER_PAGES}`);
  console.log(`   - MAX_TABLE_ROWS_EMB = ${MAX_TABLE_ROWS_EMB}`);
  console.log(`   - MAX_CAPTION_PAGES = ${MAX_CAPTION_PAGES}`);
  console.log(`   - UPLOAD_DIR = ${UPLOAD_DIR}`);
  if (HWP2TXT_EXE) console.log(`   - HWP2TXT_EXE = ${HWP2TXT_EXE}`);
});
