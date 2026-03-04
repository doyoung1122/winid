import path from "path";
import fs from "fs";
import fsp from "fs/promises";
import { spawn } from "child_process";
import chardet from "chardet";
import iconv from "iconv-lite";
import JSZip from "jszip";
import { parseStringPromise } from "xml2js";

import {
  UPLOAD_DIR,
  DOC_PY,
  HWP2TXT_EXE,
  FAST_MODE,
  CHUNK_SIZE_TOKENS,
  CHUNK_OVERLAP_TOKENS,
  MAX_CHUNKS_EMB,
  SERVER_ROOT,
} from "../config/env.js";

import { getEmbedding, getEmbeddingsBatch } from "../services/embedding.service.js";
import { describeImage } from "../services/llm.service.js";
import { insertDocumentWithEmbedding } from "../services/vector.service.js";

import { cleanText } from "../utils/text.util.js";
import { saveOriginalFile, createLimiter, safeBasename } from "../utils/file.util.js";
import { normalizeTableMeta } from "../utils/table.util.js";

import { chunkTextTokens, chunkBySections } from "../../chunk.js";

if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

/**
 * Docling OCR로 텍스트/표/이미지 추출
 */
async function extractWithDoclingOnce(saved, filename) {
  const pyPath = path.join(SERVER_ROOT, "ocr_once.py");
  const outDir = path.join(UPLOAD_DIR, "docling_tmp", saved.sha);
  await fsp.mkdir(outDir, { recursive: true });

  // SHA 기반 캐시: 같은 파일은 Docling 재실행 없이 결과 재사용
  const cacheFile = path.join(outDir, "_result.json");
  try {
    const cached = JSON.parse(await fsp.readFile(cacheFile, "utf8"));
    if (cached?.ok) {
      console.log(`[docling] cache hit for ${saved.sha.slice(0, 8)}`);
      return cached;
    }
  } catch { /* cache miss */ }

  const stdout = await new Promise((resolve, reject) => {
    const p = spawn(DOC_PY, [pyPath, saved.absPath, outDir], {
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, PYTHONUTF8: "1", LANG: "ko_KR.UTF-8", LC_ALL: "ko_KR.UTF-8" },
    });
    let out = "", err = "";
    p.stdout.on("data", (d) => (out += d.toString()));
    p.stderr.on("data", (d) => (err += d.toString()));
    p.on("error", reject);
    p.on("close", (code) => {
      if (code !== 0) return reject(new Error(`docling exit ${code}: ${err || out}`));
      resolve(out);
    });
  });

  let parsed;
  try { parsed = JSON.parse(stdout); } catch (e) {
    throw new Error(`docling output parse error: ${e.message}`);
  }
  if (!parsed?.ok) throw new Error(parsed?.error || "docling returned ok=false");

  // 결과 캐시 저장
  await fsp.writeFile(cacheFile, JSON.stringify(parsed), "utf8").catch(() => {});

  return parsed; // { ok, text, tables, pictures, engine }
}

/**
 * HWPX (ZIP+XML) 파싱
 */
async function extractHwpx(buffer) {
  const zip = await JSZip.loadAsync(buffer);
  let sectionEntries = Object.keys(zip.files)
    .filter((n) => /^Contents\/section\d+\.xml$/i.test(n))
    .sort();
  if (sectionEntries.length === 0) {
    sectionEntries = Object.keys(zip.files).filter((n) => n.toLowerCase().endsWith(".xml"));
  }

  const texts = [];
  const crawl = (node) => {
    if (node == null || typeof node !== "object") return;
    if (typeof node._ === "string") texts.push(node._);
    for (const k of Object.keys(node)) {
      const v = node[k];
      if (Array.isArray(v)) v.forEach(crawl);
      else if (typeof v === "object") crawl(v);
    }
  };

  for (const entry of sectionEntries) {
    const xmlStr = await zip.files[entry].async("string");
    const xml = await parseStringPromise(xmlStr, { explicitArray: true, preserveChildrenOrder: true });
    crawl(xml);
  }
  return { text: cleanText(texts.join("\n")) };
}

/**
 * HWP → TXT 변환 (외부 CLI)
 */
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
    return cleanText(await fsp.readFile(tmpOut, "utf8"));
  } finally {
    try { await fsp.unlink(tmpIn); } catch {}
    try { await fsp.unlink(tmpOut); } catch {}
  }
}

/**
 * POST /upload handler
 */
export async function handleUpload(req, res) {
  try {
    if (!req.file) return res.status(400).json({ error: "file required" });

    const mimetype = req.file.mimetype || "";
    const filepath = req.file.originalname || "unknown";
    const docTitle = path.basename(filepath, path.extname(filepath));

    const saved = await saveOriginalFile(req.file.buffer, filepath);

    let text = "";
    let tablesForMeta = [];
    let picturesForMeta = [];

    // ── PDF / Office → Docling ──────────────────────────────────
    if (
      mimetype.includes("pdf") ||
      filepath.toLowerCase().endsWith(".pdf") ||
      /\.(pptx?|docx?)$/i.test(filepath) ||
      mimetype.includes("presentation") ||
      mimetype.includes("word") ||
      mimetype.includes("officedocument")
    ) {
      const outJson = await extractWithDoclingOnce(saved, filepath);
      text = cleanText(outJson.text || "");
      tablesForMeta = Array.isArray(outJson.tables) ? outJson.tables : [];
      picturesForMeta = Array.isArray(outJson.pictures) ? outJson.pictures : [];
      console.log(`[docling] engine=${outJson.engine}, tables=${tablesForMeta.length}, pictures=${picturesForMeta.length}`);
    }
    // ── TXT / MD ─────────────────────────────────────────────────
    else if (mimetype.includes("text") || /\.(txt|md)$/i.test(filepath)) {
      const encoding = chardet.detect(req.file.buffer) || "utf8";
      text = cleanText(iconv.decode(req.file.buffer, encoding));
    }
    // ── HWPX ─────────────────────────────────────────────────────
    else if (/\.(hwpx)$/i.test(filepath) || mimetype.includes("hwpx")) {
      text = (await extractHwpx(req.file.buffer)).text;
    }
    // ── HWP ──────────────────────────────────────────────────────
    else if (/\.(hwp)$/i.test(filepath) || mimetype.includes("x-hwp")) {
      try {
        text = await convertHwpToTxtViaCli(req.file.buffer, filepath);
      } catch {
        return res.status(415).json({
          error: "HWP not supported on this server",
          detail: "HWPX 또는 PDF/DOCX로 변환하여 업로드하세요.",
        });
      }
    } else {
      return res.status(400).json({ error: "Unsupported file type (PDF/TXT/MD/DOCX/PPTX/HWPX/HWP only)" });
    }

    if (!text) return res.status(400).json({ error: "empty text" });

    const baseMeta = { source: filepath, stored_path: saved.relPath, sha256: saved.sha, mimetype };

    // ── [표] Markdown → ChromaDB ─────────────────────────────────
    let tableChunks = 0;
    if (!FAST_MODE && tablesForMeta.length > 0) {
      try {
        const tableTexts = [];
        const tableMetas = [];

        for (let i = 0; i < tablesForMeta.length; i++) {
          const t = tablesForMeta[i];
          const { md } = normalizeTableMeta(t);
          if (!md?.trim()) continue;

          const caption = t.caption || `표 ${i + 1}`;
          const pageNo = t.page || 1;
          const tableText = `[${docTitle} > ${caption}]\n\n${md}`;

          tableTexts.push(tableText);
          tableMetas.push({ ...baseMeta, type: "table", page: pageNo, caption });
        }

        if (tableTexts.length > 0) {
          const vecs = await getEmbeddingsBatch(tableTexts, "passage");
          const limiter = createLimiter(8);
          await Promise.all(
            tableTexts.map((txt, i) =>
              limiter(() => insertDocumentWithEmbedding(txt, tableMetas[i], vecs[i]))
            )
          );
          tableChunks = tableTexts.length;
          console.log(`[table] embedded ${tableChunks}/${tablesForMeta.length}`);
        }
      } catch (e) {
        console.warn("[table] embedding failed:", e?.message || e);
      }
    }

    // ── [이미지] VLM 설명 → ChromaDB ────────────────────────────
    let imageChunks = 0;
    if (!FAST_MODE && picturesForMeta.length > 0) {
      try {
        const imgLimiter = createLimiter(2); // VLM 동시 처리 최대 2개
        await Promise.all(
          picturesForMeta.map((p, i) =>
            imgLimiter(async () => {
              const imageSrc = p.image_path;
              if (!imageSrc) return;

              // 20KB 미만 스킵 (로고·배경 등 노이즈 제거)
              try {
                const { size } = await fsp.stat(imageSrc);
                if (size < 20 * 1024) return;
              } catch { return; }

              let description = "";
              try {
                description = await describeImage(imageSrc, docTitle);
              } catch (e) {
                console.warn(`[VLM] image ${i + 1} failed:`, e?.message);
                return;
              }
              if (!description) return;

              const vec = await getEmbedding(description, "passage");
              await insertDocumentWithEmbedding(description, {
                ...baseMeta,
                type: "image_caption",
                page: p.page || 1,
              }, vec);
              imageChunks++;
            })
          )
        );
        console.log(`[VLM] image descriptions: ${imageChunks}/${picturesForMeta.length}`);
      } catch (e) {
        console.warn("[image] processing failed:", e?.message || e);
      }
    }

    // ── [텍스트] 섹션 청킹 → ChromaDB ───────────────────────────
    const useSection =
      mimetype.includes("pdf") ||
      filepath.toLowerCase().endsWith(".pdf") ||
      /\.(pptx?|docx?)$/i.test(filepath);

    let chunks = useSection
      ? chunkBySections(text, docTitle, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)
      : chunkTextTokens(text, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS);

    if (FAST_MODE && chunks.length > 24) {
      chunks = chunks.slice(0, 24);
    } else if (MAX_CHUNKS_EMB > 0 && chunks.length > MAX_CHUNKS_EMB) {
      console.log(`chunks truncated: ${chunks.length} -> ${MAX_CHUNKS_EMB}`);
      chunks = chunks.slice(0, MAX_CHUNKS_EMB);
    }

    const textType = /\.(hwpx)$/i.test(filepath) ? "hwpx"
      : /\.(hwp)$/i.test(filepath)   ? "hwp"
      : /\.(pptx?|docx?)$/i.test(filepath) ? "office"
      : /\.(txt|md)$/i.test(filepath) ? "text"
      : "pdf";

    const vectors = await getEmbeddingsBatch(chunks.map((c) => c.text), "passage");
    if (!Array.isArray(vectors) || vectors.length !== chunks.length) {
      throw new Error("embedding batch size mismatch");
    }

    const limiter = createLimiter(8);
    let inserted = 0;
    await Promise.all(
      chunks.map((c, idx) =>
        limiter(async () => {
          await insertDocumentWithEmbedding(c.text, {
            ...baseMeta,
            type: textType,
            chunk_index: idx,
            startTok: c.startTok,
            endTok: c.endTok,
          }, vectors[idx]);
          inserted++;
        })
      )
    );

    res.json({
      ok: true,
      chunks: inserted,
      table_chunks: tableChunks,
      image_chunks: imageChunks,
      stored: saved.relPath,
    });
  } catch (e) {
    console.error("/upload error:", e);
    res.status(500).json({ error: e?.message || "server error" });
  }
}

export default { handleUpload };
