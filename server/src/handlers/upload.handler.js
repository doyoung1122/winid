import path from "path";
import fs from "fs";
import fsp from "fs/promises";
import { spawn } from "child_process";
import chardet from "chardet";
import iconv from "iconv-lite";
import JSZip from "jszip";
import { parseStringPromise } from "xml2js";
import { fromBuffer } from "pdf2pic";

import {
  UPLOAD_DIR,
  DOC_PY,
  HWP2TXT_EXE,
  FAST_MODE,
  RENDER_PAGES,
  MAX_TABLE_ROWS_EMB,
  MAX_CAPTION_PAGES,
  ENABLE_TABLE_INDEX,
  CHUNK_SIZE_TOKENS,
  CHUNK_OVERLAP_TOKENS,
  MAX_CHUNKS_EMB,
  SERVER_ROOT,
} from "../config/env.js";

import { getEmbedding, getEmbeddingsBatch } from "../services/embedding.service.js";
import {
  insertDocumentWithEmbedding,
  insertDocAsset,
  insertDocTable,
} from "../services/vector.service.js";

import { cleanText, normalizeNumber, rowToSentence } from "../utils/text.util.js";
import {
  saveOriginalFile,
  saveDerivedImage,
  moveDoclingImageToUploads,
  createLimiter,
  safeBasename,
} from "../utils/file.util.js";
import { normalizeTableMeta } from "../utils/table.util.js";

import { chunkTextTokens } from "../../chunk.js";

// Ensure directories exist
const PUBLIC_DIR = path.join(SERVER_ROOT, "public");
const ASSET_DIR = path.join(PUBLIC_DIR, "doc-assets");
if (!fs.existsSync(ASSET_DIR)) fs.mkdirSync(ASSET_DIR, { recursive: true });
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

/**
 * Extract text using Docling OCR
 */
async function extractWithDoclingOnce(saved, filename) {
  const pyPath = path.join(SERVER_ROOT, "ocr_once.py");
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

/**
 * Extract HWPX (ZIP+XML)
 */
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

/**
 * Convert HWP to TXT via CLI
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

/**
 * POST /upload handler
 * @param {import('express').Request} req
 * @param {import('express').Response} res
 */
export async function handleUpload(req, res) {
  try {
    if (!req.file) return res.status(400).json({ error: "file required" });

    const mimetype = req.file.mimetype || "";
    const filepath = req.file.originalname || "unknown";

    const saved = await saveOriginalFile(req.file.buffer, filepath);

    let text = "",
      tablesForMeta = [],
      picturesForMeta = [],
      pageImageUrls = [];

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
      const effectiveMaxTableRowsEmb = FAST_MODE ? 0 : MAX_TABLE_ROWS_EMB;

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
            !FAST_MODE &&
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
            !FAST_MODE &&
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

    // FAST_MODE일 때 청크 개수 강제 제한
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
}

export default { handleUpload };
