import path from "path";
import fs from "fs";
import fsp from "fs/promises";
import crypto from "crypto";
import { UPLOAD_DIR, PUBLIC_BASE } from "../config/env.js";

/**
 * Ensure directory exists
 * @param {string} dir - Directory path
 */
export async function ensureDir(dir) {
  if (!fs.existsSync(dir)) {
    await fsp.mkdir(dir, { recursive: true });
  }
}

/**
 * Get safe basename from filename
 * @param {string} name - Original filename
 * @returns {string} Safe basename
 */
export function safeBasename(name = "file") {
  return name.replace(/[^\w.\-가-힣]+/g, "_").slice(0, 100);
}

/**
 * Get date string in YYYY/MM/DD format
 * @param {Date} d - Date object
 * @returns {string} Formatted date string
 */
export function yyyymmdd(d = new Date()) {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${y}/${m}/${dd}`;
}

/**
 * Save original uploaded file
 * @param {Buffer} buffer - File buffer
 * @param {string} originalName - Original filename
 * @returns {Promise<{absPath: string, relPath: string, sha: string}>}
 */
export async function saveOriginalFile(buffer, originalName) {
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

/**
 * Save derived image (page thumbnails, table images, etc.)
 * @param {Buffer} buffer - Image buffer
 * @param {Object} saved - Saved file info from saveOriginalFile
 * @param {string} kind - Image kind (pages, tables, pictures)
 * @param {string} filenameHint - Filename hint
 * @param {string} ext - File extension
 * @returns {Promise<{absPath: string, relPath: string, url: string}>}
 */
export async function saveDerivedImage(buffer, saved, kind, filenameHint, ext = ".jpg") {
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

/**
 * Move Docling-generated image to uploads directory
 * @param {string} localPath - Local path from Docling
 * @param {Object} saved - Saved file info
 * @param {string} kind - Image kind
 * @param {number} index - Image index
 * @returns {Promise<string|null>} URL or null
 */
export async function moveDoclingImageToUploads(localPath, saved, kind, index) {
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

/**
 * Create a simple concurrency limiter
 * @param {number} concurrency - Max concurrent tasks
 * @returns {Function} Limiter function
 */
export function createLimiter(concurrency = 4) {
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

export default {
  ensureDir,
  safeBasename,
  yyyymmdd,
  saveOriginalFile,
  saveDerivedImage,
  moveDoclingImageToUploads,
  createLimiter,
};
