import "dotenv/config";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

// __dirname replacement for ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// =========================
// Server Configuration
// =========================
export const PORT = Number(process.env.PORT || 8000);
export const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || "")
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);

// =========================
// Service URLs
// =========================
export const EMB_URL = (process.env.EMB_URL || "http://127.0.0.1:8001").replace(/\/$/, "");
export const LLM_URL = (process.env.LLM_URL || "http://127.0.0.1:8002").replace(/\/$/, "");

// =========================
// Model Names
// =========================
export const LLM_MODEL = process.env.LLM_MODEL || "qwen-coder";
export const EMB_MODEL = process.env.EMB_MODEL || "BAAI/bge-m3";

// =========================
// Document Processing
// =========================
export const DOC_PY =
  process.env.UNSTRUCT_PY ||
  (process.platform === "win32"
    ? "C:\\Users\\user\\anaconda3\\envs\\unstruct\\python.exe"
    : "python3");

export const HWP2TXT_EXE = process.env.HWP2TXT_EXE || "";
export const PUBLIC_BASE = (process.env.PUBLIC_BASE || `http://127.0.0.1:${PORT}`).replace(/\/$/, "");

// =========================
// Feature Flags
// =========================
export const ALWAYS_UNSTRUCTURED = String(process.env.ALWAYS_UNSTRUCTURED || "false") === "true";
export const FAST_MODE = String(process.env.FAST_MODE || "false") === "true";
export const RENDER_PAGES =
  String(
    process.env.RENDER_PAGES ??
      (process.env.ENABLE_PAGE_IMAGES === "1" ? "true" : "false")
  ) === "true";
export const ENABLE_TABLE_INDEX = String(process.env.ENABLE_TABLE_INDEX || "true") === "true";

// =========================
// Processing Limits
// =========================
export const MAX_TABLE_ROWS_EMB = Number(process.env.MAX_TABLE_ROWS_EMB || 50);
export const MAX_CAPTION_PAGES = Number(process.env.MAX_CAPTION_PAGES || 10);
export const CHUNK_SIZE_TOKENS = Number(process.env.CHUNK_SIZE_TOKENS || 800);
export const CHUNK_OVERLAP_TOKENS = Number(process.env.CHUNK_OVERLAP_TOKENS || 120);
export const MAX_CHUNKS_EMB = Number(process.env.MAX_CHUNKS_EMB || 0);

// =========================
// RAG Search Thresholds
// =========================
export const RETRIEVE_MIN = Number(process.env.RETRIEVE_MIN || 0.35);
export const USE_AS_CTX_MIN = Number(process.env.USE_AS_CTX_MIN || 0.60);
export const MIN_TOP3_AVG = Number(process.env.MIN_TOP3_AVG || 0.55);
export const TEXT_K = Number(process.env.TEXT_K || 5);
export const TABLE_K = Number(process.env.TABLE_K || 10);
export const IMAGE_K = Number(process.env.IMAGE_K || 4);

// =========================
// Directory Paths
// =========================
export const SERVER_ROOT = join(__dirname, "../..");
export const UPLOAD_DIR = join(SERVER_ROOT, "uploads");
export const PUBLIC_DIR = join(SERVER_ROOT, "public");
export const ASSET_DIR = join(PUBLIC_DIR, "doc-assets");

// =========================
// MySQL Configuration
// =========================
export const MY_HOST = process.env.MY_HOST || "localhost";
export const MY_PORT = Number(process.env.MY_PORT || 3306);
export const MY_USER = process.env.MY_USER || "root";
export const MY_PASS = process.env.MY_PASS || "";
export const MY_DB = process.env.MY_DB || "vfims";

// =========================
// Export all as config object
// =========================
export default {
  PORT,
  ALLOWED_ORIGINS,
  EMB_URL,
  LLM_URL,
  LLM_MODEL,
  EMB_MODEL,
  DOC_PY,
  HWP2TXT_EXE,
  PUBLIC_BASE,
  ALWAYS_UNSTRUCTURED,
  FAST_MODE,
  RENDER_PAGES,
  ENABLE_TABLE_INDEX,
  MAX_TABLE_ROWS_EMB,
  MAX_CAPTION_PAGES,
  CHUNK_SIZE_TOKENS,
  CHUNK_OVERLAP_TOKENS,
  MAX_CHUNKS_EMB,
  RETRIEVE_MIN,
  USE_AS_CTX_MIN,
  MIN_TOP3_AVG,
  TEXT_K,
  TABLE_K,
  IMAGE_K,
  SERVER_ROOT,
  UPLOAD_DIR,
  PUBLIC_DIR,
  ASSET_DIR,
  MY_HOST,
  MY_PORT,
  MY_USER,
  MY_PASS,
  MY_DB,
};
