import express from "express";
import cors from "cors";
import path from "path";
import fs from "fs";

import {
  PORT,
  ALLOWED_ORIGINS,
  EMB_URL,
  LLM_URL,
  ALWAYS_UNSTRUCTURED,
  FAST_MODE,
  RENDER_PAGES,
  MAX_TABLE_ROWS_EMB,
  MAX_CAPTION_PAGES,
  UPLOAD_DIR,
  PUBLIC_DIR,
  HWP2TXT_EXE,
} from "./config/env.js";

// Routes
import healthRoute from "./routes/health.route.js";
import uploadRoute from "./routes/upload.route.js";
import queryRoute from "./routes/query.route.js";

// =========================
// Express App Setup
// =========================
const app = express();

// =========================
// CORS Configuration
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

// JSON body parser
app.use(express.json({ limit: "10mb" }));

// =========================
// Ensure Directories Exist
// =========================
const ASSET_DIR = path.join(PUBLIC_DIR, "doc-assets");
if (!fs.existsSync(ASSET_DIR)) fs.mkdirSync(ASSET_DIR, { recursive: true });
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

// =========================
// Static File Serving
// =========================
app.use("/assets", express.static(PUBLIC_DIR, { maxAge: "1y" }));
app.use("/uploads", express.static(UPLOAD_DIR, { maxAge: "1y" }));

// =========================
// Routes
// =========================
app.use(healthRoute);
app.use(uploadRoute);
app.use(queryRoute);

// =========================
// Start Server
// =========================
app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server running on http://0.0.0.0:${PORT}`);
  console.log(`   - EMB_URL = ${EMB_URL}`);
  console.log(`   - LLM_URL = ${LLM_URL}`);
  console.log(`   - ALWAYS_UNSTRUCTURED = ${ALWAYS_UNSTRUCTURED}`);
  console.log(`   - FAST_MODE = ${FAST_MODE}`);
  console.log(`   - RENDER_PAGES = ${RENDER_PAGES}`);
  console.log(`   - MAX_TABLE_ROWS_EMB = ${MAX_TABLE_ROWS_EMB}`);
  console.log(`   - MAX_CAPTION_PAGES = ${MAX_CAPTION_PAGES}`);
  console.log(`   - UPLOAD_DIR = ${UPLOAD_DIR}`);
  if (HWP2TXT_EXE) console.log(`   - HWP2TXT_EXE = ${HWP2TXT_EXE}`);
});

export default app;
