import { pool } from "./mysql.js";

// 구조: { id: number, embedding: Float32Array, metadata: object }[]
let vectorCache = [];
let isCacheLoaded = false;

// ---- helpers ----
function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function l2Normalize(v) {
  if (!v || v.length === 0) throw new Error("l2Normalize: invalid vector");

  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
  const n = Math.sqrt(sum) || 1;

  if (n === 1) return new Float32Array(v);

  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
  return out;
}

function safeJsonParse(s, fallback = null) {
  try {
    return typeof s === "string" ? JSON.parse(s) : s ?? fallback;
  } catch {
    return fallback;
  }
}

// ---- cache loader ----
async function loadCache() {
  if (isCacheLoaded) return;
  console.log("Loading vectors into memory...");

  const [rows] = await pool.query(`
    SELECT d.id, d.metadata, e.embedding
    FROM embeddings e
    JOIN documents d ON d.id = e.document_id
  `);

  vectorCache = rows.map((r) => ({
    id: r.id,
    metadata: safeJsonParse(r.metadata, {}),
    // 여기서 미리 파싱하고 Float32Array로 박제해둠 (검색할 때 파싱 안 함)
    embedding: new Float32Array(safeJsonParse(r.embedding, [])),
  }));

  isCacheLoaded = true;
  console.log(`Loaded ${vectorCache.length} vectors.`);
}

// ---- schema helpers ----
async function ensureDocAssetsTables() {
  const conn = await pool.getConnection();
  try {
    // doc_assets 테이블 생성
    await conn.execute(`
      CREATE TABLE IF NOT EXISTS doc_assets (
        id INT AUTO_INCREMENT PRIMARY KEY,
        sha256 VARCHAR(64) NOT NULL,
        filepath VARCHAR(512),
        page INT,
        type VARCHAR(32) NOT NULL,
        image_url VARCHAR(1024),
        caption_text TEXT,
        caption_emb JSON,
        meta JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_sha256 (sha256),
        INDEX idx_type (type)
      ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    `);

    // doc_tables 테이블 생성
    await conn.execute(`
      CREATE TABLE IF NOT EXISTS doc_tables (
        id INT AUTO_INCREMENT PRIMARY KEY,
        asset_id INT NOT NULL,
        n_rows INT,
        n_cols INT,
        tsv MEDIUMTEXT,
        md MEDIUMTEXT,
        html MEDIUMTEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (asset_id) REFERENCES doc_assets(id) ON DELETE CASCADE
      ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    `);
  } finally {
    conn.release();
  }
}

let tablesEnsured = false;
async function ensureTables() {
  if (tablesEnsured) return;
  await ensureDocAssetsTables();
  tablesEnsured = true;
}

// ---- inserts ----
async function insertDocAsset({
  sha256,
  filepath,
  page,
  type,
  image_url,
  caption_text,
  caption_emb,
  meta,
}) {
  await ensureTables();

  // 캡션 임베딩이 있으면 정규화
  let embArray = null;
  if (caption_emb && caption_emb.length > 0) {
    const embNorm = l2Normalize(caption_emb);
    embArray = Array.from(embNorm);
  }

  const [result] = await pool.execute(
    `INSERT INTO doc_assets (sha256, filepath, page, type, image_url, caption_text, caption_emb, meta)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      sha256,
      filepath || null,
      page || null,
      type,
      image_url || null,
      caption_text || null,
      embArray ? JSON.stringify(embArray) : null,
      meta ? JSON.stringify(meta) : null,
    ]
  );

  return result.insertId;
}

async function insertDocTable({ asset_id, n_rows, n_cols, tsv, md, html }) {
  await ensureTables();

  const [result] = await pool.execute(
    `INSERT INTO doc_tables (asset_id, n_rows, n_cols, tsv, md, html)
     VALUES (?, ?, ?, ?, ?, ?)`,
    [asset_id, n_rows || null, n_cols || null, tsv || null, md || null, html || null]
  );

  return result.insertId;
}

async function insertDocumentWithEmbedding(content, metadata, embedding) {
  if (embedding.length !== 1024) throw new Error("Dimension mismatch");

  // 정규화 및 Float32 변환
  const embNorm = l2Normalize(embedding);
  // DB 저장을 위해 일반 배열로 변환 (JSON stringify용)
  const embArray = Array.from(embNorm);

  const conn = await pool.getConnection();
  try {
    await conn.beginTransaction();

    const [r1] = await conn.execute(
      `INSERT INTO documents(content, metadata) VALUES (?, ?)`,
      [content, JSON.stringify(metadata ?? {})]
    );
    const docId = r1.insertId;

    await conn.execute(
      `INSERT INTO embeddings(document_id, embedding) VALUES (?, ?)`,
      [docId, JSON.stringify(embArray)]
    );

    await conn.commit();

    if (isCacheLoaded) {
      vectorCache.push({
        id: docId,
        metadata: metadata ?? {},
        embedding: embNorm,
      });
    }

    return docId;
  } catch (err) {
    await conn.rollback();
    console.error("insert error:", err);
    throw err;
  } finally {
    conn.release();
  }
}

// ---- match ----
async function matchDocuments(queryEmbedding, options = {}) {
  if (!isCacheLoaded) await loadCache();

  const { k = 8, threshold = 0.7, types = null, sha256 = null } = options;

  // 1. 쿼리 벡터 정규화
  const q = l2Normalize(queryEmbedding);

  const scored = [];

  // 2. 메모리(Cache)에서 고속 검색 (DB 접근 X)
  for (const item of vectorCache) {
    // 필터링
    if (types && types.length > 0) {
      if (!types.includes(item.metadata.type)) continue;
    }
    if (sha256 && item.metadata.sha256 !== sha256) continue;

    // 유사도 계산 (Float32Array라 빠름)
    const sim = dot(q, item.embedding);

    if (sim >= threshold) {
      scored.push({
        id: item.id,
        metadata: item.metadata,
        similarity: sim,
      });
    }
  }

  // 3. 정렬 및 Top K 자르기
  scored.sort((a, b) => b.similarity - a.similarity);
  const topK = scored.slice(0, k);

  if (topK.length > 0) {
    const ids = topK.map((x) => x.id);
    const [rows] = await pool.query(`SELECT id, content FROM documents WHERE id IN (?)`, [ids]);
    // 결과 합치기
    for (const res of topK) {
      const match = rows.find((r) => r.id === res.id);
      if (match) res.content = match.content;
    }
  }

  return topK;
}

export {
  insertDocumentWithEmbedding,
  insertDocAsset,
  insertDocTable,
  matchDocuments,
  loadCache,
  l2Normalize,
};
