const { pool } = require("./mysql.js");

// ---- helpers ----
function l2Normalize(v) {
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
  const n = Math.sqrt(sum) || 1;
  if (n === 1) return v.slice();
  const out = new Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
  return out;
}
function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
function ensureDim(v, d = 1024) {
  if (!Array.isArray(v) || v.length !== d) {
    throw new Error(
      `embedding dimension mismatch (expected ${d}, got ${Array.isArray(v) ? v.length : "NA"})`
    );
  }
}
function safeJsonParse(s, fallback = null) {
  try {
    return typeof s === "string" ? JSON.parse(s) : s ?? fallback;
  } catch {
    return fallback;
  }
}

// ---- inserts ----
async function insertDocumentWithEmbedding(content, metadata, embedding) {
  ensureDim(embedding, 1024);
  const normVec = l2Normalize(embedding);

  const conn = await pool.getConnection();
  try {
    await conn.beginTransaction();

    const [r1] = await conn.execute(
      `INSERT INTO documents(content, metadata)
       VALUES (?, ?)`,
      [content, JSON.stringify(metadata ?? {})]
    );
    const docId = r1.insertId;

    await conn.execute(
      `INSERT INTO embeddings(document_id, embedding)
       VALUES (?, ?)`,
      [docId, JSON.stringify(normVec)]
    );

    await conn.commit();
    return docId;
  } catch (err) {
    try { await conn.rollback(); } catch {}
    console.error("❌ insertDocumentWithEmbedding 오류:", err);
    throw err;
  } finally {
    conn.release();
  }
}

async function insertDocAsset(a) {
  const conn = await pool.getConnection();
  try {
    const cap = Array.isArray(a.caption_emb) ? l2Normalize(a.caption_emb) : null;

    const [r] = await conn.execute(
      `INSERT INTO doc_assets(sha256, filepath, page, type, image_url, caption_text, caption_emb, meta)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        a.sha256,
        a.filepath,
        a.page ?? null,
        a.type,
        a.image_url ?? null,
        a.caption_text ?? null,
        cap ? JSON.stringify(cap) : null,
        JSON.stringify(a.meta ?? {}),
      ]
    );
    return r.insertId;
  } catch (err) {
    console.error("❌ insertDocAsset 오류:", err);
    throw err;
  } finally {
    conn.release();
  }
}

async function insertDocTable(t) {
  const conn = await pool.getConnection();
  try {
    const [r] = await conn.execute(
      `INSERT INTO doc_tables(asset_id, n_rows, n_cols, tsv, md, html)
       VALUES (?, ?, ?, ?, ?, ?)`,
      [t.asset_id, t.n_rows ?? null, t.n_cols ?? null, t.tsv ?? null, t.md ?? null, t.html ?? null]
    );
    return r.insertId;
  } catch (err) {
    console.error("❌ insertDocTable 오류:", err);
    throw err;
  } finally {
    conn.release();
  }
}

// ---- match (cosine = dot; 저장 시 정규화 전제) ----
async function matchDocuments(queryEmbedding, { k = 8, threshold = 0.7 } = {}) {
  ensureDim(queryEmbedding, 1024);
  const q = l2Normalize(queryEmbedding);

  const [rows] = await pool.query(`
    SELECT d.id, d.content, d.metadata, e.embedding
    FROM embeddings e
    JOIN documents d ON d.id = e.document_id
  `);

  const scored = rows
    .map((r) => {
      const emb = safeJsonParse(r.embedding, []);
      // 메타데이터도 파싱
      const meta = safeJsonParse(r.metadata, {});
      return {
        id: r.id,
        content: r.content,
        metadata: meta,
        similarity: emb.length === q.length ? dot(q, emb) : -1,
      };
    })
    .filter((x) => x.similarity >= threshold)
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, k);

  return scored;
}

module.exports = {
  insertDocumentWithEmbedding,
  insertDocAsset,
  insertDocTable,
  matchDocuments,
};