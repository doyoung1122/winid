/**
 * BM25 keyword search for Hybrid Search (Vector + BM25 → RRF)
 * Optimized for Korean fire investigation documents.
 *
 * Tokenization: whitespace/punct split + Hangul character bigrams
 * e.g. "발화열원" → ["발화열원", "발화", "화열", "열원"]
 * BM25 parameters: k1=1.5, b=0.75
 */

const BM25_K1 = 1.5;
const BM25_B  = 0.75;

// ---- Tokenizer ----

function tokenize(text) {
  if (!text) return [];
  const tokens = [];
  const words = text
    .toLowerCase()
    .split(/[\s\u3000.,!?;:()\[\]{}'"`「」『』【】、。·•\-_/\\|<>]+/);

  for (const word of words) {
    if (!word) continue;
    tokens.push(word);
    // Hangul bigrams (가-힣: U+AC00–U+D7A3)
    for (let i = 0; i < word.length - 1; i++) {
      const c1 = word.charCodeAt(i);
      const c2 = word.charCodeAt(i + 1);
      if (c1 >= 0xac00 && c1 <= 0xd7a3 && c2 >= 0xac00 && c2 <= 0xd7a3) {
        tokens.push(word.slice(i, i + 2));
      }
    }
  }
  return tokens;
}

// ---- BM25 Class ----

class BM25 {
  constructor() {
    /** @type {Array<{id:string, content:string, metadata:object, tf:Map<string,number>, dl:number}>} */
    this.docs   = [];
    this.idf    = new Map();
    this.avgdl  = 0;
    this._dirty = false;
  }

  addDoc(id, content, metadata = {}) {
    const tokens = tokenize(content);
    const tf     = new Map();
    for (const t of tokens) tf.set(t, (tf.get(t) || 0) + 1);
    this.docs.push({ id, content, metadata, tf, dl: tokens.length });
    this._dirty = true;
  }

  build() {
    const N = this.docs.length;
    if (N === 0) return;
    this.avgdl = this.docs.reduce((s, d) => s + d.dl, 0) / N;

    const df = new Map();
    for (const doc of this.docs) {
      for (const t of doc.tf.keys()) df.set(t, (df.get(t) || 0) + 1);
    }

    this.idf = new Map();
    for (const [t, d] of df.entries()) {
      // Robertson IDF: ln((N - d + 0.5) / (d + 0.5) + 1)
      this.idf.set(t, Math.log((N - d + 0.5) / (d + 0.5) + 1));
    }
    this._dirty = false;
  }

  /**
   * @param {string} query
   * @param {number} k
   * @returns {Array<{id, content, metadata, score}>}
   */
  search(query, k = 20) {
    if (this._dirty) this.build();
    if (this.docs.length === 0) return [];

    const qTokens = [...new Set(tokenize(query))];
    const scores  = [];

    for (const doc of this.docs) {
      let score = 0;
      for (const t of qTokens) {
        const idf = this.idf.get(t);
        if (!idf || idf <= 0) continue;
        const tf = doc.tf.get(t) || 0;
        if (tf === 0) continue;
        const denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * (doc.dl / this.avgdl));
        score += idf * ((tf * (BM25_K1 + 1)) / denom);
      }
      if (score > 0) scores.push({ id: doc.id, content: doc.content, metadata: doc.metadata, score });
    }

    scores.sort((a, b) => b.score - a.score);
    return scores.slice(0, k);
  }
}

// ---- Singleton Index ----

let _bm25 = null;
let _bm25Promise = null;

/**
 * Build BM25 index lazily from ChromaDB text documents.
 * Promise-deduplicated to prevent concurrent builds.
 */
async function ensureBM25Index() {
  if (_bm25) return _bm25;
  if (_bm25Promise) return _bm25Promise;

  _bm25Promise = (async () => {
    const { getAllDocuments } = await import("../../../db/chroma.js");
    console.log("[BM25] Building index from ChromaDB...");
    const docs = await getAllDocuments({ textOnly: true });
    const bm25 = new BM25();
    for (const doc of docs) bm25.addDoc(doc.id, doc.content, doc.metadata);
    bm25.build();
    _bm25 = bm25;
    _bm25Promise = null;
    console.log(`[BM25] Index built: ${bm25.docs.length} docs`);
    return bm25;
  })().catch((err) => {
    console.warn("[BM25] Build failed:", err.message);
    _bm25Promise = null;
    return null;
  });

  return _bm25Promise;
}

/**
 * Search BM25 index for top-k keyword matches.
 * @param {string} query
 * @param {number} k
 * @returns {Promise<Array<{id, content, metadata, score}>>}
 */
export async function searchBM25(query, k = 20) {
  const bm25 = await ensureBM25Index();
  if (!bm25) return [];
  return bm25.search(query, k);
}

/**
 * Add a newly inserted document to the BM25 index (if already built).
 * Safe to call even if index isn't built yet.
 */
export function addToBM25Index(id, content, metadata = {}) {
  if (!_bm25) return; // Not yet built; will include on lazy build
  _bm25.addDoc(id, content, metadata);
  // _dirty=true; build() called lazily on next search
}

/** Invalidate BM25 index (e.g., after bulk re-ingest). */
export function resetBM25Index() {
  _bm25 = null;
  _bm25Promise = null;
}
