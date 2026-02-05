/**
 * Clean and normalize text
 * @param {string} s - Input text
 * @returns {string} Cleaned text
 */
export function cleanText(s) {
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

/**
 * Normalize a numeric value with unit
 * @param {any} v - Value to normalize
 * @returns {Object|null} Normalized value object
 */
export function normalizeNumber(v) {
  if (v == null) return null;
  const s = String(v);
  const m = s.match(/^([\d.,+-]+)\s*([A-Za-z%]*)$/);
  if (!m) return { raw: s };
  const num = parseFloat(m[1].replace(/,/g, ""));
  const unit = m[2] || "";
  return { value: isNaN(num) ? null : num, unit, raw: s };
}

/**
 * Convert a table row to a natural language sentence
 * @param {string} tableTitle - Table title/caption
 * @param {string[]} headers - Column headers
 * @param {string[]} row - Row values
 * @returns {string} Natural language sentence
 */
export function rowToSentence(tableTitle, headers, row) {
  const parts = headers.map((h, i) => `${h}=${row[i] ?? ""}`);
  return `Table: ${tableTitle} | ${parts.join("; ")}`;
}

/**
 * Sanitize a table cell value
 * @param {string} s - Cell value
 * @returns {string} Sanitized value
 */
export function sanitizeCell(s = "") {
  return String(s || "").replace(/\s+/g, " ").trim();
}

/**
 * Safe JSON parse with fallback
 * @param {string|any} s - String to parse or value
 * @param {any} fallback - Fallback value
 * @returns {any} Parsed value or fallback
 */
export function safeJsonParse(s, fallback = null) {
  try {
    return typeof s === "string" ? JSON.parse(s) : s ?? fallback;
  } catch {
    return fallback;
  }
}

export default {
  cleanText,
  normalizeNumber,
  rowToSentence,
  sanitizeCell,
  safeJsonParse,
};
