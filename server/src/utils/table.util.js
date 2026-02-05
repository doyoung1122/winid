import { JSDOM } from "jsdom";
import { sanitizeCell } from "./text.util.js";

/**
 * Convert header and rows to Markdown table
 * @param {string[]} header - Table headers
 * @param {string[][]} rows - Table rows
 * @returns {string} Markdown table string
 */
export function toMarkdownTable(header, rows) {
  if (!header?.length) return "";
  const sep = header.map(() => "---");
  const lines = [
    `| ${header.join(" | ")} |`,
    `| ${sep.join(" | ")} |`,
    ...rows.slice(0, 30).map((r) => `| ${r.map(sanitizeCell).join(" | ")} |`),
  ];
  return lines.join("\n");
}

/**
 * Normalize table metadata from various sources (unstructured/docling)
 * @param {Object} t - Raw table object
 * @returns {Object} Normalized table with header, rows, tsv, md, html
 */
export function normalizeTableMeta(t) {
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
            (r || []).map((c) => `<td>${sanitizeCell(c)}</td>`).join("") +
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

  const tsv = [header, ...rows].map((r) => r.map(sanitizeCell).join("\t")).join("\n");
  const md = toMarkdownTable(header, rows);

  const n_rows_hint = Number.isFinite(t?.n_rows) ? t.n_rows : rows.length;
  const n_cols_hint = Number.isFinite(t?.n_cols)
    ? t.n_cols
    : header.length || (rows[0]?.length || 0);

  return { header, rows, tsv, md, html, n_rows_hint, n_cols_hint };
}

export default {
  toMarkdownTable,
  normalizeTableMeta,
};
