export const API_BASE =
  import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function uploadFile(file: File) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data?.error || `HTTP ${res.status}`);
  return data;
}

// ---------- REST 전용 ----------

export type SourceMeta = {
  label: string;
  filepath: string;
  filename?: string;
  page?: number;
  similarity?: number;
};

export type AskQueryResult = {
  ok: boolean;
  answer: string;
  sources: SourceMeta[];
};

/** REST(JSON) 단발 요청 */
export async function askQuery(
  body: any,
  opts?: { signal?: AbortSignal }
): Promise<AskQueryResult> {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "application/json",
    },
    body: JSON.stringify(body),
    signal: opts?.signal,
  });

  if (!res.ok) {
    let detail = "";
    try { detail = await res.text(); } catch {}
    throw new Error(`HTTP ${res.status} ${res.statusText} ${detail}`.trim());
  }

  const json = await res.json();
  return {
    ok: !!json?.ok,
    answer: String(json?.answer ?? ""),
    sources: Array.isArray(json?.sources) ? json.sources : [],
  };
}