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

/** SSE 스트리밍 요청 (POST /query/stream) */
export async function askQueryStream(
  body: any,
  onToken: (token: string) => void,
  opts?: { signal?: AbortSignal }
): Promise<{ sources: SourceMeta[]; rag_mode: string }> {
  const res = await fetch(`${API_BASE}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: opts?.signal,
  });

  if (!res.ok) {
    let detail = "";
    try { detail = await res.text(); } catch {}
    throw new Error(`HTTP ${res.status} ${res.statusText} ${detail}`.trim());
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let sources: SourceMeta[] = [];
  let rag_mode = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    let eventType = "message";
    for (const line of lines) {
      if (line.startsWith("event: ")) {
        eventType = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        const raw = line.slice(6);
        try {
          const json = JSON.parse(raw);
          if (eventType === "done") {
            sources = Array.isArray(json.sources) ? json.sources : [];
            rag_mode = json.rag_mode ?? "";
          } else if (eventType === "error") {
            throw new Error(json.error || "stream error");
          } else if (json.token !== undefined) {
            onToken(json.token);
          }
        } catch (e) {
          if (eventType === "error") throw e;
        }
        eventType = "message";
      }
    }
  }

  return { sources, rag_mode };
}