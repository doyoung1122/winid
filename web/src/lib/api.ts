export const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function uploadFile(file: File) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data?.error || `HTTP ${res.status}`);
  return data;
}

export async function* streamQuery(body: any) {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) {
    let text = "";
    try { text = await res.text(); } catch {}
    throw new Error(text || `HTTP ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let done = false;
  while (!done) {
    const r = await reader.read();
    done = r.done!;
    const chunk = decoder.decode(r.value || new Uint8Array(), { stream: !done });
    const parts = chunk.split("\n\n").filter(Boolean);
    for (const p of parts) {
      if (!p.startsWith("data:")) continue;
      const payload = p.replace(/^data:\s*/, "");
      if (payload === "[DONE]") return;
      yield payload;
    }
  }
}