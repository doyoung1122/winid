import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { askQueryStream, type SourceMeta } from "../lib/api";

const API_BASE = (() => {
  const env = (import.meta as any).env?.VITE_API_URL as string | undefined;
  if (env?.trim()) return env.trim().replace(/\/$/, "");
  const { protocol, hostname } = window.location;
  return `${protocol}//${hostname}:8000`;
})();

type Role = "user" | "assistant";
type Turn = { role: Role; content: string; sources?: SourceMeta[] };

function getUrlParams() {
  const p = new URLSearchParams(window.location.search);
  return {
    room_id: p.get("room_id") ?? undefined,
    mem_id:  p.get("mem_id")  ?? undefined,
  };
}

export default function ChatView() {
  const [input, setInput] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [loading, setLoading] = useState(false);
  const scrollerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { room_id, mem_id } = getUrlParams();

  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<"idle" | "uploading" | "success" | "error">("idle");
  const [uploadMsg, setUploadMsg] = useState("");

  useEffect(() => {
    scrollerRef.current?.scrollTo({ top: scrollerRef.current.scrollHeight, behavior: "smooth" });
  }, [turns, loading]);

  const handleFileSelect = (f: File | null) => {
    if (!f) return;
    const okExt = /\.(pdf|txt|md|ppt|pptx|doc|docx|hwp|hwpx)$/i.test(f.name);
    if (!okExt) { setUploadStatus("error"); setUploadMsg("PDF / TXT / MD / PPT / DOC / HWP 지원"); return; }
    if (f.size > 100 * 1024 * 1024) { setUploadStatus("error"); setUploadMsg("최대 100MB"); return; }
    setUploadFile(f);
    setUploadStatus("idle");
    setUploadMsg("");
  };

  const handleUpload = async () => {
    if (!uploadFile) return;
    setUploadStatus("uploading");
    setUploadMsg("업로드 중…");
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 120_000);
    try {
      const form = new FormData();
      form.append("file", uploadFile);
      const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: form, signal: controller.signal });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data?.ok) {
        setUploadStatus("success");
        setUploadMsg(`✅ ${data?.chunks ?? "?"}개 청크 저장됨`);
        setUploadFile(null);
        if (fileInputRef.current) fileInputRef.current.value = "";
      } else {
        setUploadStatus("error");
        setUploadMsg(`❌ ${data?.error || `HTTP ${res.status}`}`);
      }
    } catch (e: any) {
      setUploadStatus("error");
      setUploadMsg(e?.name === "AbortError" ? "❌ 시간 초과" : `❌ ${e?.message}`);
    } finally {
      clearTimeout(timeout);
    }
  };

  const clearUpload = () => {
    setUploadFile(null);
    setUploadStatus("idle");
    setUploadMsg("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const send = async (question: string) => {
    const history = turns.map(({ role, content }) => ({ role, content })).filter(x => x.content);
    setTurns(t => [...t, { role: "user", content: question }, { role: "assistant", content: "" }]);
    setLoading(true);
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const body = room_id && mem_id
      ? { question, room_id, mem_id, match_count: 5 }
      : { question, history, match_count: 5 };
    try {
      const { sources } = await askQueryStream(
        body,
        (token) => {
          setTurns(prev => {
            const updated = [...prev];
            const last = updated.length - 1;
            if (last >= 0 && updated[last].role === "assistant") {
              updated[last] = { ...updated[last], content: updated[last].content + token };
            }
            return updated;
          });
        },
        { signal: controller.signal }
      );
      setTurns(prev => {
        const updated = [...prev];
        const last = updated.length - 1;
        if (last >= 0 && updated[last].role === "assistant") {
          updated[last] = { ...updated[last], sources };
        }
        return updated;
      });
    } catch (e: any) {
      if (e?.name !== "AbortError") {
        setTurns(prev => {
          const updated = [...prev];
          const last = updated.length - 1;
          const msg = `❌ 오류: ${e?.message || "알 수 없는 오류"}`;
          if (last >= 0 && updated[last].role === "assistant") {
            updated[last] = { ...updated[last], content: msg };
          }
          return updated;
        });
      }
    } finally {
      setLoading(false);
      if (abortRef.current === controller) abortRef.current = null;
    }
  };

  const onSend = () => {
    if (loading) { abortRef.current?.abort(); return; }
    const q = input.trim();
    if (!q) return;
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    send(q);
  };

  const inputCard = (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden">
      {/* 상단: 텍스트 + 전송 버튼 */}
      <div className="flex items-end gap-3 px-4 pt-4 pb-3">
        <textarea
          ref={textareaRef}
          className="flex-1 resize-none bg-transparent text-sm outline-none leading-relaxed placeholder:text-gray-400 min-h-[72px] max-h-48"
          placeholder="화재 사례, 원인 분석, 통계 등 무엇이든 질문하세요…"
          rows={3}
          value={input}
          onChange={e => {
            setInput(e.target.value);
            e.target.style.height = "auto";
            e.target.style.height = Math.min(e.target.scrollHeight, 192) + "px";
          }}
          onKeyDown={e => {
            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onSend(); }
          }}
        />
        <button
          onClick={onSend}
          className={[
            "flex-shrink-0 w-9 h-9 rounded-full flex items-center justify-center text-white transition-colors self-end",
            loading ? "bg-gray-800 hover:bg-gray-700"
              : input.trim() ? "bg-gray-900 hover:bg-gray-700"
              : "bg-gray-300 cursor-not-allowed",
          ].join(" ")}
        >
          {loading ? (
            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="6" width="12" height="12" rx="2" />
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="19" x2="12" y2="5" /><polyline points="5 12 12 5 19 12" />
            </svg>
          )}
        </button>
      </div>

      {/* 구분선 */}
      <div className="border-t border-gray-100 mx-4" />

      {/* 하단: 첨부 영역 */}
      <div className="flex items-center px-4 py-2.5 gap-3 min-h-[44px]">
        <button
          onClick={() => fileInputRef.current?.click()}
          className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-700 transition-colors"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66L9.41 17.41a2 2 0 01-2.83-2.83l8.49-8.48" />
          </svg>
          {uploadFile ? uploadFile.name : "사진, 영상 파일을 여기로 드래그하거나 클릭하여 첨부"}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          accept=".pdf,.txt,.md,.ppt,.pptx,.doc,.docx,.hwp,.hwpx"
          onChange={e => handleFileSelect(e.target.files?.[0] ?? null)}
        />

        {uploadFile && (
          <>
            <button
              onClick={clearUpload}
              className="text-xs text-gray-400 hover:text-gray-600"
            >✕</button>
            <button
              onClick={handleUpload}
              disabled={uploadStatus === "uploading"}
              className="ml-1 rounded-full bg-gray-800 px-3 py-1 text-xs text-white hover:bg-gray-700 disabled:opacity-50 transition-colors"
            >
              {uploadStatus === "uploading" ? "업로드 중…" : "업로드"}
            </button>
          </>
        )}

        {uploadMsg && (
          <span className={`text-xs ${uploadStatus === "error" ? "text-red-500" : "text-emerald-600"}`}>
            {uploadMsg}
          </span>
        )}
      </div>
    </div>
  );

  return (
    <div className="flex h-screen bg-[#f2f2f2]">

      {/* ── 사이드바 (껍데기) ── */}
      <aside className="flex-shrink-0 w-12 bg-white border-r border-gray-200 flex flex-col items-center py-3 gap-1">
        <a
          target="_blank"
          rel="noopener noreferrer"
          className="w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-[11px] mb-3 hover:opacity-80 transition-opacity"
          style={{ backgroundColor: "#3289e7" }}
        >
          VF
        </a>
        <SideBtn title="홈" active>
          <svg viewBox="0 0 24 24" fill="currentColor" className="w-[18px] h-[18px]"><path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/></svg>
        </SideBtn>
        <SideBtn title="검색">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-[18px] h-[18px]"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        </SideBtn>
        <SideBtn title="대화 기록">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-[18px] h-[18px]"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
        </SideBtn>
        <div className="flex-1" />
        <SideBtn title="대화 초기화" onClick={() => setTurns([])}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-[18px] h-[18px]"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4h6v2"/></svg>
        </SideBtn>
      </aside>

      {/* ── 메인 ── */}
      <div className="flex flex-col flex-1 overflow-hidden">

        {/* 헤더 */}
        <header className="flex-shrink-0 h-11 bg-white border-b border-gray-200 flex items-center px-5 gap-2">
          <span className="font-semibold text-gray-800 text-sm">VFims AI</span>
          <div className="flex-1" />
          <div className="w-7 h-7 rounded-full bg-gray-800 flex items-center justify-center text-white text-xs font-bold select-none">관</div>
        </header>

        {/* 메시지 or 웰컴 */}
        <div ref={scrollerRef} className="flex-1 overflow-y-auto">
          {turns.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full px-4">
              <div className="w-full max-w-2xl">
                <div className="flex items-baseline justify-between mb-4">
                  <h1 className="text-2xl font-semibold text-gray-800">준비되었습니다. 시작해 볼까요?</h1>
                </div>
                {inputCard}
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">
              {turns.map((t, i) => (
                <div key={i} className={`flex ${t.role === "user" ? "justify-end" : "justify-start"}`}>
                  <div className={[
                    "max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
                    t.role === "user"
                      ? "bg-white text-gray-900 whitespace-pre-wrap shadow-sm border border-gray-100"
                      : "prose prose-sm max-w-none text-gray-900",
                  ].join(" ")}>
                    {t.role === "user" ? t.content : (
                      <>
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            table: ({ children }) => (
                              <table className="border-collapse border border-gray-200 my-2 text-xs">{children}</table>
                            ),
                            th: ({ children }) => (
                              <th className="border border-gray-200 px-2 py-1 bg-gray-50 font-semibold">{children}</th>
                            ),
                            td: ({ children }) => (
                              <td className="border border-gray-200 px-2 py-1">{children}</td>
                            ),
                          }}
                        >
                          {t.content || " "}
                        </ReactMarkdown>
                        {loading && i === turns.length - 1 && (
                          <span className="inline-block w-2 h-4 bg-gray-400 animate-pulse ml-0.5 align-middle rounded-sm" />
                        )}
                      </>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* 메시지 있을 때 하단 입력 */}
        {turns.length > 0 && (
          <div className="flex-shrink-0 px-4 pb-4 pt-2">
            <div className="max-w-3xl mx-auto">
              {inputCard}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function SideBtn({
  children,
  title,
  active,
  onClick,
}: {
  children: React.ReactNode;
  title?: string;
  active?: boolean;
  onClick?: () => void;
}) {
  return (
    <button
      title={title}
      onClick={onClick}
      className={[
        "w-9 h-9 rounded-lg flex items-center justify-center transition-colors",
        active ? "bg-gray-100 text-gray-800" : "text-gray-400 hover:bg-gray-100 hover:text-gray-700",
      ].join(" ")}
    >
      {children}
    </button>
  );
}
