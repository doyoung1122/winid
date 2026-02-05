import { useEffect, useRef, useState } from "react";
import { askQuery, type SourceMeta } from "../lib/api";

type Role = "user" | "assistant";
type Turn = { role: Role; content: string; sources?: SourceMeta[] };

export default function ChatView() {
  const [input, setInput] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [loading, setLoading] = useState(false);
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    scrollerRef.current?.scrollTo({ top: scrollerRef.current.scrollHeight, behavior: "smooth" });
  }, [turns, loading]);

  const send = async (question: string) => {
    // 기존 턴들로 history 구성
    const history = turns.map(({ role, content }) => ({ role, content })).filter(x => x.content);

    // UI에 미리 표시
    setTurns((t) => [...t, { role: "user", content: question }, { role: "assistant", content: "답변 생성 중..." }]);
    setLoading(true);

    // abort 교체
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      // 짧은 질문 자동 임계 완화
      const wordCount = question.trim().split(/\s+/).filter(Boolean).length;
      const threshold = wordCount <= 3 ? 0.45 : 0.75;

      const { answer, sources } = await askQuery(
        { question, history, match_count: 5, threshold, strict: true },
        { signal: controller.signal }
      );

      // 마지막 assistant 메시지 교체
      setTurns((prev) => {
        const updated = [...prev];
        const lastIdx = updated.length - 1;
        if (lastIdx >= 0 && updated[lastIdx].role === "assistant") {
          updated[lastIdx] = { role: "assistant", content: answer || "모릅니다.", sources };
        }
        return updated;
      });
    } catch (e: any) {
      if (e?.name !== "AbortError") {
        setTurns((prev) => {
          const updated = [...prev];
          const lastIdx = updated.length - 1;
          const msg = `❌ 오류: ${e?.message || "알 수 없는 오류"}`;
          if (lastIdx >= 0 && updated[lastIdx].role === "assistant") {
            updated[lastIdx] = { ...updated[lastIdx], content: msg };
          } else {
            updated.push({ role: "assistant", content: msg });
          }
          return updated;
        });
      }
    } finally {
      setLoading(false);
      if (abortRef.current === controller) abortRef.current = null;
    }
  };

  const onSendOrStop = () => {
    if (loading) {
      abortRef.current?.abort();
      return;
    }
    const question = input.trim();
    if (!question) return;
    setInput("");
    send(question);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSendOrStop();
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto mt-10">
      <h2 className="text-xl font-semibold mb-3">💬 문서 기반 채팅</h2>

      <div ref={scrollerRef} className="h-96 overflow-auto rounded-xl border border-gray-200 bg-white p-4 space-y-4">
        {turns.map((t, i) => (
          <div key={i} className={`flex ${t.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={[
                "max-w-[80%] rounded-2xl px-4 py-2 text-sm whitespace-pre-wrap",
                t.role === "user" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-900",
              ].join(" ")}
            >
              {t.content}

              {t.role === "assistant" && t.sources?.length ? (() => {
                const src = t.sources ?? [];
                return (
                  <div className="mt-2 text-xs text-gray-500">
                    출처:&nbsp;
                    {src.slice(0, 3).map((s, idx) => (
                      <span key={`${s.filepath}-${idx}`}>
                        {s.filename || s.filepath}
                        {typeof s.page === "number" ? `(p.${s.page})` : ""}
                        {typeof s.similarity === "number" ? ` (유사도 ${Number(s.similarity).toFixed(2)})` : ""}
                        {idx < Math.min(2, src.length - 1) ? ", " : ""}
                      </span>
                    ))}
                  </div>
                );
              })() : null}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 flex gap-2">
        <input
          className="flex-1 rounded-lg border border-gray-300 px-3 py-2 focus:ring-2 focus:ring-blue-500"
          placeholder="질문을 입력하세요. Enter로 전송 / 중지"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
        />
        <button
          onClick={onSendOrStop}
          className={`rounded-lg px-4 py-2 text-white ${loading ? "bg-red-600 hover:bg-red-700" : "bg-blue-600 hover:bg-blue-700"}`}
        >
          {loading ? "중지" : "보내기"}
        </button>
      </div>
    </div>
  );
}