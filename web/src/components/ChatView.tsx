import { useEffect, useRef, useState } from "react";
import { streamQuery } from "../lib/api";

type Turn = { role: "user" | "assistant"; content: string };

export default function ChatView() {
  const [input, setInput] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [streaming, setStreaming] = useState(false);
  const scrollerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    scrollerRef.current?.scrollTo({ top: scrollerRef.current.scrollHeight, behavior: "smooth" });
  }, [turns, streaming]);

  const ask = async () => {
    const question = input.trim();
    if (!question || streaming) return;

    setInput("");
    setTurns((t) => [...t, { role: "user", content: question }, { role: "assistant", content: "" }]);
    setStreaming(true);

    try {
      const history = turns.map(({ role, content }) => ({ role, content })).filter(x => x.content);
      const gen = streamQuery({ question, history, match_count: 8, threshold: 0.5 });

      for await (const token of gen) {
        setTurns((prev) => {
          const last = prev[prev.length - 1];
          if (!last || last.role !== "assistant") return prev;
          return prev.slice(0, -1).concat({ ...last, content: last.content + token });
        });
      }
    } catch (e: any) {
      setTurns((prev) => prev.concat({ role: "assistant", content: `❌ 오류: ${e?.message || "알 수 없는 오류"}` }));
    } finally {
      setStreaming(false);
    }
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask();
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto mt-10">
      <h2 className="text-xl font-semibold mb-3">💬 문서 기반 채팅</h2>

      <div ref={scrollerRef} className="h-96 overflow-auto rounded-xl border border-gray-200 bg-white p-4 space-y-4">
        {turns.length === 0 && (
          <div className="text-sm text-gray-500">
          </div>
        )}
        {turns.map((t, i) => (
          <div key={i} className={`flex ${t.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={[
              "max-w-[80%] rounded-2xl px-4 py-2 text-sm whitespace-pre-wrap",
              t.role === "user" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-900",
            ].join(" ")}>
              {t.content || (t.role === "assistant" && streaming ? "타이핑 중…" : "")}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 flex gap-2">
        <input
          className="flex-1 rounded-lg border border-gray-300 px-3 py-2 focus:ring-2 focus:ring-blue-500"
          placeholder="질문을 입력하세요. Enter로 전송"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          disabled={streaming}
        />
        <button
          onClick={ask}
          disabled={streaming || !input.trim()}
          className="rounded-lg bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 disabled:bg-gray-400"
        >
          보내기
        </button>
      </div>
    </div>
  );
}