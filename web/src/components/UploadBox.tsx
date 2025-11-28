import { useCallback, useMemo, useRef, useState } from "react";

type Status = "idle" | "hover" | "uploading" | "success" | "error";

function getDefaultApiBase() {
  const { protocol, hostname } = window.location;
  return `${protocol}//${hostname}:8000`;
}

export default function UploadBox() {
  const API_BASE = useMemo(() => {
    const env = (import.meta as any).env?.VITE_API_URL as string | undefined;
    const base = (env && env.trim()) || getDefaultApiBase();
    return base.replace(/\/$/, "");
  }, []);

  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [message, setMessage] = useState("");
  const inputRef = useRef<HTMLInputElement | null>(null);

  const onSelect = (f: File | null) => {
    if (!f) return;

    const name = f.name.toLowerCase();
    const type = (f.type || "").toLowerCase();
    const okExt = /\.(pdf|txt|md|ppt|pptx|doc|docx|hwp|hwpx)$/i.test(name);
    const okMime =
      type.includes("pdf") ||
      type.includes("text") ||
      type.includes("markdown") ||
      type.includes("presentation") ||
      type.includes("word") ||
      type.includes("officedocument") ||
      type.includes("hangul"); // 일부 HWP MIME

    const okType = okExt || okMime;

    if (!okType) {
      setStatus("error");
      setMessage("PDF / TXT / MD / PPT / DOC / HWP 지원");
      setFile(null);
      return;
    }
    if (f.size > 100 * 1024 * 1024) {
      setStatus("error");
      setMessage("파일 용량은 최대 100MB 입니다.");
      setFile(null);
      return;
    }
    setFile(f);
    setStatus("idle");
    setMessage("");
  };

  const handleUpload = async () => {
    if (!file) return;
    setStatus("uploading");
    setMessage("업로드 중… 문서를 처리하고 임베딩을 저장합니다.");

    // 타임아웃
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 120_000);

    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: form,
        signal: controller.signal,
      });

      const data = await res.json().catch(() => ({}));
      if (res.ok && data?.ok) {
        setStatus("success");
        setMessage(`✅ 완료: ${data?.chunks ?? "?"}개의 청크가 저장되었습니다.`);
        setFile(null);
        if (inputRef.current) inputRef.current.value = "";
      } else {
        setStatus("error");
        setMessage(`❌ 업로드 실패: ${data?.error || `HTTP ${res.status}`}`);
      }
    } catch (e: any) {
      if (e?.name === "AbortError") {
        setStatus("error");
        setMessage("❌ 업로드 시간 초과(120초). 네트워크 또는 서버 상태를 확인해주세요.");
      } else {
        setStatus("error");
        setMessage(`❌ 서버에 연결할 수 없습니다. (${e?.message || "network error"})`);
      }
    } finally {
      clearTimeout(timeout);
    }
  };

  // Drag & Drop
  const onDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    onSelect(e.dataTransfer.files?.[0] ?? null);
  }, []);
  const onDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setStatus((s) => (s !== "uploading" ? "hover" : s));
  };
  const onDragLeave = () => {
    setStatus((s) => (s === "hover" ? "idle" : s));
  };

  const isUploading = status === "uploading";
  const isSuccess = status === "success";
  const isError = status === "error";
  const isHover = status === "hover";

  const showLocalhostWarn =
    /localhost|127\.0\.0\.1/i.test(API_BASE) &&
    !/localhost|127\.0\.0\.1/i.test(window.location.hostname);

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div className="mb-4 text-center">
        <h1 className="text-2xl font-bold tracking-tight text-gray-900">📚 문서 업로더</h1>
        <div className="mt-2 text-xs text-gray-500">
          {showLocalhostWarn && (
            <span className="ml-2 inline-flex items-center rounded-full bg-amber-100 px-2 py-0.5 text-amber-800">
              ⚠️ 현재 API가 localhost로 설정되어 있어 외부(모바일)에서 업로드가 실패합니다.
            </span>
          )}
        </div>
      </div>

      <div
        className={[
          "rounded-2xl border bg-white p-6 shadow-sm transition-all",
          isHover ? "border-blue-400 ring-4 ring-blue-100" : "border-gray-200",
        ].join(" ")}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
      >
        <div className="flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed border-gray-300 px-6 py-10 text-center">
          <p className="text-sm text-gray-600">
            파일을 이곳에 <span className="font-semibold">드래그&드롭</span> 하거나
          </p>
          <div className="flex items-center gap-2">
            <label
              htmlFor="file"
              className="cursor-pointer rounded-lg bg-blue-600 px-4 py-2 text-white shadow hover:bg-blue-700 active:translate-y-px focus:outline-none focus:ring-4 focus:ring-blue-300"
            >
              파일 선택
            </label>
            <input
              id="file"
              ref={inputRef}
              type="file"
              className="hidden"
              onChange={(e) => onSelect(e.target.files?.[0] ?? null)}
              // 확장자 및 MIME 모두 허용 (브라우저별 MIME 편차 커버)
              accept={[
                ".pdf,.txt,.md,.ppt,.pptx,.doc,.docx,.hwp,.hwpx",
                "application/pdf",
                "text/plain",
                "text/markdown",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "application/vnd.ms-powerpoint",
                "application/haansofthwp",
                "application/x-hwp",
              ].join(",")}
            />
            <span className="text-xs text-gray-500">최대 100MB</span>
          </div>

          {file && (
            <div className="mt-3 rounded-md bg-gray-50 px-3 py-2 text-sm text-gray-700">
              선택된 파일: <b>{file.name}</b> ({Math.round(file.size / 1024)} KB)
            </div>
          )}
        </div>

        <div className="mt-6 flex items-center gap-3 justify-center">
          <button
            onClick={handleUpload}
            disabled={!file || isUploading}
            className={[
              "inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 font-medium text-white transition-all",
              !file || isUploading
                ? "bg-gray-300 cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-700 active:translate-y-px",
              "focus:outline-none focus:ring-4 focus:ring-blue-300",
            ].join(" ")}
          >
            {isUploading ? "업로드 중…" : "업로드"}
          </button>

          {isSuccess && (
            <span className="select-none rounded-full bg-emerald-100 px-3 py-1 text-sm font-medium text-emerald-700">
              처리 완료
            </span>
          )}
          {isError && (
            <span className="select-none rounded-full bg-red-100 px-3 py-1 text-sm font-medium text-red-700">
              오류 발생
            </span>
          )}
        </div>

        {message && (
          <div
            className={[
              "mt-4 rounded-lg px-4 py-3 text-sm text-center",
              isError
                ? "bg-red-50 text-red-700"
                : isSuccess
                ? "bg-emerald-50 text-emerald-700"
                : "bg-gray-50 text-gray-700",
            ].join(" ")}
          >
            {message}
          </div>
        )}
      </div>
    </div>
  );
}