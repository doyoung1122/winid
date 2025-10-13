import UploadBox from "./components/UploadBox";
import ChatView from "./components/ChatView";

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="border-b bg-white">
        <div className="mx-auto max-w-5xl px-4 py-3 flex items-center justify-between">
          <h1 className="text-lg font-semibold">RAG Demo</h1>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-4 py-8">
        <UploadBox />
        <ChatView />
      </main>
    </div>
  );
}