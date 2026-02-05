# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) system for enterprise document analysis. Users upload documents (PDF, TXT, DOCX, PPT, HWP), which are parsed, chunked, embedded, and stored. Questions are answered by retrieving relevant chunks via vector similarity and generating responses with an LLM. The project is Korean-language optimized (ko_KR UTF-8).

## Architecture

Three-service architecture orchestrated via `docker-compose.yml`:

- **Backend** (`server/`, port 8000) — Node.js/Express. Handles file uploads, document parsing (invokes Python Docling as subprocess), text chunking, embedding requests, vector search, and RAG query orchestration. Plain JavaScript (CommonJS). Key files: `index.js` (main server ~992 lines), `chunk.js` (token-based splitter), `rag_langchain.js` (LangChain RAG chain with intent classification).
- **Embedding service** (port 8001) — vLLM serving BAAI/bge-m3 (1024-dim vectors, max 8192 tokens). OpenAI-compatible API.
- **LLM service** (port 8002) — vLLM serving Qwen3-Coder-30B-A3B (AWQ quantized, max 12288 tokens). OpenAI-compatible API. Served model name: `qwen-coder`.
- **Frontend** (`web/`, Vite dev server) — React 19 + TypeScript + Tailwind CSS 4. Components: `UploadBox.tsx` (drag-drop file upload), `ChatView.tsx` (multi-turn conversation with streaming).
- **Database** — MySQL (mysql2/promise, charset utf8mb4). Connection config in root `.env`. Vector embeddings stored as JSON arrays and cached in-memory as Float32Array for fast dot-product search (`db/repo.js`).
- **MCP Server** (`mcp_server/`) — Model Context Protocol server using `@modelcontextprotocol/sdk`.

### Data Flow

Upload: File → SHA256 dedup → Docling OCR extraction (markdown + tables + images) → token-based chunking (1200 tokens, 200 overlap) → batch embedding via bge-m3 → MySQL insert + in-memory vector cache load.

Query: Question → query embedding → in-memory cosine similarity search (threshold 0.35–0.75, top-K) → LangChain RAG chain (intent classification → context formatting) → LLM generation → streamed response with source attribution.

### Database Tables

`documents` (content, metadata), `embeddings` (document_id, L2-normalized 1024-dim JSON vectors), `doc_assets` (images/tables with caption embeddings), `doc_tables` (TSV/MD/HTML table formats).

## Common Commands

### Frontend (web/)
```bash
cd web && npm install          # install dependencies
cd web && npm run dev          # dev server (Vite)
cd web && npm run build        # production build (tsc + vite build)
cd web && npm run lint         # ESLint
```

### Backend (server/)
```bash
cd server && npm install       # install dependencies
node server/index.js           # run server (reads server/.env)
```

Note: `server/package.json` has TS scripts (`npm run dev`, `npm run build`) but the actual source is plain JS — use `node index.js` directly.

### Docker (full stack)
```bash
docker compose up --build      # start backend + embedding + llm services
```
Requires NVIDIA GPU with docker GPU support. Embedding uses ~5% GPU memory, LLM uses ~85%.

### Python utilities
```bash
python server/ocr_once.py      # standalone Docling document extraction
python inference_chat_cli.py   # CLI chat with local Llama model + LoRA
```

## Environment Configuration

- **Root `.env`** — MySQL connection: `MY_HOST`, `MY_PORT`, `MY_USER`, `MY_PASS`, `MY_DB`
- **`server/.env`** — Service URLs (`EMB_URL`, `LLM_URL`), chunking params (`CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_TOKENS`, `MAX_CHUNKS_EMB`), table/OCR settings (`ENABLE_TABLE_INDEX`, `PDF_INFER_TABLES`, `MAX_TABLE_ROWS_EMB`)
- **`web/.env`** — `VITE_API_URL` (default `http://localhost:8000`)
- **`docker-compose.yml`** — overrides service URLs for container networking

## Key Implementation Details

- Vector search is in-memory (not a vector DB). `db/repo.js` loads all embeddings into Float32Array cache on startup, computes L2-normalized dot products for cosine similarity.
- Document parsing uses Python Docling invoked as a child process from Node.js — not a separate service.
- File upload limit is 100MB (Multer config). SHA256 is used for duplicate detection.
- Token counting uses `gpt-tokenizer` for accurate chunk boundaries.
- The LLM is accessed via vLLM's OpenAI-compatible API (not HuggingFace transformers directly).
- Concurrency is controlled with `p-limit` for batch embedding requests.
