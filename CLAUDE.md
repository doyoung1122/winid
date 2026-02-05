# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context: Fire Investigation Knowledge Augmented System (RAG)

### Project Goal
Build a system that provides fire incident statistics and reports to an LLM via the MCP protocol. The system uses a **Hybrid Database Architecture** to combine structured metadata with semantic vector search.

### Key Architecture Rules (MUST FOLLOW)

1. **Hybrid Database Pattern:**
   - **Do NOT** store high-dimensional vectors in MySQL.
   - **Do NOT** store rich metadata in ChromaDB (keep it minimal).
   - Use `doc_id` as the foreign key to link MySQL records with ChromaDB vectors.
   - **Transaction Logic:** 1. Save Metadata (MySQL) → 2. Save Vector (ChromaDB) → 3. Commit.

2. **Textification Strategy (Crucial):**
   - Raw tabular data (e.g., JSON statistics) must be converted into **Natural Language Sentences** before embedding.
   - *Bad:* `{"date": "2024-01-01", "station": "종로소방서", "count": 3}`
   - *Good:* "2024년 1월 1일, 종로소방서에서 화재 3건이 발생했습니다."
   - **Why:** To improve semantic search accuracy for LLMs.

3. **Infrastructure:**
   - All services must be defined in `docker-compose.yml`.
   - Ensure all containers share the `vfims_net` network.

---

## Project Overview

A RAG (Retrieval-Augmented Generation) system specialized for **Fire Investigation**. Users upload documents (PDF, HWP) or fetch API statistics, which are parsed, textified, chunked, embedded, and stored in a Hybrid DB. Questions are answered by retrieving relevant chunks via ChromaDB and generating responses with an LLM via **MCP (Model Context Protocol)**.

## Architecture

Services orchestrated via `docker-compose.yml`:

| Service | Port | Description |
|---------|------|-------------|
| **Backend / MCP Server** | 8000 | Node.js/Express. File uploads, Docling OCR, Textification, MCP SDK integration |
| **MySQL 8.0** | 3306 | Metadata, structured stats, file info, vector embeddings |
| **Embedding (vLLM)** | 8001 | BAAI/bge-m3 (1024-dim vectors, max 8192 tokens) |
| **LLM (vLLM)** | 8002 | Qwen3-Coder-30B-A3B (max 12288 tokens) |
| **Frontend** | 5173 | React 19 + TypeScript (Low priority) |

### Project Structure (ESM)

```
server/
├── package.json              # "type": "module"
├── chunk.js                  # Token-based chunking
├── ocr_once.py              # Docling OCR script
├── src/
│   ├── index.js              # Express entry point
│   ├── config/
│   │   └── env.js            # Environment variables
│   ├── services/
│   │   ├── llm.service.js    # LLM calls (vLLM)
│   │   ├── embedding.service.js  # Embedding calls
│   │   └── vector.service.js # Vector search
│   ├── routes/
│   │   ├── upload.route.js
│   │   ├── query.route.js
│   │   └── health.route.js
│   ├── handlers/
│   │   ├── upload.handler.js
│   │   └── query.handler.js
│   ├── rag/
│   │   ├── chain.js          # LangChain RAG pipeline
│   │   ├── prompts.js        # System prompts
│   │   └── intent.js         # Intent classification
│   ├── mcp/
│   │   ├── server.js         # MCP server (stdio)
│   │   └── tools/
│   │       ├── ask-fire-expert.tool.js
│   │       ├── textify-data.tool.js
│   │       ├── get-embedding.tool.js
│   │       └── health-check.tool.js
│   └── utils/
│       ├── text.util.js
│       ├── file.util.js
│       └── table.util.js
└── db/
    ├── mysql.js              # MySQL connection pool
    └── repo.js               # Document/embedding repository
```

### Data Flow

```
1. Ingest    → File Upload OR API Fetch (Fire Stats)
2. Process   → Files: SHA256 dedup → Docling OCR
             → Stats: Textification (JSON → 자연어 문장)
3. Chunk     → Token-based (500-1200 tokens)
4. Embed     → Batch embedding via bge-m3
5. Store     → MySQL (metadata + vectors in-memory cache)
```

### Database Schema

**MySQL:**
- `documents` - id, content, metadata (JSON)
- `embeddings` - document_id, embedding (JSON array)
- `doc_assets` - sha256, filepath, page, type, image_url, caption
- `doc_tables` - asset_id, n_rows, n_cols, tsv, md, html

---

## Common Commands

### Backend & MCP
```bash
cd server && npm install      # install dependencies
npm start                     # run backend server (node src/index.js)
npm run dev                   # run with --watch for development
npm run mcp                   # run MCP server standalone
```

### Docker (Full Stack)
```bash
docker compose up -d              # start all services
docker compose up -d --build      # rebuild and start
docker compose down               # stop services
docker compose logs -f backend    # view backend logs
```

### Python Utilities
```bash
python server/ocr_once.py         # standalone Docling OCR
```

---

## Environment Configuration

All environment variables are centralized in `server/src/config/env.js`:

| Category | Variables |
|----------|-----------|
| **Server** | `PORT`, `ALLOWED_ORIGINS` |
| **Services** | `EMB_URL`, `LLM_URL`, `LLM_MODEL`, `EMB_MODEL` |
| **MySQL** | `MY_HOST`, `MY_PORT`, `MY_USER`, `MY_PASS`, `MY_DB` |
| **Processing** | `CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_TOKENS`, `MAX_CHUNKS_EMB` |
| **Features** | `FAST_MODE`, `RENDER_PAGES`, `ENABLE_TABLE_INDEX` |
| **RAG Thresholds** | `RETRIEVE_MIN`, `USE_AS_CTX_MIN`, `MIN_TOP3_AVG` |

---

## Key Implementation Details

- **ESM Modules:** All JavaScript files use ES modules (`import`/`export`).
- **Vector Search:** In-memory cache in `db/repo.js` for fast similarity search.
- **Document Parsing:** Python Docling invoked as child process from Node.js.
- **Textification:** Statistics MUST be converted to Korean natural language before embedding.
- **Concurrency:** Custom limiter for batch embedding requests.
- **Language:** Optimized for Korean (ko_KR.UTF-8).
- **MCP Integration:** `@modelcontextprotocol/sdk` for Claude/Gemini connection.

### MCP Tools Available

| Tool | Description |
|------|-------------|
| `ask_fire_expert` | Query the fire investigation expert |
| `textify_data` | Convert JSON to natural language |
| `get_embedding` | Get embedding vector for text |
| `health_check` | Check LLM/Embedding service status |

### Claude Desktop Configuration

Copy `server/claude_desktop_config.example.json` to `~/.config/claude_desktop/config.json` and adjust paths/credentials.
