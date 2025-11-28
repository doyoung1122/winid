# RAG (Llama3.1 + BGE-M3)

사용자 문서를 업로드하고 임베딩하여 검색한 뒤, Llama3.1 기반 모델로 응답을 생성하는 **RAG 시스템**입니다.

문서 파싱(Node), 임베딩(FastAPI), LLM(FastAPI), 데이터베이스로 구성된 **RAG 아키텍처**입니다.

---

## 주요 기능 (Features)

### 1. 문서 기반 RAG 검색

- PDF · TXT · DOCX 자동 파싱 (unstructured + pdf2pic)
- 텍스트 chunking (token-based splitting)
- BGE-M3 임베딩 기반 벡터 검색
- Query / Passage Prefix 자동 적용

### 2. Llama3.1 기반 LLM 응답 생성

- 자체 호스팅된 Llama-3.1-8B-Instruct + LoRA
- SSE 기반 스트리밍 응답
- GPU 메모리 최적화(4bit QLoRA 옵션 지원)

### 3. 서비스 별 역할 분리

- **Embedding 서버 (BGE-M3)**
- **LLM 서버 (Llama3.1)**
- **Document Ingestion 서버 (Node)**

### 4. 자산(이미지/표) RAG 지원

- pdf2pic로 페이지 → PNG 변환
- 표 → HTML → row-level embedding 저장
- 이미지·표 caption 임베딩 검색 가능

---

# 설치 (Installation)

---

# 📥 Llama 3.1 모델 다운로드 안내 (LLM Server)

LLM 서버(`llm_service`)는 Meta에서 공개한 **Llama 3.1 Instruct 모델**을 사용합니다.

이 모델은 HuggingFace에서 제공되지만, **Meta 모델은 접근 권한 승인이 필요합니다.**

---

### 1. Meta 모델 접근 권한 승인

아래 모델 페이지에서 **Access request** 버튼을 눌러 승인을 받아야 합니다.

https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

승인 완료 후 다운로드가 가능합니다.

---

### 2. HuggingFace 로그인

PowerShell에서 **명령어**로 인증합니다:

```powershell
hf auth login

```

HuggingFace 토큰 발급 위치:

https://huggingface.co/settings/tokens

발급된 `hf_...` 토큰을 붙여넣고 Enter 합니다.

---

### 3. Llama 3.1 모델 다운로드 (PowerShell)

PowerShell은 줄바꿈이 안 되므로 **한 줄**로 다운로드합니다.

```powershell
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ./models/llama31 --include "*.safetensors" "*.json" "*.model"

```

다운로드된 모델은 `./models/llama31` 경로에 저장됩니다.

LLM 서버는 이 로컬 폴더를 그대로 로드하여 실행합니다.

---

# Embedding Server 설치 (FastAPI)

```bash
cd emb_service
pip install -r requirements.txt

```

---

# LLM Server 설치 (FastAPI)

```bash
cd llm_service
pip install -r requirements.txt

```

---

# Document Server 설치 (Node)

```bash
cd server
npm install
node index.js

```

---

# RAG 전체 처리 흐름

1. 문서를 업로드
2. Ingestion 서버(Node)가 pdf2pic / pdf-parse / unstructured로 텍스트·이미지·표 추출
3. chunk.js로 텍스트를 chunk 단위로 분리
4. BGE-M3 Embedding 서버로 임베딩 요청
5. DB에 `documents`, `embeddings`, `doc_assets`, `doc_tables` 저장
6. 사용자가 질문을 입력
7. query 임베딩 생성
8. DB에서 코사인 유사도로 Top-K 문서 검색
9. LLM 서버로 컨텍스트 + query 전달
10. Llama3.1이 최종 답변 생성

---

# REST API 엔드포인트 (Document Server)

### POST /upload

문서를 업로드하여 다음을 수행:

- 파일 저장
- SHA256 중복 검사
- pdf2pic로 페이지 이미지 생성
- pdf-parse / unstructured로 텍스트·이미지·표 추출
- chunk.js로 텍스트 split
- Embedding 서버(BGE-M3)로 임베딩 요청
- DB 저장
    
    → 처리된 문서 ID와 자산 개수를 반환
    

---

### POST /embed

텍스트를 BGE-M3 Embedding 서버로 전달하여 **1024차원 임베딩**을 생성

단일 텍스트·복수 텍스트 모두 지원하며 L2 정규화 옵션 포함

---

### POST /query

1. Query 텍스트 임베딩 생성
2. DB embeddings와 코사인 유사도(dot) 계산
3. threshold 이상 상위 K개 검색
4. LLM 서버에 전달하여 답변 생성 후 반환

---

### GET /health

서버 상태, Embedding 서버 URL, LLM 서버 URL, storage 경로 등을 반환

---

# 데이터베이스 구조 (요약)

### documents

- id (PK)
- content (LONGTEXT)
- metadata (JSON)

### embeddings

- id (PK)
- document_id (FK → documents.id)
- embedding (JSON, L2 정규화된 1024차원)
    
    ※ BGE-M3는 1024차원이며, 모델을 변경하면 차원도 함께 변경해야 함
    

### doc_assets

- id (PK)
- sha256
- filepath
- page
- type (image/table)
- image_url
- caption_text
- caption_emb (JSON)
- meta (JSON)

### doc_tables

- id (PK)
- asset_id (FK → doc_assets.id)
- n_rows
- n_cols
- tsv
- md
- html