# RAG (Llama3.1 + BGE-M3)

사용자 문서를 업로드하고 임베딩하여 검색한 뒤, Llama3.1 기반 모델로 응답을 생성하는 **백엔드 RAG 시스템**입니다.

문서 파싱(Node), 임베딩(FastAPI), LLM(FastAPI), 데이터베이스(MySQL/pgvector)로 구성된 **RAG 서버 아키텍처**입니다.

---

## 주요 기능 (Features)

### 1. 문서 기반 RAG 검색

- PDF, TXT, DOCX 자동 파싱 (unstructured + pdf2pic)
- 텍스트 chunking + token-based splitting
- BGE-M3 임베딩 기반 벡터 검색
- Query/Passage Prefix 자동 적용

### 2. Llama3.1 기반 LLM 응답 생성

- 자체 호스팅된 Llama-3.1-8B-Instruct + LoRA
- SSE 기반 스트리밍 응답
- GPU 메모리 최적화(4bit QLoRA 옵션)

### 3. 서비스 별 역할 분리

- Embedding 서버 (BGE-M3)
- LLM 서버 (Llama3.1)
- Ingestion 서버 (Node)

### 4. 자산(이미지/표) RAG 지원

- pdf2pic로 페이지 → PNG 변환
- 표 → HTML → row-level embedding 저장
- 이미지·표 caption 임베딩 검색 가능

---

## 프로젝트 구조

/

├── llm_service/

│   ├── main.py

│   ├── models/

│   └── .env

│

├── emb_service/

│   ├── main.py

│   └── .env

│

├── server/

│   ├── index.js

│   ├── chunk.js

│   ├── ocr_once.py

│   └── .env

│

├── db/

│   ├── mysql.js

│   └── repo.js

│

└── README.md

---

## 설치 (Installation)

### 1. Embedding Server (FastAPI)

cd emb_service

pip install -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8001

### 2. LLM Server (FastAPI)

cd llm_service

pip install -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8002

### 3. Document Server (Node)

cd server

npm install

node index.js

---

## RAG 전체 처리 흐름

1. 문서 업로드
2. Node ingestion 서버에서 pdf2pic / pdf-parse / unstructured로 텍스트·이미지·표 추출
3. chunk.js 로 텍스트 chunking
4. BGE-M3 Embedding 서버로 임베딩 요청
5. DB에 문서·chunk·자산 저장
6. 사용자가 질문 입력
7. query 임베딩 생성
8. DB에서 Top-K 문서 검색
9. LLM 서버로 컨텍스트 + query 전달
10. Llama3.1이 최종 답변 생성

---

## REST API 엔드포인트 (Document Server)

### **POST /upload**

문서를 업로드하면 다음 작업을 순차적으로 수행:

- 파일 저장
- SHA256 중복 검사
- pdf2pic로 페이지 이미지를 생성
- pdf-parse / unstructured로 텍스트·이미지·표를 추출
- chunk.js로 텍스트를 chunk 단위로 분리
- Embedding 서버(BGE-M3)에 임베딩 요청
- documents, embeddings, doc_assets, doc_tables를 MySQL에 저장
    
    업로드된 문서의 DB id 및 처리된 자산 개수를 반환
    

### **POST /embed**

텍스트를 Embedding 서버(BGE-M3)에 전달하여 1024차원 임베딩을 생성

단일 텍스트 또는 배열 모두 지원하며, query/passsage prefix 적용 및 L2 정규화 옵션을 포함

### **POST /query**

사용자의 질문을 받아 다음을 수행

- BGE-M3로 query 임베딩 생성
- embeddings 테이블과 코사인 유사도(dot)로 검색
- threshold 이상 상위 K개의 문맥을 선택
- LLM 서버(Llama3.1)에 컨텍스트 + 사용자 질문 전달
- 최종 생성된 답변과 검색된 문맥을 반환

### **GET /health**

서버 상태 체크용 엔드포인트

Embedding 서버 URL, LLM 서버 URL, storage 경로 등 현재 설정 정보를 반환

---

## 데이터베이스 구조 (요약)

### documents
- 문서 본문(content)과 metadata(JSON)를 저장
- id (PK)
- content (LONGTEXT)
- metadata (JSON)

### embeddings
- 문서별 임베딩 벡터 저장 (L2 정규화된 1024차원)
- id (PK)
- document_id (FK → documents.id)
- embedding (JSON 형태의 float 배열)

### doc_assets
- 문서에서 추출된 이미지/표 자산 관련 정보 저장
- id (PK)
- sha256 (중복 방지용 파일 해시)
- filepath (서버 로컬 저장 경로)
- page (해당 자산이 속한 페이지)
- type (image 또는 table)
- image_url (이미지 파일 접근 경로)
- caption_text (caption 원문)
- caption_emb (caption 임베딩, JSON float 배열)
- meta (추가 정보 JSON)

### doc_tables
- 표(table) 구조 저장 (tsv/md/html 형식)
- id (PK)
- asset_id (FK → doc_assets.id)
- n_rows
- n_cols
- tsv
- md
- html