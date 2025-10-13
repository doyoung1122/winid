# Rag를 이용한 Chatbot 만들기

![image.png](attachment:09cdaaaa-4662-4665-a6d7-1e1ff0160d99:image.png)

## 그림 속 단계 설명

1. **Reference documents (참고 문서)**
    - 회사 내부 문서, 공고, 정책 자료 등 → “지식 창고” 역할.
2. **LLM embedding → Vector database**
    - 문서들을 잘게 나눠서(청크) 임베딩 모델로 숫자 벡터화.
    - 이렇게 만든 임베딩을 **벡터 데이터베이스**에 저장해 둠.
3. **User query (사용자 질문)**
    - 사람이 질문을 하면, 그 질문도 임베딩 벡터로 변환됨.
4. **Find relevant documents**
    - 쿼리 벡터와 DB 속 문서 벡터들을 비교 → **가까운(유사한) 문서 청크**를 찾음.
5. **Retrieve context chunks**
    - 관련 있는 문서 조각들을 꺼내옴.
6. **Augmented query (확장된 프롬프트)**
    - “원래 질문 + 찾아온 문서 청크들”을 합쳐서 LLM에 전달.
    - 예시:
        
        ```
        질문: 인턴 마감일 알려줘
        참고자료:
        <chunk1: HR 공고>
        <chunk2: 접수 안내>
        
        ```
        
7. **LLM generation → Response**
    - LLM은 주어진 문맥 안에서만 답변을 생성.
    - 따라서 환각이 줄고, **출처 있는 답변**을 할 수 있음.

---

## 쉽게 비유

- 📚 **Vector DB** = 도서관 카드목록
- ❓ **User query** = “이 주제 책 어디 있어요?”
- 🔍 **Retriever** = 사서가 비슷한 책 찾아줌
- ✏️ **LLM** = 책 내용을 읽고 요약해서 답변 작성

# 📌 기존 LLM(파인튜닝만 한 챗봇)의 대표 문제점

1. **환각(Hallucination)**
    - 근거 없는 답변을 지어냄 → 사실 검증 어려움.
2. **최신성 부족**
    - 모델 학습 시점 이후의 데이터는 반영 불가.
    - 예: 2025년 9월의 공고는 2023년에 학습된 모델은 모름.
3. **출처 부재**
    - “왜 그렇게 답하는가?”에 대한 근거 제공 불가.
    - 기업·공공기관에서는 신뢰성 떨어짐.
4. **특정 도메인 지식 부족**
    - 법률, 재무, 사내 규정 등 **특화 문서**에 대한 답변 정확도가 낮음.

---

# 📌 RAG(Retrieval-Augmented Generation)로 해결하는 방식

1. **환각 억제**
    - 답변을 “검색된 실제 문서 청크”에 근거해 작성 → 사실 기반 강화.
2. **최신성 보장**
    - 새 문서가 나오면 **DB에만 업데이트** → 모델 재학습 없이 즉시 활용.
3. **출처 제시**
    - 검색된 문서의 제목·URL·날짜를 답변에 포함 가능 → 신뢰성 향상.
4. **도메인 최적화**
    - 사내 규정/정책 문서들을 인덱싱 → 일반 LLM도 **전문가처럼 답변** 가능.

# TodoList

- [x]  **환경 준비**
    - [x]  Python 가상환경 구성 (conda)
    - [x]  Hugging Face `transformers`, `torch`, `accelerate`, `faiss` 설치
    - [x]  GPU 세팅 (CUDA or ROCm, 또는 Llama.cpp로 CPU inference도 가능)
- [x]  **모델 다운로드 & 로딩**
    - [x]  Hugging Face에서 `meta-llama/Llama-3.1-8B-Instruct` 같은 모델 다운로드
    - [x]  `transformers.AutoModelForCausalLM` + `AutoTokenizer` 로 로드
    - [x]  로컬에서 프롬프트 테스트 (간단한 대화)
- [x]  **벡터 저장소 준비**
    - [x]  Supabase Postgres + pgvector 확장
    - [x]  문서 테이블 생성 (id, text, embedding, metadata 등)
    - [x]  Supabase JS/Python SDK 연동 확인
- [x]  **임베딩 생성**
    - [x]  BGE-M3 Embedding Service 임베딩 API 사용
    - [x]  문서 chunking 후 임베딩 생성 → Supabase DB에 insert
- [ ]  **검색 + RAG**
    - [ ]  사용자가 질문 입력 → 임베딩 생성
    - [ ]  Supabase pgvector similarity search → 관련 chunk top-k 반환
    - [ ]  context + 사용자 질문 → LLaMA 3.1 프롬프트 구성
    - [ ]  LLaMA 3.1에 넣고 답변 생성
- [ ]  **React 프론트엔드**
    - [ ]  입력창, 출력창, 대화 UI
    - [ ]  Supabase SDK로 로그인/회원별 데이터 관리
    - [ ]  업로드한 문서 리스트 보여주기
    - [ ]  답변 + 출처 chunk 하이라이트
- [ ]  **추가 기능**
    - [ ]  캐싱 (자주 쓰는 쿼리 빠르게)
    - [ ]  서버 API 구축 (FastAPI/Express) → React ↔ Backend ↔ LLaMA
    - [ ]  모델 추론 최적화 (quantization, LoRA fine-tuning)
