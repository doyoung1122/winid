import os
from typing import List, Optional, Union, Literal
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import uvicorn

MODEL_ID = os.getenv("MODEL_ID", "BAAI/bge-m3")   # bge-m3 (1024-dim)
DEVICE   = os.getenv("DEVICE",   "cuda" if torch.cuda.is_available() else "cpu")
DTYPE    = os.getenv("DTYPE",    "float16" if torch.cuda.is_available() else "float32")
PORT     = int(os.getenv("PORT", "8001"))

QUERY_PREFIX   = os.getenv("QUERY_PREFIX",   "Represent this sentence for searching relevant passages: ")
PASSAGE_PREFIX = os.getenv("PASSAGE_PREFIX", "Represent this document for retrieval: ")

# -----------------------------
# 모델 로드
# -----------------------------
print(f"[emb] loading model: {MODEL_ID} on {DEVICE} ({DTYPE})")
dtype = torch.float16 if DTYPE == "float16" and DEVICE.startswith("cuda") else torch.float32
model = SentenceTransformer(MODEL_ID, device=DEVICE)
model.max_seq_length = 512  # 필요 시 조정

# -----------------------------
# 유틸
# -----------------------------
def l2_normalize(v: np.ndarray) -> np.ndarray:
    # v: (N, D) 또는 (D,)
    arr = v if v.ndim == 2 else v[None, :]
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    out = arr / norms
    return out if v.ndim == 2 else out[0]

def apply_prefix(texts: List[str], how: Literal["auto","query","passage"]) -> List[str]:
    if how == "query":
        return [QUERY_PREFIX + t for t in texts]
    if how == "passage":
        return [PASSAGE_PREFIX + t for t in texts]
    # auto: 짧으면 query, 길면 passage 같이 간단 heuristic
    out = []
    for t in texts:
        tok_est = len(t.split())
        out.append((QUERY_PREFIX + t) if tok_est <= 32 else (PASSAGE_PREFIX + t))
    return out

# -----------------------------
# API 스키마
# -----------------------------
class EmbedRequest(BaseModel):
    # 단일 혹은 배치 중 하나 사용
    input: Optional[Union[str, List[str]]] = None
    # 'query' | 'passage' | 'auto'
    mode: Literal["auto","query","passage"] = "auto"
    # 결과 벡터 정규화
    normalize: bool = True
    # 배치 인퍼런스 크기
    batch_size: int = 32

class EmbedResponse(BaseModel):
    # 단일이면 embedding, 배치면 embeddings
    embedding: Optional[List[float]] = None
    embeddings: Optional[List[List[float]]] = None
    dim: int


app = FastAPI(title="BGE-M3 Embedding Service", version="1.0.0")

# CORS (필요 시 도메인 제한)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_ID, "device": DEVICE}

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    # 입력 정규화
    if req.input is None:
        return {"dim": 0, "embedding": None, "embeddings": None}
    if isinstance(req.input, str):
        texts = [req.input]
        single = True
    else:
        texts = list(req.input)
        single = False
    texts = [t.strip() for t in texts if t and t.strip()]
    if not texts:
        return {"dim": 0, "embedding": None, "embeddings": None}

    # prefix 적용
    texts_prefixed = apply_prefix(texts, req.mode)

    # 배치 인퍼런스
    bs = max(1, min(1024, req.batch_size))
    vecs = []
    for i in range(0, len(texts_prefixed), bs):
        batch = texts_prefixed[i:i+bs]
        with torch.inference_mode():
            embs = model.encode(
                batch,
                batch_size=len(batch),
                convert_to_numpy=True,
                normalize_embeddings=False,  # 수동 정규화
                show_progress_bar=False
            )
        vecs.append(embs)
    vecs = np.vstack(vecs)  # (N, D)
    if req.normalize:
        vecs = l2_normalize(vecs)

    dim = int(vecs.shape[1])
    if single:
        return EmbedResponse(embedding=vecs[0].astype(float).tolist(), dim=dim)
    else:
        return EmbedResponse(embeddings=vecs.astype(float).tolist(), dim=dim)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)