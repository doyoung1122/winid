import os
import torch
import threading
import concurrent.futures
from typing import Optional, List, Tuple
from io import BytesIO
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import requests
import uvicorn

from transformers import AutoProcessor, MllamaForConditionalGeneration

# =========================
# 성능/환경
# =========================
torch.backends.cuda.matmul.allow_tf32 = True
if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

VISION_MODEL_DIR   = os.getenv("VISION_MODEL_DIR", r"D:\llama_project\models\Llama-3.2-11B-Vision")
PORT               = int(os.getenv("VISION_PORT", "8003"))
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
# bf16 지원 여부에 따라 dtype 자동 선택
if DEVICE == "cuda":
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    DTYPE = torch.float32

# 안전/운영 관련 설정
MAX_IMAGE_BYTES     = int(os.getenv("MAX_IMAGE_BYTES", "10485760"))  # 10MB
MAX_CAPTION_SIDE    = int(os.getenv("MAX_CAPTION_SIDE", "2048"))     # 최대 변 길이
VISION_MAX_CONC     = int(os.getenv("VISION_MAX_CONCURRENCY", "1"))  # generate 동시성
ALLOWED_SCHEMES     = {"http", "https"}

print(f"[vision] loading model: {VISION_MODEL_DIR}  device: {DEVICE} ({DTYPE})  conc:{VISION_MAX_CONC}")

# =========================
# 모델 / 프로세서 로드
# =========================
processor = AutoProcessor.from_pretrained(
    VISION_MODEL_DIR,
    local_files_only=True,
)

model = MllamaForConditionalGeneration.from_pretrained(
    VISION_MODEL_DIR,
    local_files_only=True,
    device_map="auto" if DEVICE == "cuda" else None,
    torch_dtype=DTYPE,
)
model.eval()

# generate 동시성 제한 (GPU OOM 방지)
_GEN_SEM = threading.Semaphore(max(1, VISION_MAX_CONC))

# =========================
# 유틸
# =========================
DEFAULT_PROMPT = (
    "이미지의 핵심 내용을 자세히 설명해줘. "
    "표, 그래프, 축, 범례, 단위, 숫자 정보를 가능한 한 텍스트로 풀어써."
)

UA_HEADERS = {
    "User-Agent": "llama-3.2-vision/1.1 (+http://localhost)",
    "Accept": "image/*,application/octet-stream;q=0.8,*/*;q=0.5",
}

def _fetch_image(url: str) -> Image.Image:
    """안전한 이미지 다운로드 (스킴/크기 제한 + 스트리밍)"""
    u = urlparse(url)
    if u.scheme not in ALLOWED_SCHEMES:
        raise ValueError("unsupported URL scheme")

    with requests.get(url, headers=UA_HEADERS, timeout=25, stream=True) as r:
        r.raise_for_status()
        total = 0
        chunks = []
        for chunk in r.iter_content(1024 * 32):
            total += len(chunk)
            if total > MAX_IMAGE_BYTES:
                raise ValueError("image too large")
            chunks.append(chunk)
        data = b"".join(chunks)

    img = Image.open(BytesIO(data)).convert("RGB")
    img.thumbnail((MAX_CAPTION_SIDE, MAX_CAPTION_SIDE))
    return img

def _pad_token_id() -> int:
    return (
        getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
        or getattr(model.config, "eos_token_id", None)
        or 0
    )

def _prepare_inputs(img: Image.Image, user_prompt: str):
    """chat template이 있으면 사용, 없으면 text+image 통합 입력"""
    use_chat = hasattr(processor, "apply_chat_template")

    if use_chat:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        text_ids = processor.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        inputs = processor(
            images=img,
            text=None,
            return_tensors="pt"
        )
        inputs["input_ids"] = text_ids
    else:
        inputs = processor(
            text=user_prompt,
            images=img,
            return_tensors="pt"
        )
    return {k: v.to(model.device) for k, v in inputs.items()}

def caption_image(img: Image.Image, prompt: Optional[str] = None, max_new_tokens: int = 320) -> Tuple[str, int]:
    """
    이미지 캡션 생성. (text, generated_token_count) 반환
    """
    user_prompt = (prompt or DEFAULT_PROMPT).strip()
    img = img.convert("RGB")
    img.thumbnail((MAX_CAPTION_SIDE, MAX_CAPTION_SIDE))

    inputs = _prepare_inputs(img, user_prompt)
    pad_id = _pad_token_id()

    # 생성
    with _GEN_SEM:  # 동시성 제한
        with torch.inference_mode():
            if DEVICE == "cuda":
                with torch.cuda.amp.autocast(dtype=DTYPE):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max(32, min(max_new_tokens or 320, 512)),
                        do_sample=False,
                        pad_token_id=pad_id,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max(32, min(max_new_tokens or 320, 512)),
                    do_sample=False,
                    pad_token_id=pad_id,
                )

    # 프롬프트 부분 제거 후 디코딩
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and "input_ids" in inputs:
        prefix_len = inputs["input_ids"].shape[-1]
        gen_only = outputs[0, prefix_len:]
        token_count = int(gen_only.numel())
        text = tok.decode(gen_only, skip_special_tokens=True).strip()
    else:
        # tokenizer가 없거나 특수 케이스: 전체 디코딩
        token_count = int(outputs[0].numel())
        text = tok.decode(outputs[0], skip_special_tokens=True).strip() if tok else str(outputs[0])

    return text, token_count

# =========================
# API 스키마
# =========================
class CaptionReq(BaseModel):
    image_url: str
    prompt: Optional[str] = None
    max_new_tokens: Optional[int] = 320

class BatchCaptionReq(BaseModel):
    image_urls: List[str]
    prompt: Optional[str] = None
    max_new_tokens: Optional[int] = 320

# =========================
# FastAPI
# =========================
app = FastAPI(title="Llama-3.2-Vision Caption Service", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _warmup():
    try:
        img = Image.new("RGB", (2, 2), color=(255, 255, 255))
        _ = caption_image(img, "짧게: 흰색 더미 이미지", max_new_tokens=64)
        print("[vision] warm-up done")
    except Exception as e:
        print("[vision] warm-up failed:", e)

@app.get("/health")
def health():
    return {
        "ok": True,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "model_dir": VISION_MODEL_DIR,
        "model_class": str(type(model)),
        "has_chat_template": bool(hasattr(processor, "apply_chat_template")),
        "max_image_bytes": MAX_IMAGE_BYTES,
        "max_caption_side": MAX_CAPTION_SIDE,
        "max_concurrency": VISION_MAX_CONC,
    }

@app.post("/vision/caption")
def vision_caption(req: CaptionReq):
    try:
        img = _fetch_image(req.image_url)
        text, tokens = caption_image(img, req.prompt, max_new_tokens=req.max_new_tokens or 320)
        return {"ok": True, "text": text, "tokens": tokens}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _download(url: str) -> Tuple[str, Optional[Image.Image], Optional[str]]:
    try:
        return (url, _fetch_image(url), None)
    except Exception as e:
        return (url, None, str(e))

@app.post("/vision/caption/batch")
def vision_caption_batch(req: BatchCaptionReq):
    urls = req.image_urls or []
    results = []
    max_new = max(32, min(req.max_new_tokens or 320, 512))

    # 1) 병렬 다운로드
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(_download, u) for u in urls]
        downloads = [f.result() for f in futures]

    # 2) 순차 generate (GPU 안전)
    for i, (url, img, err) in enumerate(downloads, 1):
        if err:
            results.append({"ok": False, "index": i, "image_url": url, "error": err})
            continue
        try:
            text, tokens = caption_image(img, req.prompt, max_new_tokens=max_new)
            results.append({"ok": True, "index": i, "image_url": url, "text": text, "tokens": tokens})
        except Exception as e:
            results.append({"ok": False, "index": i, "image_url": url, "error": str(e)})

    return {"ok": True, "results": results}

@app.post("/vision/caption_bytes")
async def vision_caption_bytes(request: Request, prompt: Optional[str] = None, max_new_tokens: int = 320):
    """
    PowerShell 예시:
      Invoke-RestMethod `
        -Uri "http://127.0.0.1:8003/vision/caption_bytes?max_new_tokens=320" `
        -Method POST `
        -InFile "D:\\llama_project\\image.png" `
        -ContentType "application/octet-stream"
    """
    try:
        raw = await request.body()
        if len(raw) > MAX_IMAGE_BYTES:
            return {"ok": False, "error": "image too large"}
        img = Image.open(BytesIO(raw)).convert("RGB")
        img.thumbnail((MAX_CAPTION_SIDE, MAX_CAPTION_SIDE))
        text, tokens = caption_image(img, prompt or "", max_new_tokens=max_new_tokens)
        return {"ok": True, "text": text, "tokens": tokens}
    except Exception as e:
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run("vision:app", host="0.0.0.0", port=PORT, reload=False)