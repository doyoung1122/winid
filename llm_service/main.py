import os
import threading
import time
import torch
from typing import List, Literal, Optional, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)
import uvicorn

# ===================================
# 성능 플래그
# ===================================
torch.backends.cuda.matmul.allow_tf32 = True
if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        # 단일 GPU 0번 고정
        torch.cuda.set_device(0)
    except Exception:
        pass

# ===================================
# 환경변수
# ===================================
MODEL_DIR = os.getenv("LLM_MODEL_DIR", r"D:\llama_project\models\llama31-8b-instruct")
PORT = int(os.getenv("LLM_PORT", "8002"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = os.getenv("USE_BF16", "1") == "1"
DTYPE = torch.bfloat16 if (DEVICE == "cuda" and USE_BF16) else (
    torch.float16 if DEVICE == "cuda" else torch.float32
)

# 속도/메모리 균형: 기본 컨텍스트 길이 2048
MAX_INPUT_LEN = int(os.getenv("MAX_INPUT_LEN", "2048"))
MAX_GEN_TIME = float(os.getenv("MAX_GEN_TIME_SEC", "30"))
REP_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.05"))
USE_FA2 = os.getenv("USE_FLASH_ATTENTION2", "1") == "1"

# ===================================
# Tokenizer & EOS
# ===================================
tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
eos_ids = {tok.eos_token_id}
if isinstance(eot_id, int) and eot_id >= 0:
    eos_ids.add(eot_id)
EOS = list(eos_ids)

# ===================================
# 4bit 모델 로드 (속도/메모리 최적화)
# ===================================
compute_dtype = (
    torch.bfloat16
    if (torch.cuda.is_available() and USE_BF16)
    else (torch.float16 if torch.cuda.is_available() else torch.float32)
)

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

try:
    # 단일 GPU(0번)에 고정해서 불필요한 device_map 계산 줄이기
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        device_map={"": 0} if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
        quantization_config=bnb_cfg,
        attn_implementation=("flash_attention_2" if USE_FA2 else "eager"),
    )
except Exception:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        device_map={"": 0} if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
        quantization_config=bnb_cfg,
    )
    try:
        model.config.attn_implementation = "eager"
    except Exception:
        pass

model.eval()
try:
    model.generation_config.use_cache = True
except Exception:
    pass

print("[llm] loaded")
print("[llm] device:", DEVICE)
print("[llm] dtype:", DTYPE)
print("[llm] attn_impl:", getattr(model.config, "attn_implementation", "unknown"))

# ===================================
# 요청 모델
# ===================================
class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatReq(BaseModel):
    context: str = ""          # 문서 발췌 컨텍스트
    question: str              # 현재 사용자 질문
    history: Optional[List[Turn]] = None
    system: str = ""           # 상위 정책/규칙 (query 서버에서 주입)
    # 👉 기본 출력 길이 256 (속도/부하 대비 현실적인 값)
    max_new_tokens: int = 256
    # 속도 우선 모드에서는 기본적으로 greedy 사용
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    # 필요하면 per-request로 샘플링 켜고 싶을 때 사용할 플래그
    do_sample: Optional[bool] = None

app = FastAPI(title="QLoRA (Base Only) Chat Service - Speed Optimized", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================================
# Helper: Llama 3.1 호환 프롬프트 빌더
#  - system -> (optional) context_system -> trimmed history -> user
#  - history는 토큰 예산으로 최신부터 포함
# ===================================
def pack(role: str, content: str) -> str:
    return f"<|start_header_id|>{role}<|end_header_id|>\n{content}\n<|eot_id|>"

def build_prompt(
    system: str,
    context_sys: str,                # 문서 컨텍스트를 system으로 분리 주입
    history: List[Tuple[str, str]],
    user_msg: str,
    max_input_len: int
) -> str:
    parts = ["<|begin_of_text|>"]
    if system:
        parts.append(pack("system", system))
    if context_sys:
        parts.append(pack("system", context_sys))

    tail = pack("user", user_msg) + "<|start_header_id|>assistant<|end_header_id|>\n"
    static = parts[0] + "".join(parts[1:]) + tail
    static_len = len(tok(static, return_tensors="pt")["input_ids"][0])
    budget = max_input_len - static_len
    kept: List[str] = []

    # 최신 → 오래된 순으로 거꾸로 채우기
    for role, content in reversed(history):
        if role not in ("user", "assistant") or not content:
            continue
        seg = pack(role, content)
        seg_ids = len(tok(seg, return_tensors="pt")["input_ids"][0])
        if seg_ids <= budget:
            kept.append(seg)
            budget -= seg_ids
        else:
            break

    # 다시 시간순으로 붙이기
    parts.extend(reversed(kept))
    parts.append(pack("user", user_msg))
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
    return "".join(parts)

# ===================================
# Warm-up (첫 요청 속도 감소 방지)
# ===================================
@app.on_event("startup")
def _warmup():
    try:
        prompt = build_prompt("", "", [], "짧게: 안녕?", MAX_INPUT_LEN)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN)
        inputs = {k: v.to(model.device, non_blocking=True) for k, v in inputs.items()}
        with torch.inference_mode():
            _ = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                eos_token_id=EOS,
                use_cache=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print("[llm] warm-up done ✅")
    except Exception as e:
        print("[llm] warm-up failed:", e)

# ===================================
# Health Check
# ===================================
@app.get("/health")
def health():
    return {
        "ok": True,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "attn": getattr(model.config, "attn_implementation", "unknown"),
        "quantized": True,
        "model_dir": MODEL_DIR,
        "vocab_size": getattr(model.config, "vocab_size", None),
        "max_position_embeddings": getattr(model.config, "max_position_embeddings", None),
        "max_input_len": MAX_INPUT_LEN,
    }

# ===================================
# Chat Endpoint (속도 우선 프로파일)
# ===================================
@app.post("/chat")
def chat(req: ChatReq):
    # history 수집
    hist_pairs: List[Tuple[str, str]] = []
    if req.history:
        for t in req.history:
            if t.role in ("user", "assistant") and t.content:
                hist_pairs.append((t.role, t.content))

    # 문서 컨텍스트는 별도 system 메시지로
    context_system = (
        f"다음은 참고 문서 발췌입니다. 필요한 경우에만 근거로 사용하고, "
        f"지시된 형식 이외의 머리말/꼬리표를 출력하지 마십시오.\n\n{req.context}"
        if req.context else ""
    )

    user_msg = req.question
    prompt = build_prompt(req.system, context_system, hist_pairs, user_msg, MAX_INPUT_LEN)

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN)
    inputs = {k: v.to(model.device, non_blocking=True) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    # 출력 길이 제한 (256 기본, 상한 512)
    asked = req.max_new_tokens or 256
    max_new = max(64, min(int(asked), 512))

    # 기본은 greedy (최고 속도/안정성)
    if req.do_sample is None:
        do_sample = False
    else:
        do_sample = bool(req.do_sample)

    # 샘플링 켤 때만 temperature/top_p 사용
    if do_sample:
        temp = 0.7 if req.temperature is None else max(0.1, min(req.temperature, 1.5))
        topp = 0.9 if req.top_p is None else max(0.5, min(req.top_p, 0.98))
    else:
        temp = None
        topp = None

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new,
        pad_token_id=tok.eos_token_id,
        eos_token_id=EOS,
        use_cache=True,
        do_sample=do_sample,
        repetition_penalty=REP_PENALTY,
        max_time=MAX_GEN_TIME,
    )

    if do_sample:
        gen_kwargs["temperature"] = temp
        gen_kwargs["top_p"] = topp

    def _run():
        with torch.inference_mode():
            model.generate(**gen_kwargs)

    th = threading.Thread(target=_run, daemon=True)
    th.start()

    def event_stream():
        last_ping = time.time()
        for piece in streamer:
            if piece:
                yield f"data: {piece}\n\n"
            now = time.time()
            if now - last_ping > 10.0:
                yield "event: ping\ndata: 1\n\n"
                last_ping = now
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)