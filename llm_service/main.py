import os
import threading
import torch
from typing import List, Literal, Optional, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel
import uvicorn

# 성능 플래그
torch.backends.cuda.matmul.allow_tf32 = True
if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

MODEL_DIR   = os.getenv("LLM_MODEL_DIR",  r"D:\llama_project\models\llama31-8b-instruct")
ADAPTER_DIR = os.getenv("LLM_ADAPTER_DIR",r"D:\llama_project\outputs\llama31-8b-sft\adapter")
PORT        = int(os.getenv("LLM_PORT", "8002"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32

def build_prompt(system, history, user_msg):
    parts = ["<|begin_of_text|>"]
    if system:
        parts.append(f"<|start_header_id|>system<|end_header_id|>\n{system}\n<|eot_id|>")
    for role, content in history:
        parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n{content}\n<|eot_id|>")
    parts.append(f"<|start_header_id|>user<|end_header_id|>\n{user_msg}\n<|eot_id|>")
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
    return "".join(parts)

print(f"[llm] loading base: {MODEL_DIR}  adapter: {ADAPTER_DIR}  device: {DEVICE} ({DTYPE})")
tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

ATTN_IMPL = "flash_attention_2" if os.getenv("USE_FLASH_ATTENTION2", "1") == "1" else "eager"

base = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    device_map="auto",
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
)

MERGE_LORA = os.getenv("MERGE_LORA", "1") == "1"
if MERGE_LORA:
    lora = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model = lora.merge_and_unload()
else:
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()
try:
    model.generation_config.use_cache = True
except Exception:
    pass

# ===== API =====
from typing import Literal, Optional, List, Tuple
class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatReq(BaseModel):
    context: str = ""
    question: str
    history: Optional[List[Turn]] = None
    system: str = ""
    max_new_tokens: int = 128
    temperature: Optional[float] = None   
    top_p: Optional[float] = None         

app = FastAPI(title="LoRA LLaMA 3.1-8B Chat Service", version="1.2.0")
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"]
)

@app.on_event("startup")
def _warmup():
    # ⚡ 워밍업: 첫 요청 랙 제거
    try:
        prompt = build_prompt("", [], "짧게: 안녕?")
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        _ = model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        print("[llm] warm-up done")
    except Exception as e:
        print("[llm] warm-up failed:", e)

@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "dtype": str(DTYPE), "attn": ATTN_IMPL, "merge_lora": MERGE_LORA}

@app.post("/chat")
def chat(req: ChatReq):
    hist_pairs: List[Tuple[str, str]] = []
    if req.history:
        for t in req.history:
            if t.role in ("user", "assistant") and t.content:
                hist_pairs.append((t.role, t.content))

    user_msg = req.question if not req.context else f"CONTEXT:\n{req.context}\n\nQUESTION:\n{req.question}"
    prompt = build_prompt(req.system, hist_pairs, user_msg)

    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device, non_blocking=True) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    max_new = max(32, min(req.max_new_tokens or 128, 128))

    use_sampling = (req.temperature is not None and req.temperature > 0) or \
               (req.top_p is not None and req.top_p > 0 and req.top_p < 1.0)

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new,
        pad_token_id=tok.eos_token_id,
        use_cache=True,
    )

    if use_sampling:
        temp = 0.7 if req.temperature is None else max(0.1, min(req.temperature, 0.95))
        topp = 0.9 if req.top_p is None else max(0.7, min(req.top_p, 0.98))
        gen_kwargs.update(
            do_sample=True,
            temperature=temp,
            top_p=topp,
        )
    else:
        gen_kwargs.update(do_sample=False)

    th = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
    th.start()

    def event_stream():
        for piece in streamer:
            yield f"data: {piece}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)