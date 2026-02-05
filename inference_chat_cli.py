import sys, time, threading
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel

# === 경로 설정 ===
MODEL_DIR   = r"D:\llama_project\models\llama31-8b-instruct"
ADAPTER_DIR = r"D:\llama_project\outputs\llama31-8b-sft\adapter"

# === 대화 정책(System 프롬프트) ===
SYSTEM_MSG = (
    "Answer only in the user's language. "
    "Default to polite Korean honorifics unless a different tone is requested. "
    "If uncertain, say '모르겠습니다.'. Keep answers concise."
)

# === Llama 3.x 권장 대화 포맷 생성 ===
def build_prompt(system: str, history: List[Tuple[str, str]], user_msg: str) -> str:
    """history: [(role, content)], role in {'user','assistant'}"""
    parts = []
    parts.append("<|begin_of_text|>")
    # system
    parts.append("<|start_header_id|>system<|end_header_id|>\n" + system + "\n<|eot_id|>")
    # history
    for role, content in history:
        parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n{content}\n<|eot_id|>")
    # current user
    parts.append("<|start_header_id|>user<|end_header_id|>\n" + user_msg + "\n<|eot_id|>")
    # assistant to generate
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
    return "".join(parts)

def extract_assistant(text: str) -> str:
    # 마지막 assistant 블록 이후 ~ <|eot_id|> 사이를 답변으로
    seg = text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    seg = seg.split("<|start_header_id|>user<|end_header_id|>")[0]  # 혹시 다음 턴까지 생성되면 잘라줌
    return seg.replace("<|eot_id|>", "").strip()

def log(*a): print("[chat]", *a, flush=True)

def main():
    log("CUDA:", torch.cuda.is_available(), "| device:", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))

    # 1) 토크나이저
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # 2) 베이스 모델 + LoRA 어댑터
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()

    # 3) 대화 루프
    history: List[Tuple[str, str]] = []
    print("\n=== LoRA 챗봇 시작 (종료: exit) ===")
    while True:
        try:
            user = input("\n질문> ").strip()
        except EOFError:
            break
        if user.lower() in {"exit", "quit"}:
            print("대화를 종료합니다.")
            break
        if not user:
            continue

        prompt = build_prompt(SYSTEM_MSG, history, user)
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 4) 스트리밍 생성
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=False)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
            streamer=streamer,
        )

        # 비동기 스레드로 generate 호출
        t = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        t.start()

        # 스트리밍 출력
        sys.stdout.write("\n[응답] ")
        sys.stdout.flush()
        buf = []
        for piece in streamer:
            # special token은 최대한 숨기되, EOT는 제거
            buf.append(piece)
            sys.stdout.write(piece.replace("<|eot_id|>", ""))
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()

        full = "".join(buf)
        ans = extract_assistant(full)
        history.append(("user", user))
        history.append(("assistant", ans))

if __name__ == "__main__":
    try:
        # 출력 버퍼링 방지
        sys.stdout.reconfigure(line_buffering=True)
        main()
    except Exception as e:
        print("[chat] ERROR:", e)
        import traceback; traceback.print_exc()
        sys.exit(1)