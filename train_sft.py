import os
import json
import random
import warnings
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# =========================
# 환경변수 & 기본값
# =========================
MODEL_DIR   = os.getenv("LLM_MODEL_DIR",   r"D:\llama_project\models\llama31-8b-instruct")
DATA_PATH   = os.getenv("DATA_PATH",       r"D:\llama_project\data\sft_policy.jsonl")
OUTPUT_DIR  = os.getenv("OUTPUT_DIR",      r"D:\llama_project\outputs\llama31-8b-sft")

SEED        = int(os.getenv("SEED", "42"))
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", "1"))
GRAD_ACC    = int(os.getenv("GRAD_ACC", "8"))
LR          = float(os.getenv("LR", "2e-4"))
EPOCHS      = float(os.getenv("EPOCHS", "1"))
MAX_LEN     = int(os.getenv("MAX_LEN", "4096"))
BF16        = os.getenv("BF16", "1") == "1"
USE_GC      = os.getenv("GRADIENT_CHECKPOINTING", "1") == "1"

TARGET_MODULES = os.getenv(
    "LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
).split(",")
LORA_R       = int(os.getenv("LORA_R", "64"))
LORA_ALPHA   = int(os.getenv("LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))

# =========================
# 유틸
# =========================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                continue
    return rows

# =========================
# QLoRA 베이스 모델
# =========================
def load_qlora_base(model_dir: str, bf16: bool):
    compute_dtype = torch.bfloat16 if bf16 and torch.cuda.is_available() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    print("[QLoRA] Loading base model in 4bit ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        trust_remote_code=False,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=USE_GC)
    return model

def attach_lora(model):
    peft_conf = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_conf)
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    print(f"[QLoRA] Trainable params: {trainable:,} / Total: {total:,} "
          f"({100*trainable/total:.2f}%)")
    return model

# =========================
# Collator: 경계 인덱스 기반 마스킹
# =========================
class BoundaryMaskCollator:
    """
    각 feature는 {"text": full_text, "prefix_len": int} 형태.
    labels[:prefix_len] = -100, labels[prefix_len:] = input_ids[prefix_len:].
    """
    def __init__(self, tokenizer, max_length: int):
        self.tok = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        texts = [f.get("text", "") for f in features]
        prefix_lens = [int(f.get("prefix_len", 0)) for f in features]

        batch = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        for i, pref in enumerate(prefix_lens):
            if pref <= 0:
                labels[i, :] = -100
                continue
            # 유효 토큰 길이 기준 클램프
            if self.tok.pad_token_id is None:
                L = input_ids[i].shape[0]
            else:
                L = (input_ids[i] != self.tok.pad_token_id).sum().item()
            pref = max(0, min(pref, L))
            labels[i, :pref] = -100

        batch["labels"] = labels
        return batch

# =========================
# 메인
# =========================
def main():
    set_seed(SEED)
    print(f"[QLoRA] MODEL_DIR : {MODEL_DIR}")
    print(f"[QLoRA] DATA_PATH : {DATA_PATH}")
    print(f"[QLoRA] OUTPUT_DIR: {OUTPUT_DIR}")

    # ----- 데이터 로드 & 정규화
    raw_rows = load_jsonl(DATA_PATH)
    if not raw_rows:
        raise RuntimeError(f"No data loaded from {DATA_PATH}")

    norm_rows = []
    for r in raw_rows:
        if "messages" in r and isinstance(r["messages"], list) and r["messages"]:
            msgs = r["messages"]
        else:
            q = ((r.get("input") or {}).get("question") or "").strip()
            a = (r.get("output") or "").strip()
            if not q or not a:
                continue
            sys_ = (r.get("instruction") or "너는 RAG 기반 도우미야.").strip()
            msgs = [
                {"role":"system","content":sys_},
                {"role":"user","content":q},
                {"role":"assistant","content":a},
            ]
        if not msgs or msgs[-1].get("role")!="assistant" or not (msgs[-1].get("content") or "").strip():
            continue
        norm_rows.append({"messages": msgs})

    if not norm_rows:
        raise RuntimeError("No valid rows (must end with non-empty assistant).")
    print(f"[QLoRA] Loaded rows (normalized): {len(norm_rows)}")

    # ----- 토크나이저
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # ----- 텍스트 & 경계(prefix_len) 생성
    def to_full_and_boundary(ex: Dict):
        msgs = ex["messages"]
        full_text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        prefix_text = tok.apply_chat_template(
            msgs[:-1] + [{"role":"assistant","content":""}],
            tokenize=False, add_generation_prompt=False
        )
        prefix_ids = tok(prefix_text, add_special_tokens=False)["input_ids"]
        return {"text": full_text, "prefix_len": len(prefix_ids)}

    ds = Dataset.from_list(norm_rows)
    ds = ds.map(to_full_and_boundary, remove_columns=ds.column_names)  # ← text/prefix_len만 남김
    ds = ds.filter(lambda ex: ex["prefix_len"] > 0)

    # ----- 모델 + LoRA
    model = load_qlora_base(MODEL_DIR, BF16)
    if USE_GC:
        model.gradient_checkpointing_enable()
    model = attach_lora(model)

    # ----- collator
    collator = BoundaryMaskCollator(tok, MAX_LEN)

    # ----- SFT 설정
    sft_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        logging_steps=10,
        save_steps=200,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=BF16,
        gradient_checkpointing=USE_GC,
        packing=False,                    # completions-only 마스킹과 호환 위해 False
        optim="paged_adamw_32bit",
        dataset_text_field="text",
        max_seq_length=MAX_LEN,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tok,  # TRL 0.9.6에서는 tokenizer 인자 사용
    )

    # ----- sanity: 라벨 토큰이 실제로 있는지 점검
    try:
        sample = [ds[i] for i in range(min(3, len(ds)))]
        b = collator(sample)
        counts = (b["labels"] != -100).sum(dim=1).tolist()
        print("[Sanity] labeled-token counts per sample:", counts)
    except Exception as e:
        print("[Sanity] check failed:", e)

    # ----- 학습
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.train()

    # ----- 저장
    print("[QLoRA] Saving adapter ...")
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "adapter"))
    tok.save_pretrained(OUTPUT_DIR)
    print("[QLoRA] Done.")

if __name__ == "__main__":
    main()