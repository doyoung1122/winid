import os, sys, traceback
from typing import Dict
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL_DIR  = r"D:\llama_project\models\llama31-8b-instruct"
DATA_PATH  = r"D:\llama_project\data\sft_policy.jsonl"
OUTPUT_DIR = r"D:\llama_project\outputs\llama31-8b-sft"

def log(*a): print("[train_sft]", *a, flush=True)

def print_gpu_usage(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved  = torch.cuda.memory_reserved() / (1024**3)
        max_alloc = torch.cuda.max_memory_allocated() / (1024**3)
        log(f"[GPU] {tag} Allocated={allocated:.2f} GB, Reserved={reserved:.2f} GB, Max={max_alloc:.2f} GB")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log("MODEL_DIR:", MODEL_DIR)
    log("DATA_PATH:", DATA_PATH)
    log("OUTPUT_DIR:", OUTPUT_DIR)

    # 1) Tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # 2) Dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    raw = load_dataset("json", data_files={"train": DATA_PATH})["train"]
    if len(raw) == 0:
        raise ValueError("Dataset has 0 rows. Check the file contents.")

    def to_prompt_completion(ex: Dict) -> Dict:
        inst = (ex.get("instruction") or "").strip()
        inp  =  ex.get("input") or ""
        out  = (ex.get("output") or "").strip()
        user = inst if not inp else f"{inst}\n\n{inp}"
        return {
            "prompt":     [{"role": "user", "content": user}],
            "completion": [{"role": "assistant", "content": out}],
        }

    ds = raw.map(to_prompt_completion, remove_columns=raw.column_names)
    log("Loaded rows:", len(ds))

    # 3) QLoRA 여부
    use_4bit = False
    try:
        import bitsandbytes as _bnb
        use_4bit = torch.cuda.is_available()
    except Exception:
        use_4bit = False
    log("use_4bit:", use_4bit)

    extra_kwargs = {}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        extra_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        dtype = torch.bfloat16
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # 4) Base model
    log("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        **extra_kwargs
    )
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False
    log("Model loaded.")
    print_gpu_usage("After model load")

    # 5) LoRA
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 6) SFT 설정
    sft_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        save_safetensors=True,
        bf16=(dtype==torch.bfloat16),
        fp16=(dtype==torch.float16),
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        dataloader_num_workers=0,
        max_grad_norm=0.3,
        completion_only_loss=True,
    )

    # 7) Trainer
    log("Building trainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds,
        peft_config=lora_cfg,
        processing_class=tok,
    )
    print_gpu_usage("After trainer init")

    # 8) Training
    log("Start training...")
    trainer.train()
    print_gpu_usage("After training")
    log("Training finished.")

    # 9) 저장
    adapter_dir = os.path.join(OUTPUT_DIR, "adapter")
    ckpt_dir    = os.path.join(OUTPUT_DIR, "final_checkpoint")
    os.makedirs(adapter_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    log("Saving PEFT adapter ->", adapter_dir)
    trainer.model.save_pretrained(adapter_dir)

    log("Saving trainer checkpoint ->", ckpt_dir)
    trainer.save_state()
    trainer.save_model(ckpt_dir)

    log("=== DONE ===")
    log("Adapter:", adapter_dir)
    log("Checkpoint:", ckpt_dir)

if __name__ == "__main__":
    try:
        torch.set_num_threads(1)
        main()
    except Exception as e:
        print("[train_sft] ERROR:", e)
        traceback.print_exc()
        sys.exit(1)