"""
SFT fine-tuning script for Llama 3.1 8B Instruct (Windows/Linux CUDA)
Usage: python scripts/train_sft.py
"""

import os
import json
import pyarrow
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ── 설정 ──────────────────────────────────────────────────────────────────────
BASE_MODEL   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATA_PATH    = Path(__file__).parent.parent / "data" / "train_v2.jsonl"
OUTPUT_DIR   = Path(__file__).parent.parent / "adapters_v2"

LORA_RANK        = 8
LORA_ALPHA       = 16
LORA_DROPOUT     = 0.05
TARGET_MODULES   = ["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]

BATCH_SIZE       = 1
GRAD_ACCUM       = 8          # 유효 배치 = 8 (RTX 2060 6GB 대응)
LEARNING_RATE    = 2e-4
NUM_EPOCHS       = 3
MAX_SEQ_LENGTH   = 1024       # 2048 → 1024: VRAM 절감
SAVE_STEPS       = 100
LOGGING_STEPS    = 10
# ──────────────────────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> Dataset:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def format_chat(example, tokenizer):
    """messages 리스트 → 단일 텍스트 (chat template 적용)"""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── 토크나이저 ──────────────────────────────────────────────────────────
    print("\n[1/4] 토크나이저 로드...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = MAX_SEQ_LENGTH

    # ── 데이터셋 ────────────────────────────────────────────────────────────
    print("[2/4] 데이터 로드...")
    raw_dataset = load_jsonl(DATA_PATH)
    dataset = raw_dataset.map(lambda ex: format_chat(ex, tokenizer))
    print(f"  총 {len(dataset)}개 샘플")
    print(f"  예시: {dataset[0]['text'][:120]}...")

    # ── 모델 (4-bit QLoRA) ──────────────────────────────────────────────────
    print("[3/4] 모델 로드 (4-bit 양자화)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # RTX 20xx: bf16 미지원
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map={"": 0},           # 전체 GPU 0에 강제 배치 (4-bit CPU 오프로드 미지원)
        torch_dtype=torch.float16,    # RTX 20xx: BF16 미지원, FP16 강제
        trust_remote_code=True,
        attn_implementation="sdpa",   # flash-attn 없이 Windows에서 동작
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 학습 ────────────────────────────────────────────────────────────────
    print("[4/4] 학습 시작...")
    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=25,
        bf16=False,
        fp16=True,
        dataset_text_field="text",
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        report_to="none",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # LoRA adapter가 BF16으로 초기화됨 → FP32로 변환 (QLoRA 정석: adapter=FP32, compute=FP16)
    for name, param in trainer.model.named_parameters():
        if param.requires_grad and param.dtype in (torch.bfloat16, torch.float16):
            param.data = param.data.to(torch.float32)

    trainer.train()

    print(f"\n학습 완료. 어댑터 저장 중 → {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("완료.")


if __name__ == "__main__":
    main()
