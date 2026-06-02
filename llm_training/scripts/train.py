"""
Train the LCAE LLM LoRA adapter on a local CUDA machine.

Examples:
    python llm_training/scripts/train.py --validate-only
    python llm_training/scripts/train.py
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT_DIR / "llm_training"

DEFAULT_BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_DATASET = TRAINING_DIR / "datasets" / "train_v3.jsonl"
DEFAULT_ADAPTER_DIR = TRAINING_DIR / "adapters_v3"
TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
REQUIRED_OUTPUT_KEYS = {"topic", "intent", "mc_script"}
JSON_TASKS = {"general", "bridge"}
PLAIN_TEXT_TASKS = {"summary", "question"}
ALL_TASKS = JSON_TASKS | PLAIN_TEXT_TASKS


def has_param(cls: type, name: str) -> bool:
    return name in inspect.signature(cls.__init__).parameters


def make_sft_config(sft_config_cls, **kwargs: Any):
    allowed = inspect.signature(sft_config_cls.__init__).parameters
    return sft_config_cls(**{key: value for key, value in kwargs.items() if key in allowed})


def make_trainer(trainer_cls, model, tokenizer, args, dataset):
    kwargs = {
        "model": model,
        "args": args,
        "train_dataset": dataset,
    }
    if has_param(trainer_cls, "processing_class"):
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return trainer_cls(**kwargs)


def print_device_info() -> None:
    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    print(f"GPU: {props.name}")
    print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
    major, minor = torch.cuda.get_device_capability(idx)
    print(f"CUDA capability: {major}.{minor}")


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return records


def validate_records(records: list[dict[str, Any]]) -> Counter:
    from collections import Counter

    errors: list[str] = []
    counts: Counter = Counter()
    for idx, row in enumerate(records, 1):
        task = row.get("task", "general")
        counts[task] += 1
        if task not in ALL_TASKS:
            errors.append(f"{idx}: unsupported task {task!r}")
            continue

        messages = row.get("messages")
        if not isinstance(messages, list):
            errors.append(f"{idx}: messages must be a list")
            continue
        roles = [msg.get("role") for msg in messages if isinstance(msg, dict)]
        if roles != ["system", "user", "assistant"]:
            errors.append(f"{idx}: roles must be system,user,assistant")
            continue
        output_text = messages[-1].get("content", "").strip()

        if task in JSON_TASKS:
            try:
                output = json.loads(output_text)
            except Exception:
                errors.append(f"{idx}: {task} assistant content must be JSON")
                continue
            if set(output) != REQUIRED_OUTPUT_KEYS:
                errors.append(f"{idx}: assistant keys must be {sorted(REQUIRED_OUTPUT_KEYS)}")
        elif task in PLAIN_TEXT_TASKS:
            if not output_text:
                errors.append(f"{idx}: {task} assistant content must not be empty")
            if output_text.startswith("{"):
                errors.append(f"{idx}: {task} assistant content must be plain text")

    if errors:
        joined = "\n".join(errors[:20])
        raise ValueError(f"Dataset validation failed ({len(errors)} errors):\n{joined}")
    return counts


def parse_tasks(raw: str) -> set[str]:
    if raw.strip().lower() == "all":
        return set(ALL_TASKS)
    tasks = {part.strip() for part in raw.split(",") if part.strip()}
    unknown = tasks - ALL_TASKS
    if unknown:
        raise ValueError(f"Unknown task(s): {', '.join(sorted(unknown))}")
    if not tasks:
        raise ValueError("At least one task is required")
    return tasks


def filter_records(records: list[dict[str, Any]], tasks: set[str]) -> list[dict[str, Any]]:
    return [row for row in records if row.get("task", "general") in tasks]


def format_dataset(records: list[dict[str, Any]], tokenizer):
    from datasets import Dataset

    dataset = Dataset.from_list(records)

    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    return dataset.map(format_chat, remove_columns=dataset.column_names)


def token_length_report(dataset: Dataset, tokenizer, max_seq_length: int) -> None:
    lengths = [len(tokenizer(row["text"], add_special_tokens=False)["input_ids"]) for row in dataset]
    if not lengths:
        print("Dataset is empty")
        return
    over = sum(1 for length in lengths if length > max_seq_length)
    sorted_lengths = sorted(lengths)
    p95 = sorted_lengths[int(len(sorted_lengths) * 0.95) - 1]
    print(
        "Token lengths: "
        f"min={min(lengths)} avg={sum(lengths)/len(lengths):.1f} "
        f"p95={p95} max={max(lengths)} over_max={over}/{len(lengths)}"
    )


def train(args: argparse.Namespace) -> None:
    tasks = parse_tasks(args.tasks)

    if args.validate_only:
        records = load_records(Path(args.dataset))
        counts = validate_records(records)
        filtered = filter_records(records, tasks)
        print(f"Dataset OK: {args.dataset} ({len(records)} rows)")
        print(f"Task distribution: {dict(sorted(counts.items()))}")
        print(f"Selected tasks: {', '.join(sorted(tasks))} ({len(filtered)} rows)")
        return

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    print_device_info()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 필요한 학습 스크립트입니다.")

    adapter_dir = Path(args.adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/5] Tokenizer load: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_seq_length

    print(f"\n[2/5] Dataset load: {args.dataset}")
    records = load_records(Path(args.dataset))
    counts = validate_records(records)
    records = filter_records(records, tasks)
    if not records:
        raise RuntimeError(f"No records selected for tasks: {', '.join(sorted(tasks))}")
    dataset = format_dataset(records, tokenizer)
    print(f"Task distribution: {dict(sorted(counts.items()))}")
    print(f"Selected tasks: {', '.join(sorted(tasks))}")
    print(f"Samples: {len(dataset)}")
    print(dataset[0]["text"][:500])
    token_length_report(dataset, tokenizer, args.max_seq_length)

    print("\n[3/5] Model load: 4-bit QLoRA")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\n[4/5] Train")
    sft_args = make_sft_config(
        SFTConfig,
        output_dir=str(adapter_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=False,
        fp16=True,
        dataset_text_field="text",
        max_length=args.max_seq_length,
        max_seq_length=args.max_seq_length,
        packing=False,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    trainer = make_trainer(SFTTrainer, model, tokenizer, sft_args, dataset)

    # QLoRA convention: keep trainable adapter weights in FP32 while computing in FP16.
    for _, param in trainer.model.named_parameters():
        if param.requires_grad and param.dtype in (torch.bfloat16, torch.float16):
            param.data = param.data.to(torch.float32)

    trainer.train()

    print(f"\n[5/5] Save adapter: {adapter_dir}")
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print("Training done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local QLoRA training for LCAE LLM")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--tasks",
        default="all",
        help="Comma-separated tasks to train/validate: general,bridge,summary,question or all",
    )

    # Defaults target a 24GB-class CUDA GPU. If OOM happens, reduce batch-size to 1 or max-seq-length to 1024.
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager"])
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
