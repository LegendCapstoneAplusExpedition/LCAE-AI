"""
Merge the trained LoRA adapter into the base Llama model.

Examples:
    python llm_training/scripts/merge.py
    python llm_training/scripts/merge.py --smoke-test
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT_DIR / "llm_training"

DEFAULT_BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_ADAPTER_DIR = TRAINING_DIR / "adapters_v3"
DEFAULT_MERGED_DIR = TRAINING_DIR / "merged_model_v3"


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


def cleanup_cuda() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def merge(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = Path(args.adapter_dir)
    merged_dir = Path(args.merged_dir)
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_dir}")
    merged_dir.mkdir(parents=True, exist_ok=True)

    print_device_info()
    if args.merge_device == "cuda":
        device_map: str | dict[str, int] = {"": 0}
    elif args.merge_device == "cpu":
        device_map = "cpu"
    else:
        device_map = {"": 0} if torch.cuda.is_available() else "cpu"

    print(f"\n[1/4] Base model load: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print(f"\n[2/4] Adapter load: {adapter_dir}")
    model = PeftModel.from_pretrained(model, str(adapter_dir))

    print("\n[3/4] Merge adapter into base model")
    model = model.merge_and_unload()

    print(f"\n[4/4] Save merged model: {merged_dir}")
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    tokenizer.save_pretrained(str(merged_dir))
    print("Merge done.")

    del model
    cleanup_cuda()


def smoke_test(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_dir = Path(args.merged_dir)
    print(f"\n[SmokeTest] Load merged model: {merged_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(merged_dir),
        torch_dtype=torch.float16,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    prompts = {
        "bridge": [
            {
                "role": "system",
                "content": "너는 방송 진행을 보조하는 AI MC다. JSON만 출력한다.",
            },
            {
                "role": "user",
                "content": (
                    "[방송 주제]: MVP 개발\n"
                    "[현재 주제]: MVP 검증\n"
                    "[이전 단계]: Main\n"
                    "[멘토 발화]: \"MVP는 핵심 기능 하나를 빠르게 검증하는 게 중요합니다.\"\n"
                    "반드시 JSON 형식으로만 출력하세요."
                ),
            },
        ],
        "summary": [
            {
                "role": "system",
                "content": "당신은 방송 요약 전용 모델입니다. 전사에 있는 내용만 1~3문장으로 요약합니다.",
            },
            {
                "role": "user",
                "content": (
                    "다음은 방송에서 실제로 전사된 멘토 발화 목록입니다.\n\n"
                    "[2026-06-02 15:00:00] MVP는 가장 위험한 가설 하나를 빠르게 검증하는 방식입니다.\n"
                    "[2026-06-02 15:00:20] 기능을 많이 넣으면 검증 속도가 느려지고 판단이 어려워집니다.\n\n"
                    "요약 규칙:\n"
                    "- 위 전사에 명시된 내용만 바탕으로 요약하세요.\n"
                    "- 1~3문장으로, 다른 설명 없이 요약문만 출력하세요."
                ),
            },
        ],
        "question": [
            {
                "role": "system",
                "content": "당신은 방송의 AI MC 질문 전달자입니다. 질문 전달 멘트만 출력합니다.",
            },
            {
                "role": "user",
                "content": (
                    "[청취자]: 민지\n"
                    "[질문]: 사이드 프로젝트를 시작할 때 고객 인터뷰는 몇 명 정도 해보는 게 좋을까요?\n\n"
                    "방송에서 바로 읽을 수 있는 질문 전달 멘트만 출력하세요."
                ),
            },
        ],
    }

    for name, prompt in prompts.items():
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        print(f"\n[{name}]")
        print(generated.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge trained LoRA adapter for LCAE LLM")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--merged-dir", type=Path, default=DEFAULT_MERGED_DIR)
    parser.add_argument("--merge-device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge(args)
    if args.smoke_test:
        smoke_test(args)


if __name__ == "__main__":
    main()
