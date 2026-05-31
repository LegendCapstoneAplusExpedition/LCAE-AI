"""
LoRA 어댑터를 베이스 모델에 병합 → 독립 실행 가능한 merged_model 저장
필요 RAM: ~20GB (FP16 기준), VRAM 불필요 (CPU 병합)
Usage: python scripts/merge_adapter.py
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ADAPTER_DIR = Path(__file__).parent.parent / "adapters_v2"
BASE_MODEL  = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR  = Path(__file__).parent.parent / "merged_model"


def main():
    print(f"어댑터 경로 : {ADAPTER_DIR}")
    print(f"출력 경로   : {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 베이스 모델 — 양자화 없이 CPU FP16 로드 (VRAM 불필요)
    print("\n[1/4] 베이스 모델 로드 (CPU, FP16)...")
    print("  ⚠ 약 15~20GB RAM 필요 / 수 분 소요")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # LoRA 어댑터 로드
    print(f"\n[2/4] LoRA 어댑터 로드 ({ADAPTER_DIR.name})...")
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))

    # 병합 (adapter weights → base model weights)
    print("\n[3/4] 어댑터 병합 중 (merge_and_unload)...")
    model = model.merge_and_unload()
    print("  병합 완료")

    # 저장
    print(f"\n[4/4] 병합 모델 저장 → {OUTPUT_DIR}")
    print("  ⚠ 약 15GB 디스크 필요 / 수 분 소요")
    model.save_pretrained(str(OUTPUT_DIR), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR), trust_remote_code=True)
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"\n완료. 병합 모델: {OUTPUT_DIR}")
    print("  로드 예시:")
    print(f"    from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"    model = AutoModelForCausalLM.from_pretrained('{OUTPUT_DIR}', torch_dtype=torch.float16, device_map='auto')")


if __name__ == "__main__":
    main()
