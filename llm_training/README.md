# LLM Training

LLM 학습 전용 폴더입니다. 런타임 RAG 입력은 루트의 `data/`를 사용하고, 학습 데이터와 노트북/스크립트는 이 폴더 안에서 관리합니다.

## Structure

- `datasets/`: SFT 학습 데이터
- `notebooks/`: Kaggle/Colab 학습 및 병합 노트북
- `notebooks/kaggle_train_sft_v3.ipynb`: `train_v3.jsonl` 학습
- `notebooks/colab_merge_to_gguf.ipynb`: LoRA 어댑터 병합 및 GGUF 변환

## Current Dataset

- `datasets/train_v3.jsonl`: 현재 LLM 플로우 기준 데이터
- `datasets/train_v2.jsonl`: 이전 플로우 데이터
- `datasets/train.jsonl`: 초기 데이터

## Run In Kaggle

1. `notebooks/kaggle_train_sft_v3.ipynb`로 `adapters_v3`를 생성합니다.
2. 생성된 어댑터를 Kaggle Dataset으로 저장합니다.
3. `notebooks/colab_merge_to_gguf.ipynb`에서 어댑터를 병합하고 GGUF로 변환합니다.
