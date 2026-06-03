# LLM Training

LLM 학습 전용 폴더입니다. 런타임 RAG 입력은 루트의 `data/`를 사용하고, 학습 데이터와 노트북/스크립트는 이 폴더 안에서 관리합니다.

## Structure

- `datasets/`: SFT 학습 데이터
- `notebooks/`: Kaggle/Colab 학습 및 병합 노트북
- `scripts/train.py`: 로컬 학습 스크립트
- `scripts/merge.py`: 로컬 병합 스크립트
- `notebooks/kaggle_train_sft_v3.ipynb`: `train_v3.jsonl` 학습
- `notebooks/colab_merge_to_gguf.ipynb`: LoRA 어댑터 병합 및 GGUF 변환

## Current Dataset

- `datasets/train_v3.jsonl`: 현재 LLM 플로우 기준 데이터
  - `general`: 일반 의도 분류, 질문요청, 정리요청, 마무리, 대기
  - `bridge`: 설명/질문 상황에서 짧은 브릿지 멘트 생성
  - `summary`: 실제 전사 기반 방송 요약
  - `question`: 청취자 질문 전달 멘트
- `datasets/train_v2.jsonl`: 이전 플로우 데이터
- `datasets/train_v1.jsonl`: 초기 데이터

## Run In Kaggle

1. `notebooks/kaggle_train_sft_v3.ipynb`로 `adapters_v3`를 생성합니다.
2. 생성된 어댑터를 Kaggle Dataset으로 저장합니다.
3. `notebooks/colab_merge_to_gguf.ipynb`에서 어댑터를 병합하고 GGUF로 변환합니다.

## Run Locally

CUDA용 PyTorch를 먼저 설치한 뒤 학습 의존성을 설치합니다.
`gguf.py`가 llama.cpp를 clone/build하므로 시스템에 `git`도 필요합니다.

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchaudio
pip install -r llm_training/requirements.txt
```

Hugging Face 토큰이 필요한 경우 로그인합니다.

```bash
huggingface-cli login
```

데이터 검증, 학습, 병합을 실행합니다.

```bash
python llm_training/scripts/train.py --validate-only
python llm_training/scripts/train.py
python llm_training/scripts/merge.py --smoke-test
python llm_training/scripts/gguf.py
```

특정 기능만 골라 학습/검증할 수도 있습니다.

```bash
python llm_training/scripts/train.py --validate-only --tasks general,bridge
python llm_training/scripts/train.py --tasks summary,question
```

기본 출력 경로:

- LoRA adapter: `llm_training/adapters_v3/`
- Merged model: `llm_training/merged_model_v3/`
- Q4 GGUF: `llm_training/gguf/driving-mentor-v3-q4_k_m.gguf`

OOM이 나면 아래처럼 줄입니다.

```bash
python llm_training/scripts/train.py --batch-size 1
python llm_training/scripts/train.py --max-seq-length 1024
```
