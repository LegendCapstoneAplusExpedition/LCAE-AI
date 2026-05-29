'''
기존 API 호출 방식은 토큰 소모가 크고 학습에 용이하지 못해 Ollama 모델을 받아서 씀 (https://ollama.com/)
다만 Ollama 모델 파일 자체가 정적 데이터로 고정됨. 내부에서 가중치 업데이트를 위한 연산은 힘듦.
모델의 가중치를 수정하는 것이 아니라 Modelfile을 활용 모델 초기 상태와 샘플 데이터를 고정하는 방식 채택함
그리고 LoRA 연산을 직접적으로 수행할 순 없음. 대신 다른 프레임워크에서 학습 후 추출한 LoRA 가중치를 Ollama에 결합하여 구동할 수 있음.

학습 명령어(adapter 생성)
uv run mlx_lm.lora \
    --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
    --train \
    --data ./data \
    --iters 300 \
    --batch-size 2 \
    --num-layers 16 \
    --learning-rate 1e-5

가중치 병합
uv run mlx_lm.fuse \
    --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
    --adapter ./adapters \
    --save-path ./merged_model \
    --dequantize

[llama.cpp 의존성 설치]
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
uv pip install -r requirements.txt

[GGUF 변환 및 8-bit 양자화 적용(기본 모델)]
python convert_hf_to_gguf.py ../merged_model \
    --outfile ../finetuned_llama3_1.gguf \
    --outtype q8_0

[4-bit 양자화 적용 모델]
# FP16 상태의 폴더 구조를 GGUF 파일 1개로 직렬화
python convert_hf_to_gguf.py ../merged_model \
    --outfile ../finetuned_llama3_1_fp16.gguf

4-bit 블록 압축 연산
./build/bin/llama-quantize ../finetuned_llama3_1_fp16.gguf ../finetuned_llama3_1_Q4_K_M.gguf Q4_K_M


[모델 레지스트리 주입]
ollama create podcast-mc -f ./Modelfile
ollama create podcast-mc-q4 -f ./Modelfile_q4

ollama run podcast-mc-q4
--------------------------------------------------------------------------------
모델 삭제
ollama --rm podcast-mc
'''

from langchain_ollama import ChatOllama

# 구조화 출력용 (preprocess/analyzer) — JSON 모드 + 낮은 temperature
llm_structured = ChatOllama(
    model       = "podcast-mc-q4",
    temperature = 0.1,
    format      = "json",
)

# 자유 텍스트 생성용 (script_writer) — JSON 모드 OFF + 적절한 temperature
llm = ChatOllama(
    model       = "podcast-mc-q4",
    temperature = 0.7,
)