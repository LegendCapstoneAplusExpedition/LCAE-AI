# PRD: Whisper 한국어 파인튜닝 파이프라인

| 항목 | 내용 |
|---|---|
| 모듈명 | `ASR Finetuning Pipeline` (asr_Finetuning.py) |
| 작성일 | 2026-04-18 |
| 버전 | v1.0 |
| 작성자 | sucheoli |

---

## 1. 개요

OpenAI Whisper 모델을 한국어 음성 데이터로 파인튜닝하여 한국어 전사 정확도를 향상시키는 오프라인 학습 파이프라인. 학습 완료 후 HuggingFace 체크포인트를 CTranslate2 형식으로 변환하여 실시간 추론 모듈(`asr_pipeline.py`)에서 바로 사용할 수 있는 faster-whisper 모델을 생성한다.

---

## 2. 배경 및 목표

### 배경

기존 `asr_pipeline.py`는 Whisper 기본 모델(`openai/whisper-base`)을 사용한다. 기본 모델은 범용 한국어는 인식하지만 도메인 특화 어휘나 억양에서 오류율이 높다. 파인튜닝을 통해 캡스톤 프로젝트 도메인에 최적화된 모델을 생성하고 WER을 개선한다.

### 목표

- 한국어 공개 음성 데이터셋(Common Voice, KsponSpeech)으로 Whisper 파인튜닝
- 파인튜닝 후 faster-whisper 형식으로 자동 변환
- 변환된 모델을 `PipelineConfig(model=<경로>)`로 즉시 추론에 투입

### 성공 기준

| 지표 | 목표값 |
|---|---|
| WER (한국어, Common Voice test) | 기본 모델 대비 상대적 20% 이상 감소 |
| 파인튜닝 재현성 | 동일 config로 동일 결과 재현 가능 |
| 변환 성공률 | ct2-transformers-converter 정상 완료 |
| 추론 호환성 | PipelineConfig(model=<변환 경로>) 로드 성공 |

---

## 3. 범위 (Scope)

### In Scope

- HuggingFace `transformers` 기반 Whisper 파인튜닝 (`Seq2SeqTrainer`)
- 한국어 데이터셋 로드 및 전처리 (Common Voice ko, KsponSpeech audiofolder)
- WER 기반 체크포인트 선택 및 저장
- CTranslate2 형식 자동 변환 (`ct2-transformers-converter`)
- CLI 인터페이스

### Out of Scope

- 실시간 오디오 스트리밍 (asr_pipeline.py 담당)
- 분산 학습 / 멀티 노드 학습
- 화자 분리, 다국어 동시 인식
- 모델 양자화 이외의 압축 기법 (pruning, distillation)
- HuggingFace Hub 업로드

---

## 4. 시스템 인터페이스 명세

### 4.1 입력

| 항목 | 명세 |
|---|---|
| 음성 데이터 | PCM 오디오, 16kHz, Mono |
| 데이터 소스 | HuggingFace Hub (`common_voice_11_0/ko`) 또는 로컬 audiofolder 디렉터리 |
| 텍스트 레이블 | UTF-8 한국어 전사 텍스트 |
| 베이스 모델 | `openai/whisper-{size}` (tiny / base / small / medium / large-v3) |

### 4.2 출력

| 항목 | 형식 | 경로 |
|---|---|---|
| HF 체크포인트 | HuggingFace `PreTrainedModel` 디렉터리 | `./checkpoints/checkpoint-{step}` |
| faster-whisper 모델 | CTranslate2 바이너리 디렉터리 | `./faster-whisper-finetuned/` |

### 4.3 downstream 연동

변환된 모델은 `asr_pipeline.py`의 `PipelineConfig`에 경로를 지정하여 즉시 사용:

```python
from pipeline.stt import PipelineConfig, WhisperTranscriber

config = PipelineConfig(model="./faster-whisper-finetuned")
transcriber = WhisperTranscriber(config)
```

또는 `.env`의 `ASR_MODEL` 값을 변환 경로로 변경:

```dotenv
ASR_MODEL=./faster-whisper-finetuned
```

---

## 5. 기능 요구사항

| ID | 요구사항 | 우선순위 |
|---|---|---|
| FR-01 | Common Voice ko 데이터셋 자동 다운로드 및 전처리 | 필수 |
| FR-02 | 로컬 KsponSpeech audiofolder 형식 지원 (`--dataset-path`) | 필수 |
| FR-03 | 오디오 자동 리샘플링 (16kHz) | 필수 |
| FR-04 | log-mel spectrogram 특징 추출 (80 mel, 3000 time frames) | 필수 |
| FR-05 | 한국어 forced decoder IDs 설정 (언어 자동 감지 비활성화) | 필수 |
| FR-06 | WER 기반 최적 체크포인트 자동 선택 (`load_best_model_at_end`) | 필수 |
| FR-07 | 학습 완료 후 `ct2-transformers-converter` 자동 실행 (`--convert`) | 필수 |
| FR-08 | GPU 자동 감지 및 fp16 학습 활성화 | 권장 |
| FR-09 | max_label_length 초과 샘플 자동 필터링 | 필수 |
| FR-10 | CLI로 모델 크기, 배치 크기, 에폭 수 지정 | 필수 |

---

## 6. 비기능 요구사항

| ID | 요구사항 |
|---|---|
| NFR-01 | Windows 환경에서 `num_proc=1` (multiprocessing deadlock 방지) |
| NFR-02 | `.env`의 `ASR_MODEL`, `ASR_LANGUAGE`, `HF_ENDPOINT` 재사용 |
| NFR-03 | `HF_ENDPOINT` 환경변수를 통한 HuggingFace 미러 자동 적용 |
| NFR-04 | 로깅: Python `logging` 모듈 사용 (`print` 금지) |
| NFR-05 | GPU OOM 대응 안내: `--batch-size 2~4` 축소 권장 메시지 |
| NFR-06 | 코드 스타일: `asr_pipeline.py` 규칙 준수 (dataclass, 한국어/영어 이중 docstring) |

---

## 7. 데이터 흐름 및 아키텍처

### 7.1 전체 파이프라인

```
[CLI args + .env]
        │
        ▼
FinetuningConfig (dataclass)
        │
  ┌─────┴──────────────────────────────┐
  │                                    │
  ▼                                    ▼
KoreanDatasetLoader               WhisperFinetuner
  load_dataset(hub or local)        WhisperProcessor.from_pretrained()
  cast_column Audio(16kHz)          WhisperForConditionalGeneration.from_pretrained()
  map(_prepare_dataset)             forced_decoder_ids = [<ko>, <transcribe>]
    ├ feature_extractor → (80,3000)  suppress_tokens = []
    └ tokenizer → token_ids                  │
  filter(len ≤ 448)                          │
        │                                    │
        └──── DatasetDict ─────────────────►│
                                    WhisperDataCollator
                                      pad input_features (feature_extractor.pad)
                                      pad labels (tokenizer.pad) → -100 masking
                                             │
                                             ▼
                                    Seq2SeqTrainer.train()
                                      eval: compute_metrics → WER (×100)
                                      best checkpoint (load_best_model_at_end)
                                             │
                                   [--convert 플래그]
                                             │
                                             ▼
                                    ModelConverter.convert()
                                      subprocess: ct2-transformers-converter
                                        --model <best_ckpt>
                                        --output_dir <faster_whisper_dir>
                                        --quantization int8 --force
                                             │
                                             ▼
                                    faster-whisper 모델 디렉터리
                                             │
                                             ▼
                              PipelineConfig(model=<dir>) → 추론
```

### 7.2 클래스 책임

| 클래스 | 책임 |
|---|---|
| `FinetuningConfig` | 모든 하이퍼파라미터 및 경로 중앙 관리 |
| `WhisperDataCollator` | Seq2Seq 배치 패딩 및 -100 레이블 마스킹 |
| `KoreanDatasetLoader` | 데이터셋 로드, 리샘플링, 특징 추출, 토크나이징 |
| `WhisperFinetuner` | 모델 초기화, Seq2SeqTrainer 빌드, WER 계산, 학습 실행 |
| `ModelConverter` | HF 체크포인트 → CTranslate2 변환 (subprocess) |

---

## 8. 기술 스택

| 패키지 | 버전 | 용도 |
|---|---|---|
| `transformers` | >=4.36.0 | Whisper 모델, Seq2SeqTrainer |
| `datasets` | >=2.6.1 | 데이터셋 로드 및 전처리 |
| `evaluate` | >=0.30 | WER 메트릭 |
| `accelerate` | >=1.13.0 | Seq2SeqTrainer 분산 학습 지원 |
| `torch` | >=2.0.0 | 학습 프레임워크, fp16 |
| `ctranslate2` | 최신 | faster-whisper 형식 변환 |
| `python-dotenv` | >=1.0.0 | .env 로드 |

---

## 9. CLI 사용 예시

```bash
# Common Voice ko로 base 모델 파인튜닝 후 자동 변환
python pipeline/stt/asr_Finetuning.py \
  --model base \
  --epochs 3 \
  --batch-size 8 \
  --convert

# 로컬 KsponSpeech로 small 모델 파인튜닝 (변환 별도 실행)
python pipeline/stt/asr_Finetuning.py \
  --model small \
  --dataset ksponspeech \
  --dataset-path /data/kspon \
  --output-dir ./kspon-checkpoints \
  --epochs 5

# GPU OOM 대응: 배치 크기 축소 + gradient accumulation 으로 유효 배치 유지
python pipeline/stt/asr_Finetuning.py \
  --model small \
  --batch-size 4 \
  --convert
```

---

## 10. 테스트 시나리오

### 단위 테스트

- `WhisperDataCollator.__call__()`: labels의 pad_token_id 위치가 -100으로 치환되는지 확인
- `KoreanDatasetLoader._filter_long_labels()`: max_label_length 초과 샘플 필터 동작 확인
- `ModelConverter._check_converter_available()`: CLI 미설치 시 RuntimeError 발생 확인

### 통합 테스트

1. `python asr_Finetuning.py --model tiny --epochs 1 --batch-size 2` 실행
2. 첫 eval 스텝에서 WER 로그 출력 확인
3. `--convert` 플래그로 `./faster-whisper-finetuned` 생성 확인
4. 생성된 모델 경로로 `WhisperTranscriber` 로드 성공 확인

### 성능 검증

- Common Voice ko test split에서 기본 모델 WER vs. 파인튜닝 모델 WER 비교
- RTF 측정: faster-whisper 변환 후 `asr_pipeline.py --mode mic` 로 실시간 성능 확인

---

## 11. 향후 고려 사항

- **AIHub 공식 데이터셋 연동**: `load_dataset("audiofolder")` 경로 지정으로 지원 가능
- **LoRA 파인튜닝**: PEFT 라이브러리 적용으로 소형 GPU(8GB 이하)에서도 large 모델 파인튜닝 가능
- **데이터 증강**: 속도 변환, 노이즈 추가 등으로 강건성 향상
- **스트리밍 학습**: 대용량 데이터셋을 위한 `IterableDataset` 전환
- **WER → CER**: 한국어 음절 기반 CER 메트릭 추가 옵션
