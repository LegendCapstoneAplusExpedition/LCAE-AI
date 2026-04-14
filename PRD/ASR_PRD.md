# PRD: 백엔드 음성 수신 → Whisper 텍스트 변환 모듈

| 항목 | 내용 |
|---|---|
| 모듈명 | `ASR Pipeline` (asr_pipeline.py) |
| 작성일 | 2026-04-10 |
| 버전 | v1.0 |
| 작성자 | sucheoli |

---

## 1. 개요

백엔드 서버로부터 WebSocket을 통해 실시간 오디오 스트림을 수신하고, Fine-tuning된 Whisper 모델(faster-whisper 형식)로 한국어 음성을 텍스트로 전사(transcription)하여 LLM 모듈에 전달하는 ASR(Automatic Speech Recognition) 파이프라인 모듈.

---

## 2. 배경 및 목표

### 배경

2026 캡스톤 시스템은 사용자의 음성을 실시간으로 인식하여 LLM이 처리할 수 있는 텍스트로 변환하는 기능을 필요로 한다. 기존 `asr_pipeline.py` 프로토타입을 기반으로, 백엔드 연동 인터페이스와 결과 데이터 구조를 명확히 정의하여 리팩토링한다.

### 목표

- 백엔드 WebSocket 서버로부터 오디오를 안정적으로 수신
- Fine-tuning된 한국어 Whisper 모델로 정확한 전사 제공
- 텍스트 + 신뢰도(confidence)를 포함한 결과를 LLM 모듈 콜백으로 전달
- 실시간 처리 성능 확보 (RTF < 1.0 on CPU)

### 성공 기준

| 지표 | 목표값 |
|---|---|
| RTF (Real-Time Factor) | < 1.0 (CPU), < 0.3 (GPU) |
| 전사 지연 (발화 종료 후) | < 2초 |
| 메모리 사용량 | < 2GB (CPU, base 모델) |
| 한국어 WER | Fine-tuning 후 기준 WER 대비 개선 |

---

## 3. 범위 (Scope)

### In Scope

- WebSocket 클라이언트로 백엔드에서 오디오 수신
- VAD(Voice Activity Detection)로 발화 구간 감지
- Fine-tuning된 Whisper → faster-whisper 변환 모델 로드 및 추론
- 전사 결과 (텍스트 + 신뢰도) 콜백 전달
- 동시 다중 스트림 처리 (asyncio)
- 재연결 로직 (지수 백오프)

### Out of Scope

- LLM 처리 및 응답 생성
- TTS (Text-to-Speech)
- 화자 분리 (Speaker Diarization)
- 프론트엔드 UI
- 파인튜닝 코드 (별도 스크립트)

---

## 4. 시스템 인터페이스 명세

### 4.1 오디오 입력 (WebSocket)

| 항목 | 명세 |
|---|---|
| 프로토콜 | WebSocket (ws:// 또는 wss://) |
| 프레임 포맷 | Binary (raw PCM) 또는 Text (base64 인코딩) |
| 오디오 포맷 | PCM 16-bit, 16kHz, Mono |
| 청크 크기 | 설정 가능 (기본: 512 samples = 32ms) |
| 연결 URI | `ws://<host>:<port>/audio` (PipelineConfig로 설정) |

#### WebSocket 이벤트 처리

```
CONNECT   → 연결 수립, 수신 루프 시작
MESSAGE   → 오디오 청크 → VAD → SpeechBuffer → (발화 종료 시) Whisper 전사
CLOSE     → 정상 종료, 리소스 해제
ERROR     → 로깅 후 재연결 시도 (지수 백오프: 1s, 2s, 4s, 8s, 최대 60s)
```

### 4.2 전사 결과 출력 (콜백)

```python
from typing import TypedDict, Callable

class TranscriptionResult(TypedDict):
    text: str           # 전사된 텍스트
    confidence: float   # 세그먼트 평균 log-prob → [0, 1] 정규화
    language: str       # 감지/설정된 언어 코드 (예: "ko")

on_transcription: Callable[[TranscriptionResult], None]
```

**confidence 계산:**
- faster-whisper 세그먼트의 `avg_logprob` 활용
- 정규화: `confidence = exp(avg_logprob)` → 범위 [0, 1]
- 여러 세그먼트인 경우: 세그먼트 길이 가중 평균

### 4.3 LLM 모듈 연동

LLM 모듈은 `on_transcription` 콜백을 구현하여 `ASRCore` 또는 `RealtimeASRPipeline` 초기화 시 주입한다.

```python
def handle_transcription(result: TranscriptionResult) -> None:
    # LLM 모듈에서 구현
    text = result["text"]
    confidence = result["confidence"]
    # downstream 처리...

pipeline = RealtimeASRPipeline(
    config=PipelineConfig(ws_uri="ws://backend:8080/audio"),
    on_transcription=handle_transcription,
)
```

---

## 5. 기능 요구사항

| ID | 요구사항 | 우선순위 |
|---|---|---|
| FR-01 | WebSocket 클라이언트로 백엔드에 연결하여 오디오 스트림 수신 | 필수 |
| FR-02 | 에너지 기반 VAD(EnergyVAD)로 발화/침묵 구간 감지 | 필수 |
| FR-03 | 발화 종료 감지 시 SpeechBuffer에 누적된 오디오를 WhisperTranscriber로 전사 | 필수 |
| FR-04 | 전사 결과에 텍스트 + 신뢰도(confidence) 포함하여 on_transcription 콜백 호출 | 필수 |
| FR-05 | 동시 다중 WebSocket 스트림 처리 (asyncio 기반) | 권장 |
| FR-06 | 언어 기본값 한국어("ko"), PipelineConfig로 변경 가능 | 필수 |
| FR-07 | WebSocket 연결 끊김 시 지수 백오프 재연결 | 필수 |
| FR-08 | Fine-tuning된 로컬 모델 경로 로드 지원 (HuggingFace 모델명 또는 로컬 디렉터리) | 필수 |
| FR-09 | torch 가용 시 SileroVAD 자동 선택 옵션 | 선택 |

---

## 6. 비기능 요구사항

| ID | 요구사항 |
|---|---|
| NFR-01 | RTF < 1.0 (CPU base 모델), < 0.3 (GPU) |
| NFR-02 | 발화 종료 후 전사 결과 반환까지 < 2초 |
| NFR-03 | 메모리 사용량 < 2GB (CPU, base 모델 기준) |
| NFR-04 | 재연결 로직: 지수 백오프 (1→2→4→8→…→60초 최대) |
| NFR-05 | 로깅: 연결 상태, VAD 이벤트, 전사 결과, 에러 (Python logging 모듈) |
| NFR-06 | 스레드 안전성: asyncio ↔ threading ↔ sounddevice 간 thread-safe queue 사용 |

---

## 7. 모델 준비 파이프라인

### 7.1 전체 흐름

```
[1] Whisper 파인튜닝
    - 프레임워크: Hugging Face transformers + datasets
    - 베이스 모델: openai/whisper-small (또는 medium)
    - 데이터셋: 한국어 음성 데이터 (AIHub, KsponSpeech 등)
    - 출력: fine-tuned checkpoint 디렉터리

[2] CTranslate2 변환
    ct2-transformers-converter \
      --model <fine-tuned-checkpoint-dir> \
      --output_dir <faster-whisper-model-dir> \
      --quantization int8

[3] 추론
    faster-whisper WhisperModel(model_size_or_path="<faster-whisper-model-dir>")
```

### 7.2 PipelineConfig 모델 경로 설정

```python
# HuggingFace 허브 모델 (기본)
config = PipelineConfig(model="base")

# 로컬 Fine-tuned 모델
config = PipelineConfig(model="/path/to/faster-whisper-finetuned")
```

---

## 8. 모듈 구조 (Architecture)

### 8.1 데이터 흐름

```
Backend WebSocket Server
        │ (PCM binary / base64)
        ▼
RealtimeASRPipeline          ← WebSocket 클라이언트
        │
        ▼
    ASRCore
   ┌─────┴──────────────────────────────┐
   │  EnergyVAD / SileroVAD             │  발화 감지
   │  SpeechBuffer (state machine)      │  발화 세그먼트 수집
   │  WhisperTranscriber                │  전사 + confidence
   └─────────────────────────────────────┘
        │ TranscriptionResult
        ▼
on_transcription(result) callback
        │
        ▼
   [LLM Module]
```

### 8.2 클래스 책임

| 클래스 | 책임 | 변경 사항 |
|---|---|---|
| `PipelineConfig` | 중앙 설정 (모델 경로, VAD 임계값, WebSocket URI 등) | `model` 필드: 로컬 경로 허용 |
| `EnergyVAD` | RMS 에너지 기반 VAD | 변경 없음 |
| `SileroVAD` | torch 기반 고정밀 VAD (선택) | 변경 없음 |
| `SpeechBuffer` | 발화 구간 상태 머신 | 변경 없음 |
| `WhisperTranscriber` | faster-whisper 래퍼, 전사 실행 | **`confidence` 반환 추가** (`avg_logprob` → exp 정규화) |
| `ASRCore` | VAD + Buffer + Transcriber 오케스트레이션 | **콜백 시그니처 `TranscriptionResult`로 변경** |
| `RealtimeASRPipeline` | WebSocket 클라이언트, 오디오 수신 | 재연결 로직 강화 |
| `RealtimeASRServer` | WebSocket 서버 (테스트/독립 실행용) | 유지 |
| `MicrophoneASRTest` | 마이크 직접 테스트 | 유지 |

### 8.3 동시성 모델

```
asyncio event loop     → WebSocket I/O (RealtimeASRPipeline)
threading              → Whisper 추론 (백그라운드 스레드)
thread-safe queue      → asyncio ↔ threading 연결
sounddevice callback   → 독립 오디오 스레드 (MicrophoneASRTest 전용)
```

---

## 9. 기술 스택

### 추론 (Runtime)

| 패키지 | 버전 | 용도 |
|---|---|---|
| `faster-whisper` | >=1.0.0 | Whisper 추론 엔진 |
| `websockets` | >=12.0 | WebSocket 클라이언트/서버 |
| `numpy` | >=1.24.0 | 오디오 데이터 처리 |
| `sounddevice` | >=0.4.6 | 마이크 입력 (테스트용) |

### 파인튜닝 (개발 환경)

| 패키지 | 용도 |
|---|---|
| `transformers` | Whisper 파인튜닝 |
| `datasets` | 학습 데이터 로드 |
| `torch` / `torchaudio` | 학습 프레임워크 |
| `ctranslate2` | faster-whisper 형식 변환 |

### 선택 (Optional)

| 패키지 | 용도 |
|---|---|
| `torch` | GPU 가속, SileroVAD |

---

## 10. 테스트 시나리오

### 단위 테스트

- `WhisperTranscriber.transcribe()` → `TranscriptionResult` 형식 검증
- `confidence` 값이 [0, 1] 범위인지 확인
- 로컬 모델 경로 로드 정상 동작 확인

### 통합 테스트

- WebSocket mock 서버 → `RealtimeASRPipeline` → `ASRCore` → 콜백 텍스트/신뢰도 확인
- 연결 끊김 시 재연결 로직 동작 확인

### 성능 테스트

- 테스트 오디오 파일로 RTF 측정 스크립트 실행
- CPU / GPU 환경별 지연 시간 측정

### 회귀 테스트

- Fine-tuning 전/후 한국어 샘플 WER 비교
- 모델 교체 후 콜백 인터페이스 호환성 확인

---

## 11. 향후 고려 사항 (Future Considerations)

- **SileroVAD 기본 적용**: torch 의존성 해결 시 정확도 향상
- **화자 분리(Diarization)**: 다화자 환경 지원 필요 시 추가
- **스트리밍 전사**: 발화 종료 전 부분 결과 반환 (faster-whisper streaming API)
- **다국어 지원**: `language=None` 자동 감지 모드 활성화
