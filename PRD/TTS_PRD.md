# PRD: LLM 응답 텍스트 → 음성 변환 TTS 모듈

| 항목 | 내용 |
|---|---|
| 모듈명 | `TTS Pipeline` (pipeline/tts/tts_pipeline.py) |
| 작성일 | 2026-04-10 |
| 버전 | v2.0 |
| 작성자 | sucheoli |

### 변경 이력

| 버전 | 날짜 | 변경 내용 |
|---|---|---|
| v1.0 | 2026-04-10 | 최초 작성 (Coqui TTS VITS 기반) |
| v2.0 | 2026-05-12 | TTS 엔진 Coqui → **Kokoro-82M** 교체, TTSConfig 필드 개편, LLM 파이프라인 연동 완료 반영 |

---

## 1. 개요

LLM이 생성한 텍스트 응답을 **Kokoro-82M** 모델로 한국어 음성으로 합성하고, 합성된 PCM 오디오를 WebSocket으로 백엔드에 전송하는 TTS(Text-to-Speech) 파이프라인 모듈. STT → LLM → **TTS** 파이프라인의 마지막 구간을 담당한다.

---

## 2. 배경 및 목표

### 배경

2026 캡스톤 시스템은 LLM이 생성한 텍스트 응답을 사용자에게 음성으로 전달하는 기능을 필요로 한다. `ASR_PRD.md`의 STT 모듈과 대칭되는 구조로 설계하여 파이프라인 통합 및 유지보수를 단순화한다.

v2.0에서 Coqui TTS(VITS)를 **Kokoro-82M**으로 교체하였다. Kokoro는 경량 고품질 모델로 음성 자연스러움이 향상되었으며, `lang_code` / `voice` / `speed` 파라미터로 세밀한 제어가 가능하다.

### 목표

- LLM 응답 텍스트를 Kokoro-82M(한국어)으로 실시간 합성
- 합성된 PCM 오디오(16-bit, Mono, 24000Hz)를 WebSocket으로 백엔드에 전송
- 엔진 교체 인터페이스 제공 (추후 외부 API 전환 시 TTSCore 무변경)
- 실시간 처리 성능 확보 (RTF < 1.0 on CPU)

### 성공 기준

| 지표 | 목표값 |
|---|---|
| RTF (Real-Time Factor) | < 1.0 (CPU), < 0.3 (GPU) |
| 합성 지연 (텍스트 입력 후 오디오 출력까지) | < 3초 |
| 메모리 사용량 | < 2GB (CPU) |
| 엔진 교체 | 외부 API 구현체 주입 시 TTSCore 코드 변경 없음 |

---

## 3. 범위 (Scope)

### In Scope

- LLM 콜백으로 텍스트 수신
- Kokoro-82M(`kokoro` + `misaki[ko]`)으로 한국어 음성 합성
- 합성 결과(SynthesisResult) on_synthesis 콜백 전달
- WebSocket 클라이언트로 백엔드에 PCM 전송
- 로컬 스피커 재생 테스트 (SpeakerTTSTest)
- 엔진 교체 인터페이스 (BaseTTSSynthesizer)
- 재연결 로직 (지수 백오프)

### Out of Scope

- LLM 텍스트 생성 (LLM_PRD.md 참조)
- STT (ASR_PRD.md 참조)
- 화자 분리 (Speaker Diarization)
- 프론트엔드 UI
- TTS 모델 파인튜닝 코드

---

## 4. 시스템 인터페이스 명세

### 4.1 텍스트 입력 — LLM 콜백

```python
# TTSCore.synthesize()를 직접 호출하거나,
# RealtimeTTSPipeline.synthesize()를 통해 WebSocket 전송까지 자동 처리합니다.

tts = TTSCore(config, on_synthesis=handle)
tts.synthesize("LLM이 생성한 응답 텍스트")
```

### 4.2 합성 결과 출력 — SynthesisResult (콜백)

```python
from typing import TypedDict, Callable

class SynthesisResult(TypedDict):
    audio: bytes       # PCM 16-bit, Mono raw bytes
    sample_rate: int   # 합성 샘플레이트 (Kokoro 기본: 24000)
    duration: float    # 합성 음성 길이 (초)
    language: str      # 언어 코드 (예: "ko")

on_synthesis: Callable[[SynthesisResult], None]
```

### 4.3 WebSocket 오디오 송신

| 항목 | 명세 |
|---|---|
| 프로토콜 | WebSocket (ws:// 또는 wss://) |
| 프레임 포맷 | Binary (raw PCM) |
| 오디오 포맷 | PCM 16-bit, Mono, **24000Hz** |
| 연결 URI | `ws://<host>:<port>/tts` (TTSConfig.ws_uri로 설정) |

#### WebSocket 이벤트 처리

```
CONNECT   → 연결 수립, 합성 대기 루프 시작
합성 완료  → asyncio 큐에 PCM 푸시 → ws.send(audio_bytes)
CLOSE     → 정상 종료, 리소스 해제
ERROR     → 로깅 후 재연결 시도 (지수 백오프: 1s, 2s, 4s, 8s, 최대 60s)
```

### 4.4 파이프라인 연동 (main.py)

```python
from pipeline.stt import PipelineConfig, TranscriptionResult
from pipeline.tts import TTSConfig, TTSCore
from pipeline.llm.chain.graph import app as llm_app
from pipeline.llm.chain.state import AgentState
from langchain_core.messages import HumanMessage

tts = TTSCore(TTSConfig(), on_synthesis=on_synthesis)

def on_transcription(result: TranscriptionResult) -> None:
    # LLM 연결 시 아래 주석 해제
    # state = AgentState(messages=[HumanMessage(content=result["text"])], ...)
    # llm_result = llm_app.invoke(state)
    # tts.synthesize(llm_result["messages"][-1].content)

    tts.synthesize(result["text"])  # 현재: STT → TTS 직결 (LLM 미연결)
```

---

## 5. 기능 요구사항

| ID | 요구사항 | 우선순위 |
|---|---|---|
| FR-01 | LLM(또는 STT) 텍스트를 synthesize() 메서드로 수신 | 필수 |
| FR-02 | Kokoro-82M(`kokoro` + `misaki[ko]`)으로 한국어 음성 합성 | 필수 |
| FR-03 | SynthesisResult (audio bytes + duration + sample_rate) on_synthesis 콜백 전달 | 필수 |
| FR-04 | WebSocket 클라이언트로 합성된 PCM을 백엔드에 전송 | 필수 |
| FR-05 | 로컬 스피커 재생 테스트 (SpeakerTTSTest, sounddevice) | 권장 |
| FR-06 | TTSConfig를 통한 lang_code / voice / speed / device / WebSocket URI 설정 | 필수 |
| FR-07 | BaseTTSSynthesizer 인터페이스로 엔진 교체 지원 | 필수 |
| FR-08 | WebSocket 연결 끊김 시 지수 백오프 재연결 | 필수 |
| FR-09 | 합성 히스토리 조회 (SpeakerTTSTest.get_synthesis_history()) | 권장 |
| FR-10 | PCM float32 → int16 변환 및 클리핑 처리 (`(wav * 32767).clip(-32768, 32767)`) | 필수 |

---

## 6. 비기능 요구사항

| ID | 요구사항 |
|---|---|
| NFR-01 | RTF < 1.0 (CPU), < 0.3 (GPU) |
| NFR-02 | 텍스트 입력 후 최초 오디오 출력까지 < 3초 |
| NFR-03 | 메모리 사용량 < 2GB (CPU, Kokoro-82M 기준) |
| NFR-04 | 재연결 로직: 지수 백오프 (1→2→4→8→…→60초 최대) |
| NFR-05 | 스레드 안전성: asyncio(WebSocket) ↔ threading(합성) 간 thread-safe asyncio.Queue 사용 |
| NFR-06 | 엔진 인터페이스 추상화: 다른 BaseTTSSynthesizer 구현체 주입 시 TTSCore 코드 변경 없음 |
| NFR-07 | Kokoro 모델은 첫 합성 시 HuggingFace에서 자동 다운로드, 이후 캐시 사용 |

---

## 7. 모델 준비 파이프라인

### 7.1 Kokoro-82M (자동 다운로드)

```bash
# 최초 실행 시 Kokoro 모델이 HuggingFace에서 자동으로 다운로드됩니다.
python -m pipeline.tts.tts_pipeline --mode speaker
```

### 7.2 한국어 음성(Voice) 목록

| Voice ID | 성별 | 설명 |
|---|---|---|
| `kf_bella` | 여성 | 기본값, 자연스러운 한국어 여성 음성 |
| `kf_heart` | 여성 | 감성적인 여성 음성 |
| `km_blade` | 남성 | 남성 음성 |
| `km_echo` | 남성 | 남성 음성 |

### 7.3 언어 코드 (lang_code)

| lang_code | 언어 |
|---|---|
| `k` | 한국어 (기본값) |
| `a` | 미국 영어 |
| `b` | 영국 영어 |

---

## 8. 모듈 구조 (Architecture)

### 8.1 데이터 흐름

```
[LLM Module] or [STT 직결]
        │ text: str
        ▼
    TTSCore
        │
  KokoroTTSSynthesizer.synthesize(text)   ← 별도 데몬 스레드
        │ SynthesisResult(audio, sample_rate, duration, language)
        ▼
on_synthesis(result) callback
        │
        ├── RealtimeTTSPipeline ──→ asyncio Queue ──→ WebSocket ──→ Backend
        │
        └── SpeakerTTSTest ──→ sounddevice.play()  [로컬 테스트용]
```

### 8.2 클래스 책임

| 클래스 | 책임 |
|---|---|
| `SynthesisResult` | TypedDict — audio(bytes), sample_rate, duration, language |
| `TTSConfig` | 중앙 설정 dataclass — lang_code, voice, speed, device, ws_uri, sample_rate |
| `BaseTTSSynthesizer` | 엔진 교체 추상 인터페이스 — `synthesize(text) -> SynthesisResult` |
| `KokoroTTSSynthesizer` | Kokoro-82M 구현체 — `KPipeline` 래퍼, PCM 변환 처리 |
| `TTSCore` | 오케스트레이터 — 합성 데몬 스레드 관리, on_synthesis 콜백 호출 |
| `RealtimeTTSPipeline` | WebSocket 클라이언트 — 합성 결과 전송, 지수 백오프 재연결 |
| `SpeakerTTSTest` | 로컬 스피커 테스트 — 키보드 입력 → 합성 → sounddevice 재생, 히스토리 관리 |

### 8.3 동시성 모델

```
asyncio event loop     → WebSocket I/O (RealtimeTTSPipeline.run)
threading (daemon)     → Kokoro 합성 (TTSCore._run)
asyncio.Queue          → threading → asyncio 브리지 (run_coroutine_threadsafe)
sounddevice            → 독립 오디오 스레드 (SpeakerTTSTest 전용)
```

---

## 9. 기술 스택

| 패키지 | 버전 | 용도 |
|---|---|---|
| `kokoro` | >=0.9.4 | Kokoro-82M TTS 추론 엔진 |
| `misaki[ko]` | ― | 한국어 음소 변환기 (Kokoro 의존) |
| `websockets` | >=12.0 | WebSocket 클라이언트 |
| `numpy` | >=1.24.0 | 오디오 데이터 변환 (float32 → int16) |
| `sounddevice` | >=0.4.6 | 로컬 스피커 재생 (SpeakerTTSTest) |
| `python-dotenv` | >=1.2.2 | .env 파일 로드 |

### 선택 (Optional)

| 패키지 | 용도 |
|---|---|
| `torch>=2.3.1` | GPU 가속 합성 |

---

## 10. 설정 (TTSConfig 필드 및 환경변수)

| 필드 | 환경변수 | 기본값 | 설명 |
|---|---|---|---|
| `ws_uri` | `TTS_WS_URI` | `ws://localhost:8080/tts` | 백엔드 WebSocket URI |
| `lang_code` | `TTS_LANG_CODE` | `k` | Kokoro 언어 코드 (k=한국어) |
| `voice` | `TTS_VOICE` | `kf_bella` | Kokoro 음성 ID |
| `speed` | `TTS_SPEED` | `1.0` | 합성 속도 배율 |
| `device` | `TTS_DEVICE` | `auto` | auto / cpu / cuda |
| `sample_rate` | `TTS_SAMPLE_RATE` | `24000` | 합성 샘플레이트 (Kokoro 기본) |

---

## 11. 테스트 시나리오

### 단위 테스트

- `KokoroTTSSynthesizer.synthesize()` → `SynthesisResult` 형식 검증
- `audio` 필드가 비어있지 않고 `duration > 0` 확인
- PCM 16-bit 변환 정합성 ([-32768, 32767] 범위)
- `sample_rate == 24000` 확인

### 통합 테스트

- `TTSCore.synthesize("텍스트")` → on_synthesis 콜백 호출 확인
- WebSocket mock 서버 → `RealtimeTTSPipeline` → PCM 수신 확인
- 연결 끊김 시 재연결 로직 동작 확인

### 성능 테스트

- 텍스트 길이별 합성 시간 측정 (10자 / 50자 / 200자)
- CPU / GPU 환경별 RTF 측정

### 엔진 교체 테스트

```python
class DummySynthesizer(BaseTTSSynthesizer):
    def synthesize(self, text: str) -> SynthesisResult:
        audio = b"\x00" * 48000  # 1초 묵음 (24000Hz × 2 bytes)
        return SynthesisResult(audio=audio, sample_rate=24000, duration=1.0, language="ko")

core = TTSCore(config, on_synthesis=handle, synthesizer=DummySynthesizer())
core.synthesize("테스트")  # TTSCore 코드 변경 없이 엔진 교체 확인
```

---

## 12. 실행 방법

```bash
# 로컬 스피커 테스트 (키보드 입력 → 합성 → 재생)
python -m pipeline.tts.tts_pipeline --mode speaker

# 음성 변경
python -m pipeline.tts.tts_pipeline --mode speaker --voice km_blade

# 사용 가능한 스피커 목록 확인
python -m pipeline.tts.tts_pipeline --list-devices

# WebSocket 클라이언트 모드 (합성 결과를 백엔드에 전송)
python -m pipeline.tts.tts_pipeline --mode client --ws-uri ws://localhost:8080/tts

# 통합 파이프라인 (STT → TTS)
python main.py --mode mic
```

---

## 13. 향후 고려 사항 (Future Considerations)

- **LLM 연동 완성**: `pipeline/llm/` 구현 완료 — `on_transcription → llm_app.invoke → tts.synthesize` 연결 시 `main.py` 주석 해제
- **외부 API 구현체 추가**: `OpenAITTSSynthesizer`, `ClovaTTSSynthesizer` — `BaseTTSSynthesizer` 상속으로 TTSCore 무변경 교체
- **스트리밍 합성**: 텍스트 청크 단위 실시간 합성 (첫 음절 지연 단축)
- **화자(voice) 파인튜닝**: 커스텀 화자 음성으로 Kokoro 파인튜닝
- **음성 감정 제어**: 속도/피치 파라미터는 TTSConfig.speed로 이미 지원, 감정 태그 추가 검토
