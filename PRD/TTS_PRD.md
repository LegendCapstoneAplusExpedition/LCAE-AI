# PRD: LLM 응답 텍스트 → 음성 변환 TTS 모듈

| 항목 | 내용 |
|---|---|
| 모듈명 | `TTS Pipeline` (pipeline/tts/tts_pipeline.py) |
| 작성일 | 2026-04-10 |
| 버전 | v3.0 |
| 작성자 | sucheoli |

### 변경 이력

| 버전 | 날짜 | 변경 내용 |
|---|---|---|
| v1.0 | 2026-04-10 | 최초 작성 (Coqui TTS VITS 기반) |
| v2.0 | 2026-05-12 | TTS 엔진 Coqui → **Kokoro-82M** 교체, TTSConfig 필드 개편, LLM 파이프라인 연동 완료 반영 |
| v3.0 | 2026-05-12 | TTS 엔진 Kokoro-82M → **gTTS (Google TTS)** 교체, TTSConfig 필드 개편 (lang_code/tld/slow), 오프라인 의존성 제거 |

---

## 1. 개요

LLM이 생성한 텍스트 응답을 **gTTS (Google Text-to-Speech)** API로 한국어 음성으로 합성하고, 합성된 PCM 오디오를 WebSocket으로 백엔드에 전송하는 TTS(Text-to-Speech) 파이프라인 모듈. STT → LLM → **TTS** 파이프라인의 마지막 구간을 담당한다.

---

## 2. 배경 및 목표

### 배경

v3.0에서 Kokoro-82M을 **gTTS**로 교체하였다. gTTS는 Google의 TTS API를 사용하여 별도 모델 파일 다운로드 없이 고품질 음성을 합성한다. 경량 의존성(gtts, pydub)으로 환경 구성이 간단하고, 인터넷 연결만 있으면 즉시 사용 가능하다.

- gTTS 출력 포맷(MP3)은 pydub을 통해 PCM 16-bit Mono로 변환하여 기존 인터페이스(SynthesisResult)를 유지한다.
- ffmpeg가 MP3 디코딩에 필요하다.

### 목표

- LLM 응답 텍스트를 gTTS(Google TTS)로 한국어 음성 합성
- 합성된 PCM 오디오(16-bit, Mono, 24000Hz)를 WebSocket으로 백엔드에 전송
- 엔진 교체 인터페이스 제공 (추후 외부 API 전환 시 TTSCore 무변경)

### 성공 기준

| 지표 | 목표값 |
|---|---|
| 합성 지연 (텍스트 입력 후 오디오 출력까지) | < 5초 (네트워크 포함) |
| 엔진 교체 | 외부 API 구현체 주입 시 TTSCore 코드 변경 없음 |

---

## 3. 범위 (Scope)

### In Scope

- LLM 콜백으로 텍스트 수신
- gTTS(`gtts`) + pydub으로 MP3 합성 후 PCM 변환
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
- 오프라인 TTS (인터넷 연결 필수)

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
    sample_rate: int   # 합성 샘플레이트 (기본: 24000)
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
| FR-02 | gTTS(`gtts`)로 한국어 음성 MP3 합성 | 필수 |
| FR-03 | pydub으로 MP3 → PCM 16-bit Mono 변환 | 필수 |
| FR-04 | SynthesisResult (audio bytes + duration + sample_rate) on_synthesis 콜백 전달 | 필수 |
| FR-05 | WebSocket 클라이언트로 합성된 PCM을 백엔드에 전송 | 필수 |
| FR-06 | 로컬 스피커 재생 테스트 (SpeakerTTSTest, sounddevice) | 권장 |
| FR-07 | TTSConfig를 통한 lang_code / tld / slow / WebSocket URI 설정 | 필수 |
| FR-08 | BaseTTSSynthesizer 인터페이스로 엔진 교체 지원 | 필수 |
| FR-09 | WebSocket 연결 끊김 시 지수 백오프 재연결 | 필수 |
| FR-10 | 합성 히스토리 조회 (SpeakerTTSTest.get_synthesis_history()) | 권장 |

---

## 6. 비기능 요구사항

| ID | 요구사항 |
|---|---|
| NFR-01 | 합성 지연 < 5초 (Google TTS 네트워크 왕복 포함) |
| NFR-02 | 인터넷 연결 필수 (Google TTS API 사용) |
| NFR-03 | ffmpeg 설치 필수 (pydub MP3 디코딩) |
| NFR-04 | 재연결 로직: 지수 백오프 (1→2→4→8→…→60초 최대) |
| NFR-05 | 스레드 안전성: asyncio(WebSocket) ↔ threading(합성) 간 thread-safe asyncio.Queue 사용 |
| NFR-06 | 엔진 인터페이스 추상화: 다른 BaseTTSSynthesizer 구현체 주입 시 TTSCore 코드 변경 없음 |

---

## 7. 환경 준비

### 7.1 패키지 설치

```bash
pip install gtts pydub sounddevice
```

### 7.2 ffmpeg 설치 (MP3 디코딩 필수)

```bash
# Windows: https://ffmpeg.org/download.html → PATH 등록
# Ubuntu
sudo apt install ffmpeg
# macOS
brew install ffmpeg
```

### 7.3 언어 코드 (lang_code) — ISO 639-1

| lang_code | 언어 |
|---|---|
| `ko` | 한국어 (기본값) |
| `en` | 영어 |
| `ja` | 일본어 |
| `zh` | 중국어 |

### 7.4 tld — 발음 변형 제어

| tld | 발음 |
|---|---|
| `com` | 기본 (기본값) |
| `co.uk` | 영국식 영어 |
| `com.au` | 호주식 영어 |

---

## 8. 모듈 구조 (Architecture)

### 8.1 데이터 흐름

```
[LLM Module] or [STT 직결]
        │ text: str
        ▼
    TTSCore
        │
  GTTSSynthesizer.synthesize(text)   ← 별도 데몬 스레드
        │  gTTS → MP3(BytesIO) → pydub → PCM 16-bit Mono
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
| `TTSConfig` | 중앙 설정 dataclass — lang_code, tld, slow, ws_uri, sample_rate |
| `BaseTTSSynthesizer` | 엔진 교체 추상 인터페이스 — `synthesize(text) -> SynthesisResult` |
| `GTTSSynthesizer` | gTTS 구현체 — Google TTS API 호출, MP3→PCM 변환 (pydub) |
| `TTSCore` | 오케스트레이터 — 합성 데몬 스레드 관리, on_synthesis 콜백 호출 |
| `RealtimeTTSPipeline` | WebSocket 클라이언트 — 합성 결과 전송, 지수 백오프 재연결 |
| `SpeakerTTSTest` | 로컬 스피커 테스트 — 키보드 입력 → 합성 → sounddevice 재생, 히스토리 관리 |

### 8.3 동시성 모델

```
asyncio event loop     → WebSocket I/O (RealtimeTTSPipeline.run)
threading (daemon)     → gTTS 합성 + pydub 변환 (TTSCore._run)
asyncio.Queue          → threading → asyncio 브리지 (run_coroutine_threadsafe)
sounddevice            → 독립 오디오 스레드 (SpeakerTTSTest 전용)
```

---

## 9. 기술 스택

| 패키지 | 버전 | 용도 |
|---|---|---|
| `gtts` | >=2.5.0 | Google TTS API 클라이언트 (MP3 합성) |
| `pydub` | >=0.25.1 | MP3 → PCM 변환 (ffmpeg 의존) |
| `websockets` | >=12.0 | WebSocket 클라이언트 |
| `numpy` | >=1.24.0 | PCM float32 → int16 변환 |
| `sounddevice` | >=0.4.6 | 로컬 스피커 재생 (SpeakerTTSTest) |
| `python-dotenv` | >=1.2.2 | .env 파일 로드 |

### 시스템 의존성

| 도구 | 용도 |
|---|---|
| `ffmpeg` | pydub MP3 디코딩 필수 |

---

## 10. 설정 (TTSConfig 필드 및 환경변수)

| 필드 | 환경변수 | 기본값 | 설명 |
|---|---|---|---|
| `ws_uri` | `TTS_WS_URI` | `ws://localhost:8080/tts` | 백엔드 WebSocket URI |
| `lang_code` | `TTS_LANG_CODE` | `ko` | ISO 언어 코드 (ko=한국어) |
| `tld` | `TTS_TLD` | `com` | Google TTS 도메인 (발음 변형 제어) |
| `slow` | `TTS_SLOW` | `false` | 느린 속도 합성 여부 |
| `sample_rate` | `TTS_SAMPLE_RATE` | `24000` | 출력 PCM 샘플레이트 (pydub 리샘플링) |

---

## 11. 테스트 시나리오

### 단위 테스트

- `GTTSSynthesizer.synthesize()` → `SynthesisResult` 형식 검증
- `audio` 필드가 비어있지 않고 `duration > 0` 확인
- `sample_rate == 24000` 확인
- PCM 16-bit 변환 정합성 검증

### 통합 테스트

- `TTSCore.synthesize("텍스트")` → on_synthesis 콜백 호출 확인
- WebSocket mock 서버 → `RealtimeTTSPipeline` → PCM 수신 확인
- 연결 끊김 시 재연결 로직 동작 확인

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

# 언어/도메인 변경
python -m pipeline.tts.tts_pipeline --mode speaker --lang en --tld co.uk

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
- **오프라인 대체**: 인터넷 미연결 환경에서는 로컬 모델(Kokoro, Coqui 등) BaseTTSSynthesizer 구현체로 교체
- **스트리밍 합성**: 텍스트 청크 단위 실시간 합성 (첫 음절 지연 단축)
