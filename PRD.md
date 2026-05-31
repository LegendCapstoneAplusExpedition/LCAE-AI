# LCAE-AI Product Requirements Document

**프로젝트명**: LCAE-AI (Live Caption & AI Emcee)  
**목적**: 실시간 라디오 방송을 위한 AI MC 파이프라인  
**컨셉**: "드라이빙 멘토링" 라디오 프로그램에서 멘토 발화를 실시간으로 받아 AI MC가 적절한 멘트를 생성  
**작성일**: 2026-05-31  

---

## 1. 전체 아키텍처 개요

```
Audio Input (마이크 / WebSocket)
         │
         ▼
  [STT] Whisper ASR + Silero VAD
         │  TranscriptionResult {text, confidence}
         ▼
  [Buffer] ListenList (JSONL, max 50)
         │  {time, text, conf}
         ▼
  [LLM] LangGraph 파이프라인
         │  ├─ preprocess  (regex 필러 제거)
         │  ├─ search      (ChromaDB RAG)
         │  ├─ analyze     (Ollama/Groq → JSON)
         │  └─ output      (AIMessage)
         │  mc_script
         ▼
  [TTS] gTTS → pydub PCM 변환
         │  SynthesisResult {audio(PCM), sample_rate, duration}
         ▼
  Audio Output (sounddevice 스피커)
```

---

## 2. Stage 1: STT (Speech-to-Text)

### 2.1 역할
멘토 발화 오디오를 텍스트로 변환하고 신뢰도 점수를 반환한다.

### 2.2 핵심 컴포넌트

| 컴포넌트 | 구현 | 역할 |
|---|---|---|
| VAD | Silero VAD (fallback: Energy VAD) | 발화 구간 감지 |
| Transcriber | faster-whisper | 음성→텍스트 |
| Buffer | SpeechBuffer | 침묵 기반 발화 단위 분리 |

### 2.3 설정 파라미터 (`.env`)

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `ASR_MODEL` | `base` | Whisper 모델 크기 |
| `AUDIO_SAMPLE_RATE` | `16000` | 입력 샘플레이트 (Hz) |
| `AUDIO_CHUNK_DURATION_MS` | `30` | 청크 단위 (ms) |
| `VAD_THRESHOLD` | `0.02` | VAD 에너지 임계값 |
| `VAD_SILENCE_DURATION_S` | `0.8` | 침묵 판단 기준 (초) |
| `VAD_MIN_SPEECH_DURATION_S` | `0.3` | 최소 발화 길이 |
| `VAD_MAX_BUFFER_DURATION_S` | `30.0` | 최대 버퍼 길이 |
| `ASR_LANGUAGE` | `ko` | 인식 언어 |

### 2.4 출력 형식

```python
@dataclass
class TranscriptionResult:
    text: str            # 인식된 텍스트
    confidence: float    # [0, 1] 신뢰도 (segment log-prob 가중평균)
```

### 2.5 실행 모드

| 모드 | 설명 |
|---|---|
| `mic` | 직접 마이크 입력 (로컬 단독 실행) |
| `server` | WebSocket 서버로 동작 (백엔드 수신) |
| `client` | WebSocket 클라이언트로 연결 (재접속: 지수 백오프 1→60s) |

### 2.6 핵심 파일
- [pipeline/stt/asr_pipeline.py](pipeline/stt/asr_pipeline.py) — ASRCore, SpeechBuffer, VAD (612 lines)

---

## 3. Stage 2: ListenList Buffer

### 3.1 역할
STT 결과를 임시 저장하여 LLM이 이전 발화 컨텍스트를 참조할 수 있도록 한다.  
동시에 현재 처리 중인 항목과 미처리 항목을 분리 관리한다.

### 3.2 저장 형식 (JSONL)

```json
{"time": "2026-05-31 14:23:01", "text": "오늘 코드 리뷰를 해봤는데요", "conf": 0.9321}
```

### 3.3 동작 규칙

| 규칙 | 내용 |
|---|---|
| 최대 항목 | 50개 (초과 시 가장 오래된 항목 자동 제거) |
| 스레드 안전 | `threading.Lock()` 사용 |
| 컨텍스트 참조 | LLM 호출 시 최근 10개 항목을 prior context로 전달 |
| 항목 삭제 | LLM 처리 완료 후 `remove_entry(time_ms)` 호출 |

### 3.4 핵심 파일
- [pipeline/listenlist/listen_list.py](pipeline/listenlist/listen_list.py) — ListenList 클래스 (80 lines)
- [pipeline/listenlist/transcriptions.jsonl](pipeline/listenlist/transcriptions.jsonl) — 런타임 버퍼 파일

---

## 4. Stage 3: LLM (LangGraph 파이프라인)

### 4.1 역할
멘토 발화를 분석하고 방송 상황에 맞는 AI MC 멘트를 생성한다.

### 4.2 상태 구조 (AgentState)

```python
class AgentState(TypedDict):
    messages: List[BaseMessage]      # 대화 히스토리
    question_queue: List[Dict]       # 청취자 Q&A 목록
    broadcast_topics: List[str]      # 방송 사전 주제 목록
    current_topic: str               # LLM이 추출한 현재 주제
    context_summary: str             # 누적 요약 (3문장 이내)
    retrieved_info: List[str]        # RAG 검색 결과
    streaming_stage: str             # Main | QnA | Outro
    intent: str                      # 분류된 발화 의도
    cleaned_text: str                # 필러 제거 후 텍스트
    mc_script: str                   # 최종 MC 출력 멘트
    silence_duration: float          # STT로부터 전달된 침묵 길이
```

### 4.3 방송 상태 머신 (streaming_stage)

```
Main ──(intent=질문요청)──▶ QnA
QnA ──(intent≠질문요청,마무리)──▶ Main
* ──(intent=마무리)──▶ Outro (잠금, 이후 출력 없음)
```

### 4.4 LangGraph 노드 파이프라인

```
START → preprocess_node → knowledge_search_node → analyze_write_node → decision_node → (speak) output_node → END
                                                                                      → (wait) END
```

#### Node 1: preprocess_node
- **역할**: 한국어 필러 제거 (정규식 기반)
- **처리 시간**: ~0ms
- **입력**: `messages[-1].content`
- **출력**: `cleaned_text`
- **예시 제거 패턴**: 아~, 그니까, 뭐랄까, 어..., 이제, 그래서 등

#### Node 2: knowledge_search_node
- **역할**: 벡터 DB에서 관련 지식 검색 (RAG)
- **처리 시간**: ~50ms
- **건너뜀 조건**: DB 미초기화 또는 5단어 미만 발화
- **설정**: ChromaDB + OllamaEmbeddings(nomic-embed-text), k=2
- **출력**: `retrieved_info` (관련 텍스트 청크 목록)

#### Node 3: analyze_write_node
- **역할**: 단일 LLM 호출로 의도 분류 + MC 스크립트 생성
- **처리 시간**: ~12s (Ollama) / ~1-2s (Groq)
- **건너뜀 조건**: 유효 문자(한글+영숫자) 5자 미만 → `intent=대기`, 빈 스크립트 반환
- **구조화 출력** (Pydantic):

```python
class AnalyzeAndWriteResult(BaseModel):
    topic: str      # 2-5 단어 명사구 (현재 주제)
    summary: str    # 누적 요약 갱신 (3줄 이내, 복사 금지)
    intent: str     # 설명 | 질문 | 질문요청 | 정리요청 | 마무리 | 대기
    mc_script: str  # MC 출력 멘트 (발화 안 할 경우 빈 문자열)
```

- **intent 분류 기준**:

| intent | 조건 | MC 출력 여부 |
|---|---|---|
| `설명` | 멘토가 내용 설명 중 | X (대기) |
| `질문` | 멘토가 청취자에게 질문 | X (대기) |
| `질문요청` | 청취자 Q&A 시작 요청 | O |
| `정리요청` | 내용 정리 요청 | O |
| `마무리` | 방송 마무리 | O |
| `대기` | 짧거나 의미없는 발화 | X |

#### Node 4: decision_node
- **역할**: MC 출력 여부 결정 (라우팅 함수)
- **"speak" 조건**: `intent in ("질문요청", "정리요청", "마무리")`
- **Outro 잠금**: 이미 Outro 상태이면 항상 "wait"

#### Node 5: output_node
- **역할**: `mc_script` → `AIMessage` 변환 후 messages에 추가
- **조건**: decision_node가 "speak" 반환 시에만 실행

### 4.5 LLM 프로바이더

| 프로바이더 | 모델 | 응답속도 | 설정 |
|---|---|---|---|
| Ollama (기본) | `driving-mentor` (GGUF q4km) | ~12s | 로컬, 인터넷 불필요 |
| Groq (대안) | `llama-3.1-8b-instant` | ~1-2s | `GROQ_API_KEY` 필요 |

**전환 방법**: `.env`에서 `LLM_PROVIDER=groq` 설정

### 4.6 핵심 파일

| 파일 | 역할 |
|---|---|
| [pipeline/llm/chain/graph.py](pipeline/llm/chain/graph.py) | LangGraph 워크플로우 정의 |
| [pipeline/llm/chain/state.py](pipeline/llm/chain/state.py) | AgentState + Pydantic 모델 |
| [pipeline/llm/chain/nodes.py](pipeline/llm/chain/nodes.py) | 노드 구현체 |
| [pipeline/llm/prompts/persona.py](pipeline/llm/prompts/persona.py) | 시스템 프롬프트 (MC 페르소나) |
| [pipeline/llm/utils/text_cleaner.py](pipeline/llm/utils/text_cleaner.py) | 필러 제거 유틸 |
| [pipeline/llm/utils/llm.py](pipeline/llm/utils/llm.py) | LLM 프로바이더 전환 |
| [pipeline/llm/utils/ingest_data.py](pipeline/llm/utils/ingest_data.py) | RAG 데이터 적재 |

---

## 5. Stage 4: TTS (Text-to-Speech)

### 5.1 역할
LLM이 생성한 MC 스크립트를 음성으로 합성한다.

### 5.2 설정 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `TTS_LANG_CODE` | `ko` | 합성 언어 |
| `TTS_SAMPLE_RATE` | `24000` | 출력 샘플레이트 (Hz) |
| `TTS_SLOW` | `false` | 발화 속도 |
| `TTS_TLD` | `com` | Google 도메인 변형 |

### 5.3 처리 흐름

```
mc_script (str)
    → gTTS → MP3 (in-memory)
    → pydub decode → PCM
    → 리샘플링 (24kHz, mono 16-bit)
    → SynthesisResult {audio: bytes, sample_rate: 24000, duration: float}
    → sounddevice.play() (float32 정규화)
```

### 5.4 의존성

- `gtts`: Google TTS API (인터넷 필요)
- `pydub`: MP3→PCM 변환
- `ffmpeg`: pydub 백엔드 (시스템 설치 필요)
- `sounddevice`: 오디오 재생

### 5.5 플러그인 구조
`BaseTTSSynthesizer` 추상 클래스 기반으로 교체 가능:
- 현재: `GTTSSynthesizer` (Google TTS)
- 교체 후보: Naver Clova, OpenAI TTS, Coqui

### 5.6 핵심 파일
- [pipeline/tts/tts_pipeline.py](pipeline/tts/tts_pipeline.py) — TTSCore, GTTSSynthesizer

---

## 6. 오케스트레이션 (main.py)

### 6.1 파이프라인 조립 흐름

```python
on_transcription(result: TranscriptionResult):
    1. ListenList.append(result.text, result.confidence)
    2. prior_context = ListenList.read_all()[-10:]  # 최근 10개
    3. messages = [context_messages..., HumanMessage(result.text)]
    4. state = mentor_setup(topics)
    5. state["messages"] = messages
    6. result = llm_app.invoke(state)
    7. if AIMessage in result["messages"]:
          tts.synthesize(message.content)
    8. ListenList.remove_entry(result.time)
```

### 6.2 CLI 인수

```
python main.py [--mode {mic|server|client}]
               [--model {tiny|base|small|medium|large-v3}]
               [--language {ko|en|...}]
               [--device {auto|cpu|cuda}]
               [--mic-device ID]
               [--host HOST] [--port PORT]
               [--ws-uri URI]
               [--list-devices]
```

### 6.3 핵심 파일
- [main.py](main.py) — 전체 파이프라인 진입점 (157 lines)

---

## 7. 모델 학습 & 관리

### 7.1 Whisper ASR 파인튜닝

| 항목 | 내용 |
|---|---|
| 스크립트 | [pipeline/stt/asr_Finetuning.py](pipeline/stt/asr_Finetuning.py) |
| 베이스 모델 | openai/whisper-* |
| 데이터 | Common Voice KO, KsponSpeech, 커스텀 |
| 출력 | faster-whisper 포맷 (ct2-transformers-converter) |

### 7.2 LLM LoRA 파인튜닝

| 항목 | 내용 |
|---|---|
| 스크립트 | [scripts/train_sft.py](scripts/train_sft.py) |
| 베이스 모델 | Meta-Llama-3.1-8B-Instruct |
| 방법 | LoRA + QLoRA (4-bit 양자화) |
| 하드웨어 | RTX 2060 6GB (FP16, batch=1, grad_accum=8) |
| 학습 데이터 | [data/train_v2.jsonl](data/train_v2.jsonl) (채팅 포맷) |
| 출력 | `adapters_v2/` (LoRA 가중치) |

### 7.3 어댑터 병합

| 항목 | 내용 |
|---|---|
| 스크립트 | [scripts/merge_adapter.py](scripts/merge_adapter.py) |
| 과정 | base model + LoRA adapter → merge_and_unload() |
| 메모리 요구 | ~15-20GB RAM (CPU FP16) |
| 출력 | `merged_model/` (독립 실행 가능) |

### 7.4 Ollama 모델 배포

| 항목 | 내용 |
|---|---|
| 모델 파일 | `driving-mentor-q4km.gguf` (Q4_K_M 양자화) |
| 설정 | [Modelfile](Modelfile) |
| 파라미터 | temperature=0.7, top_p=0.9, num_ctx=1024 |
| 등록 | `ollama create driving-mentor -f Modelfile` |

---

## 8. RAG 지식 베이스

### 8.1 데이터 적재

```bash
# ./data/*.txt 파일을 ChromaDB에 적재
python -m pipeline.llm.utils.ingest_data
```

### 8.2 파이프라인

```
data/*.txt
    → TextLoader + DirectoryLoader
    → RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
    → OllamaEmbeddings(nomic-embed-text)
    → ChromaDB (./chroma_db/ 영구 저장)
```

---

## 9. 프로젝트 구조

```
c:\0.MyLab\LCAE-AI\
├── main.py                              # 파이프라인 진입점
├── Modelfile                            # Ollama 모델 정의
├── pyproject.toml                       # 프로젝트 메타데이터 + 의존성
├── .env                                 # 공통 설정 (git 포함)
├── .env.local.example                   # 민감 설정 템플릿 (API 키 등)
│
├── data/
│   ├── train_v2.jsonl                  # LLM 학습 데이터 (채팅 포맷)
│   └── train.jsonl
│
├── pipeline/
│   ├── stt/
│   │   ├── asr_pipeline.py             # ASRCore, VAD, Buffer
│   │   └── asr_Finetuning.py           # Whisper 파인튜닝
│   ├── tts/
│   │   └── tts_pipeline.py             # TTSCore, GTTSSynthesizer
│   ├── listenlist/
│   │   ├── listen_list.py              # ListenList 버퍼
│   │   └── transcriptions.jsonl        # 런타임 버퍼 파일
│   └── llm/
│       ├── chain/
│       │   ├── state.py                # AgentState + Pydantic 모델
│       │   ├── graph.py                # LangGraph 워크플로우
│       │   ├── nodes.py                # 노드 구현체
│       │   ├── setup.py                # 초기 상태 빌더
│       │   └── test_broadcast.py       # 방송 시뮬레이션 테스트
│       ├── prompts/
│       │   └── persona.py              # 시스템 프롬프트 (MC 페르소나)
│       └── utils/
│           ├── llm.py                  # 프로바이더 전환
│           ├── embeddings.py           # OllamaEmbeddings
│           ├── text_cleaner.py         # 필러 제거
│           └── ingest_data.py          # RAG 데이터 적재
│
├── scripts/
│   ├── train_sft.py                    # LLM LoRA 학습
│   └── merge_adapter.py               # 어댑터 병합
│
├── tests/
│   └── llm/
│       ├── test_rag.py                 # RAG 검색 테스트
│       └── run_loop.py                 # LLM 단독 대화 테스트
│
└── chroma_db/                          # 벡터 DB (런타임 생성)
```

---

## 10. 실행 가이드

### 10.1 환경 설정

```bash
# 가상환경 활성화
python -m venv venv
venv\Scripts\activate  # Windows

# 의존성 설치
pip install -e .

# Ollama 모델 등록
ollama create driving-mentor -f Modelfile

# RAG 데이터 적재 (선택)
python -m pipeline.llm.utils.ingest_data
```

### 10.2 실행 명령

```bash
# 마이크 직접 입력 (로컬 테스트)
python main.py --mode mic --model base --language ko

# WebSocket 서버
python main.py --mode server --port 8765

# LLM 단독 테스트 (대화형)
python -m tests.llm.run_loop

# 방송 시뮬레이션 테스트
python -m pipeline.llm.chain.test_broadcast

# RAG 검색 테스트
python -m tests.llm.test_rag

# 오디오 디바이스 목록
python main.py --list-devices
```

---

## 11. 핵심 의존성

| 패키지 | 버전 | 용도 |
|---|---|---|
| `langchain` | ≥1.2.15 | LLM 프레임워크 |
| `langgraph` | ≥1.1.6 | 그래프 오케스트레이션 |
| `faster-whisper` | ≥1.2.0 | ASR |
| `gtts` | ≥2.5.0 | TTS 합성 |
| `pydub` | ≥0.25.1 | 오디오 코덱 |
| `chromadb` | ≥1.5.7 | 벡터 DB |
| `langchain-ollama` | ≥1.1.0 | Ollama 연동 |
| `torch` | ≥2.3.1 | GPU/ML 백엔드 |
| `sounddevice` | ≥0.4.6 | 오디오 I/O |
| `langchain-groq` | 선택 | Groq 백엔드 |
| `peft`, `transformers` | 선택 | LoRA 학습 |

---

## 12. 설계 원칙

| 원칙 | 구현 |
|---|---|
| 콜백 아키텍처 | 각 Stage가 콜백으로 결과 전달, 스테이지 간 결합도 최소화 |
| 상태 머신 | `streaming_stage`로 방송 흐름 제어 (Main→QnA→Outro) |
| 플러그인 프로바이더 | LLM, TTS, VAD 엔진 교체 가능 |
| 구조화 출력 | Pydantic 모델로 LLM 출력 스키마 강제 |
| 레이턴시 최적화 | 정규식 전처리 → LLM 토큰 비용 절감 |
| 재접속 내성 | WebSocket 지수 백오프 (1→2→4→8→60s) |
