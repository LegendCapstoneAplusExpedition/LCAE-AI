# PRD: 실시간 멘토링 AI MC 브릿지 멘트 생성 LLM 모듈

| 항목 | 내용 |
|---|---|
| 모듈명 | `LLM Pipeline` (pipeline/llm/) |
| 작성일 | 2026-05-12 |
| 버전 | v1.0 |
| 작성자 | sucheoli |

---

## 1. 개요

STT 모듈로부터 전달받은 멘토의 발화 텍스트를 LangGraph 기반 멀티노드 파이프라인으로 분석하고, RAG(Retrieval-Augmented Generation)를 통해 관련 전문 지식을 검색한 뒤, AI MC 페르소나로 자연스러운 브릿지 멘트를 생성하여 TTS 모듈에 전달하는 LLM 파이프라인 모듈.

STT → **LLM(LangGraph)** → TTS 파이프라인의 중간 지능 구간을 담당한다.

---

## 2. 배경 및 목표

### 배경

2026 캡스톤 시스템은 실시간 멘토링 방송 환경에서 멘토의 발화를 인식하고, AI MC가 적절한 타이밍에 개입하여 멘토-멘티 간 대화를 자연스럽게 연결하는 기능을 필요로 한다. 단순한 STT → TTS 직결 구조에서 벗어나, LLM이 문맥을 파악하고 전문 지식을 보강하여 방송 품질을 높이는 중간 지능 계층이 요구된다.

### 목표

- 멘토 발화에서 주제·요약·의도를 구조화된 형태로 자동 추출
- ChromaDB 기반 RAG로 관련 전문 지식을 실시간 검색·주입
- 규칙 기반 판단으로 AI MC 개입 여부(speak / wait) 결정
- AI MC 페르소나(아나운서 스타일)를 유지한 브릿지 멘트 생성
- 생성된 텍스트를 TTS 모듈 콜백으로 전달

### 성공 기준

| 지표 | 목표값 |
|---|---|
| 전체 LLM 처리 지연 (STT 결과 수신 → 텍스트 출력) | < 5초 |
| script_writer_node 단독 응답 시간 | < 3초 (gpt-4o-mini 기준) |
| 개입 판단 정확도 (speak/wait) | 의도 기반 규칙 충족 시 100% |
| RAG 검색 결과 반환 수 | 상위 k=2 문서 조각 |

---

## 3. 범위 (Scope)

### In Scope

- STT `TranscriptionResult`로부터 텍스트 수신
- LangGraph StateGraph를 통한 멀티노드 순차/조건 처리
- 멘토 발화 분석 (주제·요약·의도 추출, 구조화 출력)
- ChromaDB 벡터 검색 기반 RAG
- 규칙 기반 AI MC 개입 판단 (침묵 시간, 의도, 질문 큐 수)
- AI MC 페르소나 브릿지 멘트 생성
- RAG 데이터 수집·인덱싱 스크립트 (`ingest_data.py`)
- LLM 단독 실행 테스트 루프 (`tests/llm/run_loop.py`)

### Out of Scope

- STT (ASR_PRD.md 참조)
- TTS (TTS_PRD.md 참조)
- 백엔드 WebSocket 서버/클라이언트
- 프론트엔드 UI
- 멘티 질문 큐 수집 로직 (별도 구현 예정)
- 화자 분리 (Speaker Diarization)
- LLM 모델 파인튜닝

---

## 4. 시스템 인터페이스 명세

### 4.1 텍스트 입력 — STT 콜백 연동

STT 모듈의 `on_transcription` 콜백에서 `AgentState`를 구성하여 LangGraph `app.invoke()`를 호출한다.

```python
from pipeline.llm.chain.graph import app
from pipeline.llm.chain.state import AgentState
from langchain_core.messages import HumanMessage

def on_transcription(result: TranscriptionResult) -> None:
    state: AgentState = {
        "messages": [HumanMessage(content=result["text"])],
        "silence_duration": 6.0,   # STT 침묵 감지 값 전달
        "question_queue": [],
        "current_topic": "",
        "retrieved_info": [],
        "intent": "",
        # ...
    }
    llm_result = app.invoke(state)
    mc_text = llm_result["messages"][-1].content  # TTS 입력
```

### 4.2 AgentState — 파이프라인 내부 상태

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # 대화 기록
    is_speaking: bool           # 멘토 발화 여부
    silence_duration: float     # 침묵 지속 시간(초)
    question_queue: List[Dict]  # 멘티 질문 큐
    current_topic: Optional[str]   # 현재 주제 키워드 (analyzer 추출)
    context_summary: str           # 대화 내용 요약 (analyzer 추출)
    retrieved_info: List[str]      # RAG 검색 결과 (search 추출)
    streaming_stage: str           # 방송 단계: Intro / Main / QnA / Outro
    intent: str                    # 멘토 발화 의도 (analyzer 추출)
```

### 4.3 LLM 출력 — TTS 연동

`script_writer_node`가 생성한 멘트는 `AgentState.messages`의 마지막 `AIMessage`로 저장된다. `app.invoke()` 반환 후 `result["messages"][-1].content`를 TTS 모듈에 전달한다.

```python
mc_text: str = llm_result["messages"][-1].content
tts.synthesize(mc_text)
```

`decision_node`가 `"wait"`을 반환하면 `writer` 노드가 실행되지 않으므로 `messages` 길이가 증가하지 않는다 — TTS 호출 없이 경청 상태를 유지한다.

### 4.4 RAG 데이터 수집 인터페이스

```python
# 프로젝트 루트에서 실행
python -m pipeline.llm.chain.ingest_data

# data/ 폴더에 .txt 파일을 넣으면 자동으로 500자 청크 분할 → ChromaDB 저장
# 저장 경로: ./chroma_db/
```

---

## 5. 기능 요구사항

| ID | 요구사항 | 우선순위 |
|---|---|---|
| FR-01 | STT `TranscriptionResult`의 텍스트를 `HumanMessage`로 변환하여 `AgentState`에 주입 | 필수 |
| FR-02 | `analyzer_node`: 멘토 발화에서 topic·summary·intent를 `AnalysisResult` 구조화 출력으로 추출 | 필수 |
| FR-03 | `knowledge_search_node`: 추출된 topic으로 ChromaDB 유사도 검색(k=2) 실행 | 필수 |
| FR-04 | `decision_node`: silence_duration≥5초 / intent에 "question" 포함 / question_queue≥3개 조건 중 하나 충족 시 "speak" 반환 | 필수 |
| FR-05 | `script_writer_node`: SYSTEM_PROMPT 페르소나 + 요약 + 발화 + 검색 지식을 결합하여 브릿지 멘트 생성 | 필수 |
| FR-06 | `decision_node`가 "wait" 반환 시 writer 노드 건너뛰고 END로 즉시 종료 | 필수 |
| FR-07 | LangGraph `add_messages` 리듀서로 대화 기록 누적 관리 | 필수 |
| FR-08 | `ingest_data.py`로 텍스트 문서를 ChromaDB에 인덱싱 | 필수 |
| FR-09 | SYSTEM_PROMPT를 `prompts/persona.py`로 분리하여 노드 간 공유 | 필수 |
| FR-10 | LLM 모델(gpt-4o-mini)·임베딩 모델(text-embedding-3-small)을 환경변수(`OPENAI_API_KEY`)로 설정 | 필수 |
| FR-11 | `script_writer_node` 응답 시간을 콘솔에 출력하여 성능 모니터링 | 권장 |
| FR-12 | `question_queue` 누적 수에 따른 개입 우선순위 지원 | 권장 |

---

## 6. 비기능 요구사항

| ID | 요구사항 |
|---|---|
| NFR-01 | 전체 LangGraph 처리 지연 < 5초 (네트워크 latency 포함) |
| NFR-02 | `script_writer_node` 단독 LLM 응답 시간 < 3초 (gpt-4o-mini 기준) |
| NFR-03 | `AgentState`는 TypedDict로 타입 안전성 보장, 구조화 출력은 Pydantic BaseModel(`AnalysisResult`) 사용 |
| NFR-04 | OPENAI_API_KEY는 `.env` 파일에서만 로드, 코드에 하드코딩 금지 |
| NFR-05 | ChromaDB 벡터 DB는 `./chroma_db/`에 로컬 파일로 저장, 별도 서버 불필요 |
| NFR-06 | LangGraph 노드 함수는 순수 함수 구조 — 입력 State → 출력 dict, 사이드이펙트 없음 |
| NFR-07 | `pipeline.llm` 모듈은 `from pipeline.llm import app` 단일 진입점으로 외부에 노출 |

---

## 7. 데이터 파이프라인 (RAG 인덱싱)

### 7.1 전체 흐름

```
[1] 텍스트 문서 준비
    - 위치: ./data/*.txt
    - 내용: 멘토링 도메인 전문 지식 (기술 문서, 강의 자료 등)

[2] 문서 로드 및 청크 분할
    - 로더: DirectoryLoader + TextLoader
    - 분할: RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)

[3] 벡터 임베딩 및 저장
    - 임베딩 모델: OpenAIEmbeddings (text-embedding-3-small)
    - 벡터 DB: ChromaDB (./chroma_db/)

[4] 실시간 검색
    - knowledge_search_node에서 similarity_search(topic, k=2)
    - 반환된 문서 조각 → AgentState.retrieved_info
```

### 7.2 인덱싱 실행

```bash
# data/ 폴더에 .txt 파일 준비 후 실행
python -m pipeline.llm.chain.ingest_data
# 출력 예: 총 42개의 텍스트 조각으로 분할되었습니다.
#          인덱싱 완료! 이제 ./chroma_db 폴더에 데이터가 저장되었습니다.
```

---

## 8. 모듈 구조 (Architecture)

### 8.1 LangGraph 데이터 흐름

```
[STT Module]
    │ TranscriptionResult.text
    ▼
HumanMessage → AgentState.messages
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  LangGraph StateGraph (pipeline/llm/chain/graph.py) │
│                                                     │
│  START                                              │
│    │                                                │
│    ▼                                                │
│  analyzer_node ─────────────────────────────────►  │
│    (topic, summary, intent 추출)                    │
│    │                                                │
│    ▼                                                │
│  knowledge_search_node ──────────────────────────►  │
│    (ChromaDB RAG 검색 → retrieved_info)             │
│    │                                                │
│    ▼                                                │
│  decision_node (Conditional Edge)                   │
│    ├─ "speak" ──► script_writer_node ──► END        │
│    │               (AI MC 브릿지 멘트 생성)          │
│    └─ "wait"  ──► END (경청 유지)                   │
└─────────────────────────────────────────────────────┘
    │ AIMessage (브릿지 멘트 텍스트)
    ▼
[TTS Module]
    tts.synthesize(mc_text)
```

### 8.2 모듈 구성 및 파일 책임

| 파일 | 책임 |
|---|---|
| `chain/state.py` | `AgentState` TypedDict 정의, `AnalysisResult` Pydantic 스키마 |
| `chain/graph.py` | StateGraph 정의 — 노드 등록, 고정 엣지, 조건부 엣지, 앱 컴파일 |
| `chain/nodes.py` | 4개 노드 함수 구현 — analyzer, knowledge_search, decision, script_writer |
| `chain/ingest_data.py` | RAG 데이터 수집·청크 분할·ChromaDB 인덱싱 스크립트 |
| `prompts/persona.py` | `SYSTEM_PROMPT` — AI MC 페르소나 지침 (스타일·연결 로직·개입 타이밍) |
| `utils/llm.py` | `ChatOpenAI(gpt-4o-mini)` 싱글턴 인스턴스, `.env` 로드 |
| `utils/embeddings.py` | `OpenAIEmbeddings(text-embedding-3-small)` 싱글턴 인스턴스 |
| `__init__.py` | `app` (컴파일된 LangGraph) 외부 노출 |

### 8.3 노드 상세

| 노드 | 입력 State 필드 | 출력 State 필드 | LLM 호출 |
|---|---|---|---|
| `analyzer_node` | `messages[-1].content` | `current_topic`, `context_summary`, `intent` | `with_structured_output(AnalysisResult)` |
| `knowledge_search_node` | `current_topic` | `retrieved_info` | 없음 (ChromaDB 검색) |
| `decision_node` | `silence_duration`, `intent`, `question_queue` | `"speak"` / `"wait"` (라우팅 값) | 없음 (규칙 기반) |
| `script_writer_node` | `context_summary`, `current_topic`, `retrieved_info`, `messages[-1]` | `messages` (+AIMessage), `streaming_stage` | `llm.invoke()` |

### 8.4 개입 판단 규칙 (decision_node)

```
silence_duration >= 5.0초   → speak  (멘토 발화 종료 후 정적)
"question" in intent        → speak  (멘토가 질문을 던짐)
len(question_queue) >= 3    → speak  (멘티 질문이 누적됨)
그 외                        → wait   (경청 유지)
```

---

## 9. 기술 스택

| 패키지 | 버전 | 용도 |
|---|---|---|
| `langgraph` | >=1.1.6 | StateGraph 기반 멀티노드 파이프라인 |
| `langchain` | >=1.2.15 | LLM 추상화 레이어, 메시지 타입 |
| `langchain-openai` | >=1.1.12 | ChatOpenAI, OpenAIEmbeddings |
| `langchain-community` | >=0.4.1 | DirectoryLoader, TextLoader |
| `langchain-chroma` | >=1.1.0 | ChromaDB LangChain 연동 |
| `chromadb` | >=1.5.7 | 벡터 DB (로컬 파일 기반) |
| `pydantic` | >=2.12.5 | AnalysisResult 구조화 출력 스키마 |
| `python-dotenv` | >=1.2.2 | `.env` 환경변수 로드 |

### 외부 서비스

| 서비스 | 용도 | 환경변수 |
|---|---|---|
| OpenAI API | ChatCompletion (gpt-4o-mini) | `OPENAI_API_KEY` |
| OpenAI API | Embeddings (text-embedding-3-small) | `OPENAI_API_KEY` |

---

## 10. 설정 (환경변수)

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `OPENAI_API_KEY` | ― (필수) | OpenAI LLM·임베딩 API 키 |

LLM 모델명, 임베딩 모델명, ChromaDB 경로는 현재 코드에 고정값으로 관리되며, 향후 환경변수화 가능.

| 설정 | 현재 고정값 | 위치 |
|---|---|---|
| LLM 모델 | `gpt-4o-mini` | `utils/llm.py` |
| LLM temperature | `0.7` | `utils/llm.py` |
| 임베딩 모델 | `text-embedding-3-small` | `utils/embeddings.py` |
| ChromaDB 경로 | `./chroma_db` | `chain/nodes.py`, `chain/ingest_data.py` |
| RAG 검색 결과 수 | `k=2` | `chain/nodes.py` |
| 개입 침묵 임계값 | `5.0초` | `chain/nodes.py` |
| 질문 큐 임계값 | `3개` | `chain/nodes.py` |

---

## 11. 테스트 시나리오

### 단위 테스트

- `analyzer_node` → `AnalysisResult` 형식 검증 (`topic`, `summary`, `intent` 비어있지 않음)
- `decision_node` → `silence_duration=6.0` 입력 시 `"speak"` 반환 확인
- `decision_node` → `silence_duration=2.0`, `intent="설명"` 입력 시 `"wait"` 반환 확인
- `script_writer_node` → `messages[-1]`이 `AIMessage`이고 content 비어있지 않음 확인

```bash
# 단일 노드 테스트 (프로젝트 루트에서 실행)
python tests/llm/test_analyzer.py
python tests/llm/test_full_logic.py
python tests/llm/test_rag.py
```

### 통합 테스트

- `app.invoke(state)` → 전체 그래프 순회, `streaming_stage == "Output_Ready"` 확인
- `silence_duration=2.0` → `app.invoke()` → messages 증가 없음 (wait 경로) 확인

```bash
python tests/llm/test_full_sequence.py
```

### 성능 테스트

- `script_writer_node` 응답 시간 3초 이내 확인 (콘솔 출력 `⏱Script Writer 노드 응답 시간` 확인)
- 전체 `app.invoke()` 처리 시간 5초 이내 확인

### 실시간 루프 테스트

```bash
# 텍스트 입력으로 LLM 파이프라인 전체 동작 확인 (STT/TTS 없이)
python tests/llm/run_loop.py
```

---

## 12. 실행 방법

```bash
# 1. 환경변수 설정
echo "OPENAI_API_KEY=sk-..." >> .env

# 2. RAG 데이터 인덱싱 (최초 1회)
#    data/ 폴더에 .txt 파일 준비 후:
python -m pipeline.llm.chain.ingest_data

# 3. LLM 파이프라인 단독 테스트
python tests/llm/run_loop.py

# 4. 통합 파이프라인 (STT → TTS, LLM 연결 후)
python main.py --mode mic
```

---

## 13. 향후 고려 사항 (Future Considerations)

- **STT 침묵 감지 연동**: `ASRCore`의 침묵 시간을 `AgentState.silence_duration`에 실시간 주입하는 인터페이스 구체화
- **멘티 질문 큐 연동**: 프론트엔드/백엔드에서 멘티 질문을 수신하여 `question_queue`에 누적하는 별도 핸들러 구현
- **스트리밍 응답**: LangChain `stream()` API 활용으로 TTS 첫 음절 지연 단축
- **LLM 모델 교체**: `utils/llm.py` 수정만으로 GPT-4o / Claude 등 다른 모델 전환
- **방송 단계별 페르소나 분기**: `streaming_stage`(Intro/Main/QnA/Outro)에 따라 `SYSTEM_PROMPT` 동적 선택
- **ChromaDB 경로 외부화**: `.env`의 `CHROMA_DB_PATH` 환경변수로 벡터 DB 경로 설정
- **멀티모달 입력 지원**: 멘토 화면 공유(이미지/슬라이드)를 RAG 소스로 추가
