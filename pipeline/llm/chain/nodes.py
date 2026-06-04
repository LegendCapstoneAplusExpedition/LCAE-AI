from langchain_core.messages import AIMessage
from langchain_chroma import Chroma
from pipeline.llm.utils.llm import llm_structured, llm_lock
from pipeline.llm.utils.embeddings import embeddings
from pipeline.llm.utils.text_cleaner import clean_fillers
from pipeline.llm.chain.state import AgentState, AnalyzeAndWriteResult
from pipeline.llm.chain.scripted_responses import (
    curated_bridge,
    curated_closing,
    curated_summary,
    opening_script,
)
from pipeline.listenlist.paths import ready_summary_path, ready_question_path
import re
import time
from difflib import SequenceMatcher

_WORD_CHARS = re.compile(r'[가-힣a-zA-Z0-9]')
_SUMMARIZE_RE       = re.compile(r'(정리해|요약해|지금까지\s*내용)')
_QUESTION_REQ_RE    = re.compile(r'(질문\s*(받|정리|해주|넘겨|있|없|들어|올라|왔)|다음\s*질문|궁금한\s*거|채팅.{0,6}질문)')
_OUTRO_RE           = re.compile(r'(방송.{0,12}(마치|마무리|여기까지|종료)|오늘.{0,12}여기까지|다음에\s*또\s*(만나|뵙)|함께해\s*주셔서\s*감사)')

import os as _os

# 진행자 브릿지 멘트 발화 쿨다운(초). 이 시간 안에는 설명/질문에 다시 끼어들지 않음.
_BRIDGE_COOLDOWN_S = float(_os.getenv("AI_BRIDGE_COOLDOWN_S", "25"))

_vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
_db_has_data = _vector_db._collection.count() > 0

def _normalize_intent(intent: str) -> str:
    labels = ("질문요청", "정리요청", "마무리", "설명", "질문", "대기")
    raw = (intent or "").strip()
    if raw in labels:
        return raw
    for label in labels:
        if label in raw:
            return label
    return "대기"

def _is_echo_script(mc_script: str, mentor_text: str) -> bool:
    script = (mc_script or "").strip()
    source = (mentor_text or "").strip()
    if not script or not source:
        return False
    if script in source or source in script:
        return True
    return SequenceMatcher(None, script, source).ratio() >= 0.82

def _is_low_quality_bridge(mc_script: str) -> bool:
    script = (mc_script or "").strip()
    if not script:
        return True
    generic_phrases = (
        "현재 주제",
        "계속 진행",
        "논의하겠습니다",
        "이야기를 들어보겠습니다",
        "주요 요소",
        "강화하는 방법",
        "방향으로 개선",
        "알아보겠습니다",
        "중요합니다",
        "더 중요합니다",
    )
    if any(phrase in script for phrase in generic_phrases):
        return True
    words = set(_WORD_CHARS.findall(script))
    if len(words) < 8:
        return True
    return len(_WORD_CHARS.findall(script)) < 8

def _session_transcripts(state: AgentState) -> list[str]:
    from pipeline.listenlist.listen_list import ListenList

    broadcast_id = state.get("broadcast_id") or None
    return [
        str(entry.get("text", "")).strip()
        for entry in ListenList(broadcast_id=broadcast_id).read_all()
        if str(entry.get("text", "")).strip()
    ]


def _is_usable_summary(summary: str) -> bool:
    text = (summary or "").strip()
    if not text:
        return False
    invalid_markers = ("[topic]", "[intent]", "[mc_script]", "```", '{"topic"')
    return not any(marker in text for marker in invalid_markers)


def _read_ready_summary(state: AgentState) -> str:
    import json

    path = ready_summary_path(state.get("broadcast_id") or None)
    if not path.exists():
        return ""
    try:
        summary = str(
            json.loads(path.read_text(encoding="utf-8")).get("summary", "") or ""
        )
    except (AttributeError, json.JSONDecodeError):
        return ""
    return summary.strip() if _is_usable_summary(summary) else ""


# ──────────────────────────────────────────────
# fast_summarize_check  (LLM 없이 키워드로 정리요청 감지)
# ──────────────────────────────────────────────
def fast_intent_check(state: AgentState) -> str:
    cleaned = state.get("cleaned_text", "").strip()
    if state.get("streaming_stage") == "Opening":
        print("[FastCheck] Opening stage -> opening (오프닝 멘트 1회)")
        return "opening"
    if state.get("streaming_stage") == "Outro":
        print("[FastCheck] Outro stage -> wait")
        return "wait"
    if _SUMMARIZE_RE.search(cleaned):
        print(f"[FastCheck] 정리요청 감지 → pre-computed 요약 즉시 반환")
        return "summarize"
    if _QUESTION_REQ_RE.search(cleaned):
        print(f"[FastCheck] 질문요청 감지 → pre-computed 질문 즉시 반환")
        return "question"
    if _OUTRO_RE.search(cleaned):
        print("[FastCheck] 마무리 감지 → 클로징 즉시 반환")
        return "closing"
    bridge = curated_bridge(cleaned)
    if bridge:
        print(f"[FastCheck] 핵심 브릿지 감지 → \"{bridge}\"")
        return "bridge"
    if len(_WORD_CHARS.findall(cleaned)) < 5:
        print("[FastCheck] short/filler utterance -> skip LLM")
        return "wait"
    stage = state.get("streaming_stage", "Main")
    elapsed = time.time() - state.get("last_ai_speech_ts", 0.0)
    if stage == "Main" and elapsed < _BRIDGE_COOLDOWN_S:
        print(f"[FastCheck] bridge cooldown {elapsed:.0f}/{_BRIDGE_COOLDOWN_S:.0f}s -> skip LLM")
        return "wait"
    return "analyze"


# ──────────────────────────────────────────────
# preprocess_node  (~0ms, regex 기반)
# 재학습 완료 후 LLM 기반 중요도 분석으로 교체 예정
# ──────────────────────────────────────────────
def preprocess_node(state: AgentState):
    if not state["messages"]:
        return {"cleaned_text": ""}

    t0 = time.time()
    raw_text = state["messages"][-1].content
    cleaned = clean_fillers(raw_text)

    print(f"[Preprocess] 원문: \"{raw_text}\"")
    print(f"[Preprocess] 정제: \"{cleaned}\"  ({(time.time()-t0)*1000:.1f}ms)")
    return {"cleaned_text": cleaned}


# ──────────────────────────────────────────────
# knowledge_search_node  (~0.05s, Vector DB)
# cleaned_text로 직접 검색 (topic 추출 전에 실행)
# ──────────────────────────────────────────────
def knowledge_search_node(state: AgentState):
    search_text = state.get("cleaned_text", "").strip()

    if not _db_has_data:
        print("[Search] DB 비어있음 → 스킵")
        return {"retrieved_info": []}

    if len(search_text.split()) < 5:
        print(f"[Search] 짧은 발화({len(search_text.split())}단어) → 스킵")
        return {"retrieved_info": []}

    t0 = time.time()
    print(f"[Search] 검색 텍스트: \"{search_text[:60]}{'...' if len(search_text) > 60 else ''}\"")
    docs = _vector_db.similarity_search(search_text, k=2)
    retrieved = [doc.page_content for doc in docs]

    print(f"[Search] 결과 {len(retrieved)}건  ({time.time()-t0:.2f}s)")
    for i, doc in enumerate(retrieved, 1):
        print(f"[Search]   [{i}] {doc[:80]}{'...' if len(doc) > 80 else ''}")
    return {"retrieved_info": retrieved}


# ──────────────────────────────────────────────
# analyze_write_node  (~12s, LLM 1회 호출)
# 분석(topic/summary/intent) + MC 멘트 작성을 동시에 처리
# ──────────────────────────────────────────────
def analyze_write_node(state: AgentState):
    cleaned      = state.get("cleaned_text", "").strip()
    last_message = cleaned if cleaned else state["messages"][-1].content
    prev_summary = state.get("context_summary", "")
    stage        = state.get("streaming_stage", "Main")
    knowledge    = "\n".join(state.get("retrieved_info", []))

    # 유효 글자(한글·영문·숫자) 5자 미만 → 필러/추임새, LLM 스킵
    if len(_WORD_CHARS.findall(last_message)) < 5:
        print(f"[AnalyzeWrite] 짧은 발화 → LLM 스킵 (intent=대기)")
        return {
            "current_topic":   state.get("current_topic", ""),
            "context_summary": prev_summary,
            "intent":          "대기",
            "streaming_stage": stage,
            "mc_script":       "",
        }

    broadcast_topics = state.get("broadcast_topics", [])
    current_topic    = state.get("current_topic", "")
    topics_sec  = f"[방송 주제]: {', '.join(broadcast_topics)}" if broadcast_topics else ""
    current_sec = f"[현재 주제]: {current_topic}" if current_topic else ""
    knowledge_sec = f"\n[검색된 참고 지식]:\n{knowledge}" if knowledge else ""

    prompt = f"""{topics_sec}
{current_sec}
[이전 단계]: {stage}
{knowledge_sec}
[멘토 발화]: "{last_message}"

역할: 방송 흐름을 방해하지 않는 보조 MC.
원칙:
- 진행자의 말을 반복하거나 요약만 하지 마세요.
- "현재 주제", "계속 진행", "논의하겠습니다" 같은 빈 진행 멘트는 금지.
- 새 정보나 자연스러운 전환점이 없으면 intent는 "대기", mc_script는 "".
- 브릿지를 할 때만, 진행자 발화의 구체 키워드 1개를 짚고 다음 흐름을 열어주는 한 문장으로 작성.
- mc_script는 35자 이내의 짧은 한국어 한 문장.
- 종결어미는 "~네요", "~하네요", "~이네요" 같은 부드러운 구어체로. "~합니다", "~하겠습니다", "~입니다"체는 피하기.

좋은 브릿지 예시:
- 반응이 늦으면 경험이 끊긴다 → "결국 속도도 UX의 일부라는 말이네요."
- MVP는 핵심 기능 하나를 검증한다 → "핵심 기능 하나에 집중하자는 얘기네요."
- 역할 분담보다 커뮤니케이션이 중요하다 → "공유 방식이 성패를 가르는 지점이네요."
- 피드백을 받으며 작은 단위로 개선한다 → "피드백을 다음 개선으로 잇는 흐름이네요."

나쁜 브릿지 예시:
- "현재 주제를 계속 진행하겠습니다."
- "이 부분에 대해 논의하겠습니다."
- 진행자 발화를 그대로 반복한 문장.

반드시 아래 JSON 형식으로만 출력하세요.
{{"topic": "...", "intent": "설명|질문|질문요청|정리요청|마무리|대기", "mc_script": "..."}}
"""

    t0 = time.time()
    print(f"[AnalyzeWrite] LLM 호출 시작")

    # 백그라운드 요약(_summarize_to_ready)과 같은 Ollama 모델을 동시에 치지 않도록 직렬화
    with llm_lock:
        result: AnalyzeAndWriteResult = llm_structured.with_structured_output(
            AnalyzeAndWriteResult,
            method="json_mode",
        ).invoke([
            ("human", prompt),
        ])

    elapsed = time.time() - t0

    normalized_intent = _normalize_intent(result.intent)
    mc_script = (result.mc_script or "").strip()
    if _is_echo_script(mc_script, last_message):
        print("[AnalyzeWrite] mc_script echoes mentor text -> clear")
        mc_script = ""
    elif _is_low_quality_bridge(mc_script):
        print("[AnalyzeWrite] low-quality/generic bridge -> clear")
        mc_script = ""
    if not mc_script and normalized_intent in ("설명", "질문"):
        curated = curated_bridge(last_message)
        if curated:
            print("[AnalyzeWrite] curated bridge applied")
            mc_script = curated
    if normalized_intent == "대기":
        mc_script = ""

    if normalized_intent == "마무리":
        new_stage = "Outro"
    elif normalized_intent == "질문요청":
        new_stage = "QnA"
    elif stage == "QnA" and normalized_intent not in ("질문요청", "마무리"):
        new_stage = "Main"
    else:
        new_stage = stage

    print(f"[AnalyzeWrite] 완료  ({elapsed:.2f}s)")
    print(f"[AnalyzeWrite] topic={result.topic} | intent={normalized_intent} | stage={stage}→{new_stage}")
    print(f"[AnalyzeWrite] mc_script: \"{mc_script[:80]}{'...' if len(mc_script) > 80 else ''}\"")


    try:
        import json as _json, os as _os
        from pipeline.listenlist.chat_list import ChatList as _ChatList
        broadcast_id = (state.get("broadcast_id") or _os.getenv("BROADCAST_ID", "")).strip() or None
        ready_q_path = ready_question_path(broadcast_id)
        next_q = _ChatList(broadcast_id=broadcast_id).peek_next_question(broadcast_id=broadcast_id)
        if next_q:
            question = next_q.get("message", "").strip()
            username = next_q.get("username", "").strip()
            mc_text = f"{username}님 질문입니다. {question}" if username else question
            ready_q_path.write_text(
                _json.dumps({"time": time.strftime("%H:%M:%S"), "mc_text": mc_text}, ensure_ascii=False),
                encoding="utf-8",
            )
        elif ready_q_path.exists():
            ready_q_path.unlink()
    except Exception:
        pass

    return {
        "current_topic":   result.topic,
        "intent":          normalized_intent,
        "streaming_stage": new_stage,
        "mc_script":       mc_script,
    }

# ──────────────────────────────────────────────
# generate_opening_node  (방송 합류 직후 1회, 주제 기반 오프닝 멘트)
# streaming_stage="Opening"일 때만 발화하고 즉시 "Main"으로 전이한다.
# ──────────────────────────────────────────────
def generate_opening_node(state: AgentState):
    # Opening 단계가 아니면(이미 오프닝을 마쳤으면) 발화하지 않는다 → 1회 보장.
    if state.get("streaming_stage", "Main") != "Opening":
        print("[Opening] 이미 오프닝 완료 → 발화 생략")
        return {"intent": "대기", "mc_script": ""}

    topics = state.get("broadcast_topics", [])
    mc_script = opening_script(topics)
    print(f"[Opening] 오프닝 멘트 생성: \"{mc_script}\"")
    return {
        "intent":          "설명",
        "streaming_stage": "Main",  # 오프닝 후 본방송 단계로 전이
        "current_topic":   state.get("current_topic", "") or (topics[0] if topics else ""),
        "mc_script":       mc_script,
    }

def generate_curated_bridge_node(state: AgentState):
    mc_script = curated_bridge(state.get("cleaned_text", ""))
    return {
        "intent":          "설명" if mc_script else "대기",
        "streaming_stage": "Main" if mc_script else state.get("streaming_stage", "Main"),
        "mc_script":       mc_script,
        "pending_question": "",
    }


# ──────────────────────────────────────────────
# decision_node  (즉시, 규칙 기반 라우팅 함수)
# ──────────────────────────────────────────────
def decision_node(state: AgentState) -> str:
    intent  = state.get("intent", "")
    stage   = state.get("streaming_stage", "Main")

    if intent == "정리요청":
        print(f"[Decision] intent=\"{intent}\" → summarize")
        return "summarize"

    if intent == "질문요청":
        print(f"[Decision] intent=\"{intent}\" → ask_question")
        return "ask_question"

    # 마무리는 항상 발화
    if intent == "마무리":
        print(f"[Decision] intent=\"{intent}\" → speak")
        return "speak"

    # 설명·질문: 진행자 브릿지 멘트가 있으면 발화하되, 쿨다운으로 페이싱.
    # (멘토가 말할 때마다 끼어들지 않고, 마지막 발화 후 일정 시간 경과 시에만 반응)
    if intent in ("설명", "질문"):
        mc = state.get("mc_script", "").strip()
        if not mc:
            print(f"[Decision] intent=\"{intent}\" → wait (브릿지 멘트 없음)")
            return "wait"
        elapsed = time.time() - state.get("last_ai_speech_ts", 0.0)
        if elapsed >= _BRIDGE_COOLDOWN_S:
            print(f"[Decision] intent=\"{intent}\" → speak (쿨다운 경과 {elapsed:.0f}s)")
            return "speak"
        print(f"[Decision] intent=\"{intent}\" → wait (쿨다운 {elapsed:.0f}/{_BRIDGE_COOLDOWN_S:.0f}s)")
        return "wait"

    print(f"[Decision] intent=\"{intent}\" → wait")
    return "wait"


# ──────────────────────────────────────────────
# summarize_listenlist_node  (ready_summary.json 기반 즉시 반환)
# ──────────────────────────────────────────────
def summarize_listenlist_node(state: AgentState):
    transcripts = _session_transcripts(state)
    summary = curated_summary(transcripts) or _read_ready_summary(state)

    if not _is_usable_summary(summary):
        print("[Summarize] 요약 없음")
        summary = "아직 요약할 만한 방송 내용이 없네요."

    print(f"[Summarize] pre-computed 요약 반환: {summary[:60]}...")
    return {
        "intent":          "정리요청",
        "context_summary": summary,
        "mc_script":       summary,
    }


# ──────────────────────────────────────────────
# generate_question_node  (chat.jsonl에서 질문 pop)
# ──────────────────────────────────────────────
def generate_question_node(state: AgentState):
    import os
    import time
    from pipeline.listenlist.chat_list import ChatList

    t0 = time.time()

    broadcast_id = (state.get("broadcast_id") or os.getenv("BROADCAST_ID", "")).strip() or None
    chat_question = ChatList(broadcast_id=broadcast_id).pop_next_question(broadcast_id=broadcast_id)

    if not chat_question:
        print(f"[GenerateQuestion] 대기 질문 없음 ({(time.time()-t0)*1000:.1f}ms)")
        return {
            "intent":           "질문요청",
            "streaming_stage":  "QnA",
            "mc_script":        "아직 들어온 질문이 없네요.",
            "pending_question": "",
        }

    question = chat_question.get("message", "").strip()
    username = chat_question.get("username", "").strip()
    mc_script = f"{username}님 질문입니다. {question}" if username else question

    print(f"[GenerateQuestion] 질문 가져옴 ({(time.time()-t0)*1000:.1f}ms): {question}")
    return {
        "intent":           "질문요청",
        "streaming_stage":  "QnA",
        "mc_script":        mc_script,
        "pending_question": question,
    }


def generate_closing_node(state: AgentState):
    transcripts = _session_transcripts(state)
    mc_script = curated_closing(transcripts)
    if not mc_script:
        recap = _read_ready_summary(state)
        if not recap:
            topic = state.get("current_topic", "").strip()
            recap = f"{topic}의 핵심을 함께 짚어봤네요." if topic else ""
        recap_line = f"{recap} " if recap else ""
        mc_script = (
            f"네, 오늘도 함께해 주셔서 감사합니다. {recap_line}"
            "다음 멘토링에서 또 뵙겠습니다."
        )
    return {
        "intent": "마무리",
        "streaming_stage": "Outro",
        "mc_script": mc_script,
    }


# ──────────────────────────────────────────────
# output_node  (즉시, mc_script → messages)
# ──────────────────────────────────────────────
def output_node(state: AgentState):
    mc_script = state.get("mc_script", "").strip()

    if not mc_script:
        print("[Output] mc_script 없음 → 발화 생략")
        return {}

    print(f"[Output] 최종 멘트: \"{mc_script}\"")
    return {
        "messages": [AIMessage(content=mc_script)],
        "last_ai_speech_ts": time.time(),  # 쿨다운 기준 시각 갱신
    }
