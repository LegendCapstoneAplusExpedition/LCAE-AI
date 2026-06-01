from langchain_core.messages import AIMessage
from langchain_chroma import Chroma
from pathlib import Path
from pipeline.llm.utils.llm import llm_structured
from pipeline.llm.utils.embeddings import embeddings
from pipeline.llm.utils.text_cleaner import clean_fillers
from pipeline.llm.chain.state import AgentState, AnalyzeAndWriteResult
import re
import time

_WORD_CHARS = re.compile(r'[가-힣a-zA-Z0-9]')
_SUMMARIZE_RE       = re.compile(r'(정리해|요약해|지금까지\s*내용)')
_QUESTION_REQ_RE    = re.compile(r'(질문\s*(받|정리|해주|넘겨|있어요|들어왔)|다음\s*질문|궁금한\s*거)')

import os as _os
_READY_SUMMARY_PATH  = Path(__file__).parent.parent.parent / "listenlist" / "ready_summary.json"
_READY_QUESTION_PATH = Path(__file__).parent.parent.parent / "listenlist" / "ready_question.json"

# 진행자 브릿지 멘트 발화 쿨다운(초). 이 시간 안에는 설명/질문에 다시 끼어들지 않음.
_BRIDGE_COOLDOWN_S = float(_os.getenv("AI_BRIDGE_COOLDOWN_S", "25"))

_vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
_db_has_data = _vector_db._collection.count() > 0


# ──────────────────────────────────────────────
# fast_summarize_check  (LLM 없이 키워드로 정리요청 감지)
# ──────────────────────────────────────────────
def fast_intent_check(state: AgentState) -> str:
    cleaned = state.get("cleaned_text", "").strip()
    if _SUMMARIZE_RE.search(cleaned):
        print(f"[FastCheck] 정리요청 감지 → pre-computed 요약 즉시 반환")
        return "summarize"
    if _QUESTION_REQ_RE.search(cleaned):
        print(f"[FastCheck] 질문요청 감지 → pre-computed 질문 즉시 반환")
        return "question"
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
    question_queue = state.get("question_queue", [])
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
    q_list        = "\n".join(f"- {q}" for q in question_queue) if question_queue else "없음"
    knowledge_sec = f"\n[검색된 참고 지식]:\n{knowledge}" if knowledge else ""

    prompt = f"""{topics_sec}
{current_sec}
[이전 단계]: {stage}
[대기 질문]: {q_list}{knowledge_sec}
[멘토 발화]: "{last_message}"

반드시 아래 JSON 형식으로만 출력하세요.
- mc_script: 문자열이어야 함.
{{"topic": "...", "intent": "...", "mc_script": "..."}}
"""

    t0 = time.time()
    print(f"[AnalyzeWrite] LLM 호출 시작")

    result: AnalyzeAndWriteResult = llm_structured.with_structured_output(
        AnalyzeAndWriteResult,
        method="json_mode",
    ).invoke([
        ("human", prompt),
    ])

    elapsed = time.time() - t0

    if result.intent == "마무리":
        new_stage = "Outro"
    elif result.intent == "질문요청":
        new_stage = "QnA"
    elif stage == "QnA" and result.intent not in ("질문요청", "마무리"):
        new_stage = "Main"
    else:
        new_stage = stage

    print(f"[AnalyzeWrite] 완료  ({elapsed:.2f}s)")
    print(f"[AnalyzeWrite] topic={result.topic} | intent={result.intent} | stage={stage}→{new_stage}")
    print(f"[AnalyzeWrite] mc_script: \"{result.mc_script[:80]}{'...' if len(result.mc_script) > 80 else ''}\"")


    try:
        import json as _json, os as _os
        from pipeline.listenlist.chat_list import ChatList as _ChatList
        broadcast_id = _os.getenv("BROADCAST_ID", "").strip() or None
        next_q = _ChatList().peek_next_question(broadcast_id=broadcast_id)
        if next_q:
            question = next_q.get("message", "").strip()
            username = next_q.get("username", "").strip()
            mc_text = f"{username}님 질문입니다. {question}" if username else question
            _READY_QUESTION_PATH.write_text(
                _json.dumps({"time": time.strftime("%H:%M:%S"), "mc_text": mc_text}, ensure_ascii=False),
                encoding="utf-8",
            )
        elif _READY_QUESTION_PATH.exists():
            _READY_QUESTION_PATH.unlink()
    except Exception:
        pass

    return {
        "current_topic":   result.topic,
        "intent":          result.intent,
        "streaming_stage": new_stage,
        "mc_script":       result.mc_script,
    }


# ──────────────────────────────────────────────
# decision_node  (즉시, 규칙 기반 라우팅 함수)
# ──────────────────────────────────────────────
def decision_node(state: AgentState) -> str:
    intent  = state.get("intent", "")
    stage   = state.get("streaming_stage", "Main")
    pending = state.get("pending_question", "")

    if stage == "Outro" and intent == "마무리":
        print(f"[Decision] Outro 진입 완료 → 추가 클로징 차단")
        return "wait"

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
    import json as _json

    summary = ""
    if _READY_SUMMARY_PATH.exists():
        try:
            data = _json.loads(_READY_SUMMARY_PATH.read_text(encoding="utf-8"))
            summary = data.get("summary", "").strip()
        except Exception:
            pass

    if not summary:
        summary = state.get("context_summary", "").strip()

    if not summary:
        print("[Summarize] 요약 없음")
        return {"mc_script": "아직 요약할 방송 내용이 없습니다."}

    print(f"[Summarize] pre-computed 요약 반환: {summary[:60]}...")
    return {"mc_script": summary}


# ──────────────────────────────────────────────
# generate_question_node  (chat.jsonl에서 질문 pop)
# ──────────────────────────────────────────────
def generate_question_node(state: AgentState):
    import os
    import time
    from pipeline.listenlist.chat_list import ChatList

    t0 = time.time()

    broadcast_id = os.getenv("BROADCAST_ID", "").strip() or None
    chat_question = ChatList().pop_next_question(broadcast_id=broadcast_id)

    if not chat_question:
        print(f"[GenerateQuestion] 대기 질문 없음 ({(time.time()-t0)*1000:.1f}ms)")
        return {
            "mc_script":        "현재 대기 중인 질문이 없습니다.",
            "pending_question": "",
        }

    question = chat_question.get("message", "").strip()
    username = chat_question.get("username", "").strip()
    mc_script = f"{username}님 질문입니다. {question}" if username else question

    print(f"[GenerateQuestion] 질문 가져옴 ({(time.time()-t0)*1000:.1f}ms): {question}")
    return {"mc_script": mc_script}


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


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    from pipeline.llm.chain.setup import mentor_setup

    # 방송 전: 멘토가 주제 키워드 사전 입력
    BASE = mentor_setup(["주니어-시니어 성장", "번아웃 예방", "MVP 전략", "프리랜서 단가"])
    print(f"[Setup] broadcast_topics={BASE['broadcast_topics']}")
    print(f"[Setup] current_topic={BASE['current_topic']} | streaming_stage={BASE['streaming_stage']}")

    CASES = [
        # streaming_stage 없음 — LLM이 발화 맥락만 보고 스스로 판단
        {
            "_desc": "케이스1: 짧은 filler (대기 예상)",
            "messages":        [HumanMessage(content="음... 그러니까 그게 말이죠.")],
            "silence_duration": 1.0,
            "question_queue":  [],
            "current_topic":   "주니어-시니어 성장",
            "context_summary": "시니어는 기술 스택보다 문제 정의 능력이 중요하다.",
        },
        {
            "_desc": "케이스2: 실질 내용 발화 (브릿지 멘트 예상)",
            "messages":        [HumanMessage(content="번아웃은 시간 관리 실패가 아니라 의미를 잃었을 때 생겨요. 작은 완성 경험을 쌓는 게 핵심이에요.")],
            "silence_duration": 3.5,
            "question_queue":  [],
            "current_topic":   "번아웃 예방 전략",
            "context_summary": "하루를 집중 블록과 커뮤니케이션 블록으로 나누는 것이 효과적이다.",
        },
        {
            "_desc": "케이스3: 정리요청 — 누적 요약 읽기 예상",
            "messages":        [HumanMessage(content="지금까지 내용 정리해주세요.")],
            "silence_duration": 2.0,
            "question_queue":  [],
            "current_topic":   "MVP 최소 기능 정의",
            "context_summary": "사이드 프로젝트는 고객 인터뷰 5개로 수요 검증 후 시작해야 한다. MVP는 핵심 기능 하나로 빠르게 출시하는 것이 효율적이다.",
        },
        {
            "_desc": "케이스4: 질문요청 — 대기 질문 전달 예상",
            "messages":        [HumanMessage(content="잠깐 질문 받을게요.")],
            "silence_duration": 2.0,
            "question_queue":  ["단가 인상은 언제 하는 게 좋나요?", "포트폴리오에 사이드 프로젝트도 넣어도 되나요?"],
            "current_topic":   "프리랜서 단가 인상",
            "context_summary": "단가 인상은 JSS 90% 이상, 리뷰 5개 달성 시점이 적기다.",
        },
        {
            "_desc": "케이스5: 마무리 — 클로징 멘트 예상 + Outro 판단 예상",
            "messages":        [HumanMessage(content="오늘 방송 여기서 마무리할게요. 감사합니다.")],
            "silence_duration": 2.0,
            "question_queue":  [],
            "current_topic":   "작업 견적 산정",
            "context_summary": "작업 견적은 실제 시간에 커뮤니케이션·수정·버퍼를 더해 1.5배 곱하는 것이 현실적이다. 단가 인상은 JSS 90% 이상 시점이 적기다.",
        },
    ]

    for case in CASES:
        desc = case.pop("_desc")
        print(f"\n{'='*60}")
        print(f"  {desc}")
        print('='*60)
        # BASE(setup 초기값) 위에 케이스별 override 병합
        state = {**BASE, **case}
        state["cleaned_text"] = ""
        state["retrieved_info"] = []
        state["intent"] = ""
        state["mc_script"] = ""

        print("\n--- [1] preprocess ---")
        state.update(preprocess_node(state))

        print("\n--- [2] search ---")
        state.update(knowledge_search_node(state))

        print("\n--- [3] analyze_write ---")
        state.update(analyze_write_node(state))

        print("\n--- [4] decision ---")
        print("결과:", decision_node(state))
