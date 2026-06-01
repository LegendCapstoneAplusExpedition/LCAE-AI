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
_SUMMARIZE_RE = re.compile(r'(정리해|요약해|지금까지\s*내용)')

_READY_SUMMARY_PATH = Path(__file__).parent.parent.parent / "listenlist" / "ready_summary.json"

_vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
_db_has_data = _vector_db._collection.count() > 0


# ──────────────────────────────────────────────
# fast_summarize_check  (LLM 없이 키워드로 정리요청 감지)
# ──────────────────────────────────────────────
def fast_summarize_check(state: AgentState) -> str:
    cleaned = state.get("cleaned_text", "").strip()
    if _SUMMARIZE_RE.search(cleaned):
        print(f"[FastCheck] 정리요청 감지 → pre-computed 요약 즉시 반환")
        return "summarize"
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

    # 프롬프트에 넣는 요약은 최신 3문장으로 제한 (토큰 증가 방지)
    q_list        = "\n".join(f"- {q}" for q in question_queue) if question_queue else "없음"
    knowledge_sec = f"\n[검색된 참고 지식]:\n{knowledge}" if knowledge else ""

    prompt = f"""[이전 단계]: {stage}
[누적 요약]: {prev_summary if prev_summary else "없음"}
[대기 질문]: {q_list}{knowledge_sec}
[멘토 발화]: "{last_message}"

반드시 아래 JSON 형식으로만 출력하세요. 규칙:
- summary: 누적 요약에 이번 발화의 핵심만 1~2문장으로 추가. 기존 내용은 삭제하지 말 것. 메타 발화("정리해줘" 등)는 제외.
- mc_script: 문자열이어야 함.
{{"topic": "...", "summary": "...", "intent": "...", "mc_script": "..."}}
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

    # streaming_stage: intent 기반 전환 (LLM 출력 아님)
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
    print(f"[AnalyzeWrite] summary: {result.summary}")
    print(f"[AnalyzeWrite] mc_script: \"{result.mc_script[:80]}{'...' if len(result.mc_script) > 80 else ''}\"")

    return {
        "current_topic":   result.topic,
        "context_summary": result.summary,
        "intent":          result.intent,
        "streaming_stage": new_stage,
        "mc_script":       result.mc_script,
    }

    if result.summary:
        try:
            import json as _json
            _READY_SUMMARY_PATH.write_text(
                _json.dumps({"time": time.strftime("%H:%M:%S"), "summary": result.summary}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass


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

    if pending and stage == "QnA" and intent not in ("대기", "마무리"):
        print(f"[Decision] 대기 질문 있음 + QnA 스테이지 → answer_question")
        return "answer_question"

    should_speak = intent == "마무리"
    decision = "speak" if should_speak else "wait"
    print(f"[Decision] intent=\"{intent}\" → {decision}")
    return decision


# ──────────────────────────────────────────────
# summarize_listenlist_node  (summary.jsonl 기반 1줄 요약)
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
# generate_question_node  (채팅 DB → summary.jsonl 기반 질문 생성)
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
    return {"mc_script": mc_script, "pending_question": question}


# ──────────────────────────────────────────────
# assess_search_node  (웹 검색 필요 여부 판단)
# ──────────────────────────────────────────────
def assess_search_node(state: AgentState):
    import time
    from langchain_core.messages import HumanMessage
    from pipeline.llm.utils.llm import llm_structured

    question = state.get("pending_question", "")
    context  = state.get("context_summary", "")

    t0 = time.time()
    prompt = (
        f"질문: {question}\n"
        f"방송 주제 요약: {context}\n\n"
        "이 질문에 답하기 위해 최신 뉴스·통계·현재 상황 등 시간에 민감한 정보가 필요하면 YES, "
        "LLM 지식만으로 충분하면 NO로만 답하세요."
    )
    result = llm_structured.invoke([HumanMessage(content=prompt)])
    content = (result.content if hasattr(result, "content") else str(result)).strip().upper()
    needs = "YES" in content

    print(f"[AssessSearch] 웹 검색 필요={needs}  ({(time.time()-t0):.2f}s)")
    return {"needs_web_search": needs}


def decision_search_node(state: AgentState) -> str:
    """assess_search_node 이후 라우팅 — 검색 필요 여부에 따라 분기."""
    decision = "search" if state.get("needs_web_search", False) else "direct"
    print(f"[DecisionSearch] → {decision}")
    return decision


# ──────────────────────────────────────────────
# tavily_search_node  (Tavily 웹 검색)
# ──────────────────────────────────────────────
def tavily_search_node(state: AgentState):
    import time
    from langchain_community.tools.tavily_search import TavilySearchResults

    question = state.get("pending_question", "")
    if not question:
        return {"web_search_results": []}

    t0 = time.time()
    try:
        results = TavilySearchResults(max_results=3).invoke(question)
        texts = [r.get("content", "") for r in results if isinstance(r, dict) and r.get("content")]
        print(f"[Tavily] {len(texts)}건 검색 완료  ({(time.time()-t0):.2f}s)")
        return {"web_search_results": texts}
    except Exception as e:
        print(f"[Tavily] 검색 실패: {e}")
        return {"web_search_results": []}


# ──────────────────────────────────────────────
# answer_question_node  (요약 + Tavily 기반 3줄 답변)
# ──────────────────────────────────────────────
def answer_question_node(state: AgentState):
    import time
    from pipeline.listenlist.listen_list import ListenList
    from langchain_core.messages import HumanMessage
    from pipeline.llm.utils.llm import llm

    question     = state.get("pending_question", "")
    web_results  = state.get("web_search_results", [])

    summary_ctx = state.get("context_summary", "").strip() or "없음"

    web_section = (
        "\n\n[최신 검색 결과]:\n" + "\n".join(f"- {r[:300]}" for r in web_results)
        if web_results else ""
    )

    t0 = time.time()
    prompt = (
        f"질문: {question}\n\n"
        f"[방송 내용 요약]:\n{summary_ctx}"
        f"{web_section}\n\n"
        "위 내용과 LLM 지식을 종합하여 MC가 청취자에게 전달하는 답변을 3줄 이내로 작성하세요. "
        "자연스러운 한국어로, 핵심만 간결하게 작성하세요."
    )

    result = llm.invoke([HumanMessage(content=prompt)])
    mc_script = result.content.strip()

    print(f"[AnswerQuestion] 완료 ({(time.time()-t0):.2f}s): {mc_script[:80]}...")
    return {
        "mc_script":          mc_script,
        "pending_question":   "",
        "web_search_results": [],
        "streaming_stage":    "Main",
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
