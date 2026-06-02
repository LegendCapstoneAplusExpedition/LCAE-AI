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

_READY_SUMMARY_PATH  = Path(__file__).parent.parent.parent / "listenlist" / "ready_summary.json"
_READY_QUESTION_PATH = Path(__file__).parent.parent.parent / "listenlist" / "ready_question.json"

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

    if stage == "Outro" and intent == "마무리":
        print(f"[Decision] Outro 진입 완료 → 추가 클로징 차단")
        return "wait"

    if intent == "정리요청":
        print(f"[Decision] intent=\"{intent}\" → summarize")
        return "summarize"

    if intent == "질문요청":
        print(f"[Decision] intent=\"{intent}\" → ask_question")
        return "ask_question"

    should_speak = intent == "마무리"
    decision = "speak" if should_speak else "wait"
    print(f"[Decision] intent=\"{intent}\" → {decision}")
    return decision


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
    return {
        "mc_script": mc_script,
        "pending_question": question,
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
