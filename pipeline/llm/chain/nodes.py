from langchain_core.messages import AIMessage
from langchain_chroma import Chroma
from pipeline.llm.utils.llm import llm_structured
from pipeline.llm.utils.embeddings import embeddings
from pipeline.llm.utils.text_cleaner import clean_fillers
from pipeline.llm.prompts.persona import SYSTEM_PROMPT
from pipeline.llm.chain.state import AgentState, AnalyzeAndWriteResult
import time

_vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


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
    if not search_text:
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

    q_list       = "\n".join(f"- {q}" for q in question_queue) if question_queue else "없음"
    knowledge_sec = f"\n[검색된 참고 지식]:\n{knowledge}" if knowledge else ""

    prompt = f"""[단계]: {stage}
[이전 요약]: {prev_summary if prev_summary else '없음'}
[대기 질문]: {q_list}{knowledge_sec}
[멘토 발화]: "{last_message}"
"""

    t0 = time.time()
    print(f"[AnalyzeWrite] LLM 호출 시작  (stage={stage})")

    result: AnalyzeAndWriteResult = llm_structured.with_structured_output(
        AnalyzeAndWriteResult
    ).invoke([
        ("system", SYSTEM_PROMPT),
        ("human", prompt),
    ])

    elapsed = time.time() - t0
    print(f"[AnalyzeWrite] 완료  ({elapsed:.2f}s)")
    print(f"[AnalyzeWrite] topic={result.topic} | intent={result.intent}")
    print(f"[AnalyzeWrite] summary: {result.summary}")
    print(f"[AnalyzeWrite] mc_script: \"{result.mc_script[:80]}{'...' if len(result.mc_script) > 80 else ''}\"")

    return {
        "current_topic":   result.topic,
        "context_summary": result.summary,
        "intent":          result.intent,
        "mc_script":       result.mc_script,
    }


# ──────────────────────────────────────────────
# decision_node  (즉시, 규칙 기반 라우팅 함수)
# ──────────────────────────────────────────────
def decision_node(state: AgentState) -> str:
    silence  = state.get("silence_duration", 0)
    intent   = state.get("intent", "")
    q_count  = len(state.get("question_queue", []))

    if silence >= 5.0:
        should_speak = True
    elif intent in ("질문", "질문요청", "정리요청", "마무리"):
        should_speak = True
    elif q_count >= 3:
        should_speak = True
    else:
        should_speak = False

    decision = "speak" if should_speak else "wait"
    print(f"[Decision] silence={silence:.1f}s | intent=\"{intent}\" | q_count={q_count} → {decision}")
    return decision


# ──────────────────────────────────────────────
# output_node  (즉시, mc_script → messages)
# ──────────────────────────────────────────────
def output_node(state: AgentState):
    mc_script = state.get("mc_script", "").strip()

    if not mc_script:
        print("[Output] mc_script 없음 → 발화 생략")
        return {"streaming_stage": "Output_Ready"}

    print(f"[Output] 최종 멘트: \"{mc_script}\"")
    return {
        "messages":        [AIMessage(content=mc_script)],
        "streaming_stage": "Output_Ready",
    }


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    print("[nodes.py 단독 테스트]")
    mock_state = {
        "messages":        [HumanMessage(content="계약서에서 납품 범위를 명확히 해야 나중에 분쟁이 없어요. 질문 받을게요.")],
        "cleaned_text":    "계약서에서 납품 범위를 명확히 해야 나중에 분쟁이 없어요. 질문 받을게요.",
        "silence_duration": 6.0,
        "question_queue":  ["계약서 필수 항목이 뭔가요?"],
        "current_topic":   "",
        "context_summary": "프리랜서 전환 전략과 포트폴리오 구성에 대해 논의했다.",
        "retrieved_info":  [],
        "intent":          "",
        "mc_script":       "",
        "streaming_stage": "Main",
        "is_speaking":     False,
    }

    print("\n--- [1] preprocess ---")
    mock_state.update(preprocess_node(mock_state))

    print("\n--- [2] search ---")
    mock_state.update(knowledge_search_node(mock_state))

    print("\n--- [3] analyze_write ---")
    mock_state.update(analyze_write_node(mock_state))

    print("\n--- [4] decision ---")
    print("결과:", decision_node(mock_state))
