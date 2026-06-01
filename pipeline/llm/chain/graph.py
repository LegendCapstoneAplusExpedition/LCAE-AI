from langgraph.graph import StateGraph, START, END
from pipeline.llm.chain.state import AgentState
from pipeline.llm.chain.nodes import (
    preprocess_node,
    knowledge_search_node,
    analyze_write_node,
    fast_intent_check,
    decision_node,
    summarize_listenlist_node,
    generate_question_node,
    output_node,
)

workflow = StateGraph(AgentState)

# ── 노드 등록 ────────────────────────────────────────────────────────────────
workflow.add_node("preprocess",           preprocess_node)
workflow.add_node("search",               knowledge_search_node)
workflow.add_node("analyze_write",        analyze_write_node)
workflow.add_node("summarize_listenlist", summarize_listenlist_node)
workflow.add_node("generate_question",    generate_question_node)
workflow.add_node("output",               output_node)

# ── 고정 엣지 ────────────────────────────────────────────────────────────────
workflow.add_edge(START,        "preprocess")
workflow.add_edge("preprocess", "search")

# search 이후 키워드로 정리요청·질문요청 감지 시 LLM 없이 즉시 분기
workflow.add_conditional_edges(
    "search",
    fast_intent_check,
    {
        "summarize": "summarize_listenlist",
        "question":  "generate_question",
        "analyze":   "analyze_write",
    }
)

# ── analyze_write 이후 분기 ──────────────────────────────────────────────────
workflow.add_conditional_edges(
    "analyze_write",
    decision_node,
    {
        "summarize":    "summarize_listenlist",
        "ask_question": "generate_question",
        "speak":        "output",
        "wait":         END,
    },
)

# ── 요약·질문 생성 경로 → output ────────────────────────────────────────────
workflow.add_edge("summarize_listenlist", "output")
workflow.add_edge("generate_question",   "output")

workflow.add_edge("output", END)

app = workflow.compile()
