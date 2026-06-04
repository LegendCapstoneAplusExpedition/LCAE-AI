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
    generate_curated_bridge_node,
    generate_closing_node,
    output_node,
)

workflow = StateGraph(AgentState)

# ── 노드 등록 ────────────────────────────────────────────────────────────────
workflow.add_node("preprocess",           preprocess_node)
workflow.add_node("search",               knowledge_search_node)
workflow.add_node("analyze_write",        analyze_write_node)
workflow.add_node("summarize_listenlist", summarize_listenlist_node)
workflow.add_node("generate_question",    generate_question_node)
workflow.add_node("curated_bridge",       generate_curated_bridge_node)
workflow.add_node("generate_closing",     generate_closing_node)
workflow.add_node("output",               output_node)

# ── 고정 엣지 ────────────────────────────────────────────────────────────────
workflow.add_edge(START,        "preprocess")

# preprocess 이후 확정 가능한 핵심 반응은 검색·LLM 호출 없이 즉시 분기
workflow.add_conditional_edges(
    "preprocess",
    fast_intent_check,
    {
        "summarize": "summarize_listenlist",
        "question":  "generate_question",
        "bridge":    "curated_bridge",
        "closing":   "generate_closing",
        "analyze":   "search",
        "wait":      END,
    }
)

# 확정 규칙이 없는 일반 발화만 지식 검색 후 LLM으로 분석
workflow.add_edge("search", "analyze_write")

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
workflow.add_edge("curated_bridge",      "output")
workflow.add_edge("generate_closing",    "output")

workflow.add_edge("output", END)

app = workflow.compile()
