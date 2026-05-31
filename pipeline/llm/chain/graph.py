from langgraph.graph import StateGraph, START, END
from pipeline.llm.chain.state import AgentState
from pipeline.llm.chain.nodes import (
    preprocess_node,
    knowledge_search_node,
    analyze_write_node,
    decision_node,
    decision_search_node,
    summarize_listenlist_node,
    generate_question_node,
    assess_search_node,
    tavily_search_node,
    answer_question_node,
    output_node,
)

workflow = StateGraph(AgentState)

# ── 노드 등록 ────────────────────────────────────────────────────────────────
workflow.add_node("preprocess",           preprocess_node)
workflow.add_node("search",               knowledge_search_node)
workflow.add_node("analyze_write",        analyze_write_node)
workflow.add_node("summarize_listenlist", summarize_listenlist_node)
workflow.add_node("generate_question",    generate_question_node)
workflow.add_node("assess_search",        assess_search_node)
workflow.add_node("tavily_search",        tavily_search_node)
workflow.add_node("answer_question",      answer_question_node)
workflow.add_node("output",               output_node)

# ── 고정 엣지 ────────────────────────────────────────────────────────────────
workflow.add_edge(START,             "preprocess")
workflow.add_edge("preprocess",      "search")
workflow.add_edge("search",          "analyze_write")

# ── analyze_write 이후 5-way 분기 ───────────────────────────────────────────
# summarize    : "정리요청" → summary.jsonl 1줄 요약
# ask_question : "질문요청" → 채팅DB or summary 기반 질문 생성
# answer_question: QnA 스테이지 + pending_question → 웹 검색 여부 판단
# speak        : "마무리" → 기존 output 경로
# wait         : 그 외 → 개입 없이 대기
workflow.add_conditional_edges(
    "analyze_write",
    decision_node,
    {
        "summarize":       "summarize_listenlist",
        "ask_question":    "generate_question",
        "answer_question": "assess_search",
        "speak":           "output",
        "wait":            END,
    },
)

# ── QnA 답변 서브그래프 ──────────────────────────────────────────────────────
# assess_search → (search → tavily_search → answer_question) or (direct → answer_question)
workflow.add_conditional_edges(
    "assess_search",
    decision_search_node,
    {
        "search": "tavily_search",
        "direct": "answer_question",
    },
)
workflow.add_edge("tavily_search",        "answer_question")
workflow.add_edge("answer_question",      "output")

# ── 요약·질문 생성 경로 → output ────────────────────────────────────────────
workflow.add_edge("summarize_listenlist", "output")
workflow.add_edge("generate_question",   "output")

workflow.add_edge("output", END)

app = workflow.compile()
