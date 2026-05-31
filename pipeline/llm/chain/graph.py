from langgraph.graph import StateGraph, START, END
from pipeline.llm.chain.state import AgentState
from pipeline.llm.chain.nodes import (
    preprocess_node,
    knowledge_search_node,
    analyze_write_node,
    decision_node,
    output_node,
)

# 그래프 초기화
workflow = StateGraph(AgentState)

# 노드 등록
workflow.add_node("preprocess",    preprocess_node)
workflow.add_node("search",        knowledge_search_node)
workflow.add_node("analyze_write", analyze_write_node)
workflow.add_node("output",        output_node)

# 고정 엣지
# search를 preprocess 직후에 실행 (cleaned_text로 검색 → topic 추출 전)
workflow.add_edge(START,          "preprocess")
workflow.add_edge("preprocess",   "search")
workflow.add_edge("search",       "analyze_write")

# 조건부 엣지: decision_node가 speak/wait 반환
workflow.add_conditional_edges(
    "analyze_write",
    decision_node,
    {
        "speak": "output",  # mc_script → messages 로 이동 후 종료
        "wait":  END,       # 개입하지 않고 대기
    }
)

workflow.add_edge("output", END)

# 컴파일
app = workflow.compile()
