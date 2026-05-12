from langgraph.graph import StateGraph, START, END
from pipeline.llm.chain.state import AgentState
from pipeline.llm.chain.nodes import (
    analyzer_node,
    knowledge_search_node,
    decision_node,
    script_writer_node
)

# 그래프 초기화
workflow = StateGraph(AgentState)

# 노드 등록
workflow.add_node("analyzer", analyzer_node)
workflow.add_node("search", knowledge_search_node)
workflow.add_node("writer", script_writer_node)

# 고정 엣지 연결 (Fixed Edges)
workflow.add_edge(START, "analyzer")
workflow.add_edge("analyzer", "search")

# 조건부 엣지 설정 (Conditional Edges)
workflow.add_conditional_edges(
    "search",           # 판단을 내릴 기준이 되는 이전 노드
    decision_node,      # 실행할 판단 함수
    {
        "speak": "writer",  # 함수 리턴값이 speak이면 대사 작성으로 이동
        "wait": END         # 함수 리턴값이 wait이면 루프를 종료하고 대기
    }
)

# 대사 작성이 완료되면 이번 메시지에 대한 처리를 종료합니다.
workflow.add_edge("writer", END)

# 실행 가능한 앱으로 컴파일
app = workflow.compile()
