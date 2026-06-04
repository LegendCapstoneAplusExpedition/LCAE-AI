"""
LLM (LangGraph) pipeline — AI MC 브릿지 멘트 생성

주요 인터페이스:
    from pipeline.llm import app                  # LangGraph 컴파일된 앱
    from pipeline.llm.chain.state import AgentState
    from pipeline.llm.chain.graph import app

데이터 흐름:
    AgentState → preprocess_node → fast_intent_check
               → 확정 반응(브릿지/질문/요약/마무리) → AIMessage
               → 일반 발화 → knowledge_search_node → analyze_write_node
"""

from pipeline.llm.chain.graph import app

__all__ = ["app"]
