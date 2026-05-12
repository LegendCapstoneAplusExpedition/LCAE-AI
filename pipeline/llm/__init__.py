"""
LLM (LangGraph) pipeline — AI MC 브릿지 멘트 생성

주요 인터페이스:
    from pipeline.llm import app                  # LangGraph 컴파일된 앱
    from pipeline.llm.chain.state import AgentState
    from pipeline.llm.chain.graph import app

데이터 흐름:
    AgentState → analyzer_node → knowledge_search_node
               → decision_node → (speak) script_writer_node → AIMessage
                               → (wait)  END
"""

from pipeline.llm.chain.graph import app

__all__ = ["app"]
