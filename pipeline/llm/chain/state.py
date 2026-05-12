from typing import Annotated, TypedDict, List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

# State(추후 필요 시 다중 스키마 구현)
class AgentState(TypedDict):
    # 대화 기록
    messages: Annotated[list[BaseMessage],add_messages]

    # 실시간 음성 상태
    is_speaking: bool           # 멘토 발화 여부
    silence_duration: float     # 침묵 시간(초)

    # 질문 큐레이션(멘티)
    question_queue: List[Dict[str,any]]

    # RAG 연동
    current_topic: Optional[str]        # 현재 주제 키워드
    context_summary: str                # 방송 내용 요약
    retrieved_info: List[str]           # 검색된 전문 지식

    # 방송 상태
    streaming_stage: str            # Intro, Main, QnA, Outro

    # 분석 결과
    intent: str          # 멘토 발화 의도 (분석 노드에서 추출)


# LLM의 구조화된 출력을 위한 스키마 (추가할 부분)
class AnalysisResult(BaseModel):
    topic: str = Field(description="현재 대화의 핵심 키워드나 주제")
    summary: str = Field(description="현재까지의 대화 내용을 한 줄로 요약")
    intent: str = Field(description="멘토의 발화 의도 (설명, 질문, 인사 등)")
