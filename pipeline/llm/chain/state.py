from typing import Annotated, TypedDict, List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# State(추후 필요 시 다중 스키마 구현)
class AgentState(TypedDict):
    # 대화 기록
    messages: Annotated[list[BaseMessage],add_messages]

    # 질문 큐레이션(멘티)
    question_queue: List[Dict[str,any]]

    # RAG 연동
    current_topic: Optional[str]        # 현재 주제 키워드
    context_summary: str                # 방송 내용 요약
    retrieved_info: List[str]           # 검색된 전문 지식

    # 방송 사전 설정 (mentor_setup에서 주입)
    broadcast_topics: List[str]      # 멘토가 사전에 입력한 주제 키워드 목록

    # 방송 상태
    streaming_stage: str            # Main, QnA, Outro (Intro는 외부 화면 처리)

    # 분석 결과
    intent: str          # 멘토 발화 의도 (분석 노드에서 추출)

    # 전처리 결과
    cleaned_text: str    # 필러 제거 후 텍스트

    # 분석+작성 통합 결과
    mc_script: str       # analyze_write_node가 생성한 MC 멘트 (output_node에서 messages로 이동)


# 분석+작성 통합 구조화 출력 스키마 (LLM 1회 호출로 분석과 멘트 생성을 동시에 처리)
class AnalyzeAndWriteResult(BaseModel):
    topic:     str = Field(description="현재 대화의 핵심 키워드나 주제 (명사구 2~5단어)")
    summary:   str = Field(description="이전 요약에 이번 발화 내용을 더한 누적 요약 (3줄 이내 텍스트). 반드시 최소 1줄 이상 작성. 절대 빈 문자열 불가.")
    intent:    str = Field(description="설명 / 질문 / 질문요청 / 정리요청 / 마무리 / 대기 중 정확히 하나")
    mc_script: str = Field(description="AI MC가 발화할 텍스트. 발화 불필요 시 빈 문자열 \"\". 반드시 순수 문자열이어야 함")
