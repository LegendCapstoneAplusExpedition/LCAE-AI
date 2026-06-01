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

    # 발화 페이싱 (진행자가 무지성으로 떠들지 않도록 제어)
    silence_duration: float    # 직전 발화 후 멘토 침묵 시간(초)
    last_ai_speech_ts: float   # AI가 마지막으로 발화한 시각(time.time()). 쿨다운 계산용




# 분석+작성 통합 구조화 출력 스키마 (LLM 1회 호출로 분석과 멘트 생성을 동시에 처리)
class AnalyzeAndWriteResult(BaseModel):
    topic:     str = Field(description="현재 대화의 핵심 키워드나 주제 (명사구 2~5단어)")
    intent:    str = Field(description="설명 / 질문 / 질문요청 / 정리요청 / 마무리 / 대기 중 정확히 하나")
    mc_script: str = Field(description="AI MC가 발화할 텍스트. 발화 불필요 시 빈 문자열 \"\". 반드시 순수 문자열이어야 함")
