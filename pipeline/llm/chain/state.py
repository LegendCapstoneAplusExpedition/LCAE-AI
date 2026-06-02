from typing import Annotated, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# State(추후 필요 시 다중 스키마 구현)
class AgentState(TypedDict):
    # 대화 기록
    messages: Annotated[list[BaseMessage], add_messages]

    # RAG 연동
    current_topic: str        # 현재 주제 키워드
    retrieved_info: list[str] # 검색된 전문 지식

    # 방송 사전 설정 (mentor_setup에서 주입)
    broadcast_id: str                # 세션 식별자 — 동시 방송 시 파일 경로 격리에 사용
    broadcast_topics: list[str]      # 멘토가 사전에 입력한 주제 키워드 목록

    # 방송 상태
    streaming_stage: str            # Main, QnA, Outro (Intro는 외부 화면 처리)
    silence_duration: float         # STT가 전달한 직전 침묵 길이

    # 분석 결과
    intent: str          # 멘토 발화 의도 (분석 노드에서 추출)
    context_summary: str # 현재는 기록/호환용. 요약 발화는 ready_summary.json 기준

    # 전처리 결과
    cleaned_text: str    # 필러 제거 후 텍스트

    # 분석+작성 통합 결과
    mc_script: str       # analyze_write_node가 생성한 MC 멘트 (output_node에서 messages로 이동)
    pending_question: str # 현재 전달/대기 중인 청취자 질문 텍스트

    # 발화 페이싱 (진행자가 무지성으로 떠들지 않도록 제어)
    last_ai_speech_ts: float   # AI가 마지막으로 발화한 시각(time.time()). 쿨다운 계산용

# 분석+작성 통합 구조화 출력 스키마 (LLM 1회 호출로 분석과 멘트 생성을 동시에 처리)
class AnalyzeAndWriteResult(BaseModel):
    topic:     str = Field(description="현재 대화의 핵심 키워드나 주제 (명사구 2~5단어)")
    intent:    str = Field(description="설명 / 질문 / 질문요청 / 정리요청 / 마무리 / 대기 중 정확히 하나")
    mc_script: str = Field(description="AI MC가 발화할 텍스트. 발화 불필요 시 빈 문자열 \"\". 반드시 순수 문자열이어야 함")
