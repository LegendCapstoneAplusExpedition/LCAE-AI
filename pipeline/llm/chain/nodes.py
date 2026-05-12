from langchain_core.messages import AIMessage
from pipeline.llm.utils.llm import llm
from pipeline.llm.prompts.persona import SYSTEM_PROMPT
from pipeline.llm.chain.state import AgentState, AnalysisResult
import time

def analyzer_node(state: AgentState):
    """
    멘토의 마지막 발화를 분석하여 주제와 요약을 추출하는 노드
    """

    # 1. 최근 메시지 추출 (메시지가 없을 경우를 대비한 예외 처리)
    if not state["messages"]:
        return {"current_topic": "대화 시작 전", "context_summary": "대화 없음"}

    last_message = state["messages"][-1].content

    # 2. 분석용 프롬프트 구성
    # 지침서와 현재 상황을 LLM에게 전달
    user_prompt = f"""
    아래는 실시간 멘토링 중인 멘토의 발화 내용입니다.
    이를 분석하여 주제(topic), 요약(summary), 발화 의도(intent)를 추출하세요.

    멘토의 발화:
    "{last_message}"
    """

    # 3. LLM 호출 (지능 주입)
    # .with_structured_output을 사용하여 AnalysisResult 규격에 맞는 객체를 받음
    structured_llm = llm.with_structured_output(AnalysisResult)

    # 시스템 지침(Persona)과 사용자 요청을 결합하여 전달
    analysis = structured_llm.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", user_prompt)
    ])

    # 4. 결과 반환 (State 업데이트)
    # 리턴된 딕셔너리의 키 값들이 AgentState의 해당 필드들을 자동으로 갱신
    return {
        "current_topic": analysis.topic,
        "context_summary": analysis.summary,
        "intent": analysis.intent,
    }

from langchain_chroma import Chroma
from pipeline.llm.utils.embeddings import embeddings

def knowledge_search_node(state: AgentState):
    """
    주제를 바탕으로 Vector DB에서 관련 지식을 검색하는 노드
    """
    topic = state.get("current_topic")

    if not topic or topic == "대화 시작 전":
        return {"retrieved_info": ["관련 지식을 찾을 수 없습니다."]}

    # 1. 기존에 생성된 Chroma DB 연결 (경로는 프로젝트에 맞게 수정)
    vector_db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    # 2. 유사도 기반 검색 (상위 k개 지문만 가져옴)
    docs = vector_db.similarity_search(topic, k=2)
    retrieved_docs = [doc.page_content for doc in docs]

    # 3. 상태 업데이트
    return {"retrieved_info": retrieved_docs}


def decision_node(state: AgentState):
    """
    현재 상황을 보고 AI MC가 개입할지(speak), 더 기다릴지(wait) 결정합니다.
    """

    # 1. 상태 데이터 가져오기
    silence = state.get("silence_duration", 0)
    intent = state.get("intent", "")  # Analyzer에서 뽑은 의도
    q_count = len(state.get("question_queue", []))

    # 2. 규칙 기반 판단 (Rule-based)
    # 멘토가 대화를 끝냈거나, 5초 이상 침묵할 때 개입 고려
    should_intervene = False

    if silence >= 5.0:  # 5초 이상 정적
        should_intervene = True
    elif "question" in intent.lower():  # 멘토가 질문을 던짐
        should_intervene = True
    elif q_count >= 3:  # 질문이 너무 많이 쌓임
        should_intervene = True

    # 3. 결과 반환
    # 이 결과는 LangGraph의 'Conditional Edge'에서 경로를 정하는 기준이 됨
    if should_intervene:
        return "speak"
    else:
        return "wait"

def script_writer_node(state: AgentState):
    """
    분석된 맥락과 검색된 지식을 바탕으로 AI MC의 브릿지 멘트를 작성합니다.
    """
    # 1. 재료 모으기
    summary = state.get("context_summary", "진행 중인 대화")
    topic = state.get("current_topic", "관련 주제")
    knowledge = "\n".join(state.get("retrieved_info", []))
    last_mentor_msg = state["messages"][-1].content if state["messages"] else ""

    # 2. 페르소나를 녹여낸 프롬프트 구성
    # 단순히 정보를 전달하는 게 아니라 '아나운서'로서의 역할 강조
    writer_prompt = f"""
    {SYSTEM_PROMPT}

    [현재 상황 요약]: {summary}
    [멘토의 마지막 발화]: "{last_mentor_msg}"
    [참고할 전문 지식]:
    {knowledge}

    위 상황을 바탕으로 AI MC의 개입 멘트를 작성하세요.
    - 멘토의 설명을 자연스럽게 요약하며 공감할 것.
    - 참고 지식을 활용해 멘티들이 이해하기 쉽게 한 문장 정도 보충 설명을 더할 것.
    - 마지막에는 멘토에게 다음 설명을 부탁하거나, 멘티의 질문을 전달하며 대화를 이어갈 것.
    """

    start_time = time.time()  # 응답 시간 측정 시작
    # 3. LLM 호출
    # 여기서는 '자연스러운 대사(Text)'가 중요하므로 invoke 사용
    response = llm.invoke([
        ("system", "당신은 멘토링 방송의 전문 MC입니다. 품격 있고 매끄러운 진행 멘트를 작성하세요."),
        ("human", writer_prompt)
    ])

    end_time = time.time()  # 응답 시간 측정 종료
    duration = end_time - start_time  # 응답 시간 계산
    print(f"⏱Script Writer 노드 응답 시간: {duration:.2f}초")

    # 4. 결과 반환
    # 생성된 멘트를 메시지 리스트에 추가합니다.
    # (나중에 이 메시지가 TTS의 입력값이 됩니다.)
    return {
        "messages": [AIMessage(content=response.content)],
        "streaming_stage": "Output_Ready"  # 출력이 준비되었다는 상태 표시
    }
