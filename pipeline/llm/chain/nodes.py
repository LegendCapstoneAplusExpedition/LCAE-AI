from langchain_core.messages import AIMessage
from langchain_chroma import Chroma
from pipeline.llm.utils.llm import llm
from pipeline.llm.utils.embeddings import embeddings
from pipeline.llm.prompts.persona import SYSTEM_PROMPT
from pipeline.llm.chain.state import AgentState, AnalysisResult, PreprocessResult
import time

_vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 중요도 임계값: 이 값 이상인 구간만 분석에 사용
IMPORTANCE_THRESHOLD = 0.5

def preprocess_node(state: AgentState):
    """
    STT 원문을 의미 단위로 분리하고 중요도를 평가해 핵심 내용만 추출하는 노드.
    중요도 0.5 미만 구간(필러, 추임새, 말 더듬기 등)은 제거한다.
    """
    if not state["messages"]:
        return {"cleaned_text": ""}

    raw_text = state["messages"][-1].content
    print(f"[LLM:Preprocess] 원문: \"{raw_text}\"")

    prompt = f"""아래는 실시간 음성 인식(STT)으로 변환된 한국어 발화입니다.
문장을 의미 단위 구간으로 분리하고, 각 구간의 중요도를 0.0~1.0으로 평가하세요.

중요도 기준:
- 0.0~0.2: 추임새, 감탄사 (아, 어, 음, 에-)
- 0.2~0.4: 말 더듬기, 반복 표현
- 0.4~0.5: 전환 표현 (그니까, 제 말은, 있잖아요 등)
- 0.5~0.7: 부가 설명, 맥락 연결
- 0.7~1.0: 핵심 내용, 질문, 주제 정보

발화:
"{raw_text}"
"""

    structured_llm = llm.with_structured_output(PreprocessResult)
    result: PreprocessResult = structured_llm.invoke([
        ("system", "당신은 한국어 구어체 발화를 분석하는 전문가입니다."),
        ("human", prompt)
    ])

    kept = [seg.text for seg in result.segments if seg.importance >= IMPORTANCE_THRESHOLD]
    cleaned = " ".join(kept).strip()

    print(f"[LLM:Preprocess] 구간 분석:")
    for seg in result.segments:
        flag = "✓" if seg.importance >= IMPORTANCE_THRESHOLD else "✗"
        print(f"  {flag} [{seg.importance:.2f}] \"{seg.text}\"")
    print(f"[LLM:Preprocess] 정제 결과: \"{cleaned}\"")

    return {"cleaned_text": cleaned}


def analyzer_node(state: AgentState):
    """
    멘토의 마지막 발화를 분석하여 주제와 요약을 추출하는 노드
    """

    # 1. 최근 메시지 추출 (preprocess_node가 정제한 텍스트 우선 사용)
    if not state["messages"]:
        return {"current_topic": "대화 시작 전", "context_summary": "대화 없음", "intent": "대기"}

    cleaned = state.get("cleaned_text", "").strip()
    last_message = cleaned if cleaned else state["messages"][-1].content
    prev_summary = state.get("context_summary", "")
    print(f"[LLM:Analyzer] 입력 메시지: \"{last_message}\"")

    # 2. 분석용 프롬프트 구성 (이전 요약을 넣어 누적 컨텍스트 유지)
    user_prompt = f"""
    아래는 실시간 멘토링 중인 멘토의 발화 내용입니다.
    이를 분석하여 주제(topic), 요약(summary), 발화 의도(intent)를 추출하세요.
    주제는 단어로 조합된 키워드(명사구 등)로, 요약은 3줄 이내, 발화 의도는 1줄로 정리하세요.

    {f'[이전까지의 대화 요약]: {prev_summary}' if prev_summary else ''}

    멘토의 발화:
    "{last_message}"
    """

    # 3. LLM 호출 (구조화된 출력 강제)
    # AnalysisResult 클래스가 사전에 정의되어 있어야 함
    structured_llm = llm.with_structured_output(AnalysisResult)

    analysis = structured_llm.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", user_prompt)
    ])

    # 4. 결과 반환
    print(f"[LLM:Analyzer] 주제={analysis.topic} | 의도={analysis.intent}")
    print(f"[LLM:Analyzer] 요약: {analysis.summary}")
    
    return {
        "current_topic": analysis.topic,
        "context_summary": analysis.summary,
        "intent": analysis.intent,
    }

def knowledge_search_node(state: AgentState):
    """
    주제를 바탕으로 Vector DB에서 관련 지식을 검색하는 노드
    """
    topic = state.get("current_topic")

    if not topic or topic == "대화 시작 전":
        return {"retrieved_info": ["관련 지식을 찾을 수 없습니다."]}

    # 유사도 기반 검색 (상위 k개 지문만 가져옴)
    print(f"[LLM:Search] 검색 주제: \"{topic}\"")
    docs = _vector_db.similarity_search(topic, k=2)
    retrieved_docs = [doc.page_content for doc in docs]
    print(f"[LLM:Search] 검색 결과 {len(retrieved_docs)}건")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"[LLM:Search]   [{i}] {doc[:80]}{'...' if len(doc) > 80 else ''}")

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
    elif "질문" in intent:  # 멘토가 질문을 던짐
        should_intervene = True
    elif q_count >= 3:  # 질문이 너무 많이 쌓임
        should_intervene = True

    # 3. 결과 반환
    # 이 결과는 LangGraph의 'Conditional Edge'에서 경로를 정하는 기준이 됨
    decision = "speak" if should_intervene else "wait"
    print(f"[LLM:Decision] silence={silence:.1f}s | intent=\"{intent}\" | q_count={q_count} → {decision}")
    return decision

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
    print(f"[LLM:Writer] 응답 시간: {duration:.2f}초")
    print(f"[LLM:Writer] 출력:\n{response.content}")

    # 4. 결과 반환
    # 생성된 멘트를 메시지 리스트에 추가합니다.
    # (나중에 이 메시지가 TTS의 입력값이 됩니다.)
    return {
        "messages": [AIMessage(content=response.content)],
        "streaming_stage": "Output_Ready"  # 출력이 준비되었다는 상태 표시
    }

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    
    print("[LLM 독립 테스트] nodes.py 단독 실행 검증 시작")
    
    # 1. 입력 상태 데이터(State) 정의
    mock_state = {
        "messages": [HumanMessage(content="가우시안 스플래팅 렌더링 파이프라인 최적화에 대해 설명 중입니다.")],
        "silence_duration": 6.5,  # 5초 이상으로 설정하여 개입 유도
        "question_queue": ["메모리 점유율은 어떻게 줄이나요?"],
        "current_topic": "",
        "context_summary": "",
        "retrieved_info": [],
        "intent": "",
        "cleaned_text": "",
    }
    
    # 2. 첫 번째 분석 노드 단독 테스트
    print("\n--- [1] Analyzer Node 테스트 ---")
    analyzer_result = analyzer_node(mock_state)
    print("결과 수신:", analyzer_result)
    
    # 분석 노드의 결과값을 기존 상태에 업데이트 (누적 시뮬레이션)
    mock_state.update(analyzer_result)
    
    # 3. 두 번째 검색 노드 단독 테스트
    print("\n--- [2] Knowledge Search Node 테스트 ---")
    search_result = knowledge_search_node(mock_state)
    print("결과 수신:", search_result)
    mock_state.update(search_result)
    
    # 4. 세 번째 판단 노드 단독 테스트
    print("\n--- [3] Decision Node 판단 테스트 ---")
    decision_action = decision_node(mock_state)
    print(f"결과 수신: 최종 경로 분기는 -> [{decision_action}] 입니다.")