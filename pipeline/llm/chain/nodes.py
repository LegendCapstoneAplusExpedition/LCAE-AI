from langchain_core.messages import AIMessage
from langchain_chroma import Chroma
from pipeline.llm.utils.llm import llm, llm_structured
from pipeline.llm.utils.embeddings import embeddings
from pipeline.llm.utils.text_cleaner import clean_fillers
from pipeline.llm.prompts.persona import SYSTEM_PROMPT
from pipeline.llm.chain.state import AgentState, AnalysisResult, PreprocessResult
import time

_vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 중요도 임계값: 이 값 이상인 구간만 분석에 사용
IMPORTANCE_THRESHOLD = 0.6

def preprocess_node(state: AgentState):
    """
    STT 원문에서 구어체 필러를 제거한다 (regex 기반, ~0ms).
    재학습 완료 후 LLM 기반 중요도 분석으로 교체 예정.
    """
    if not state["messages"]:
        return {"cleaned_text": ""}

    t0 = time.time()
    raw_text = state["messages"][-1].content
    cleaned = clean_fillers(raw_text)

    print(f"[Preprocess] 원문: \"{raw_text}\"")
    print(f"[Preprocess] 정제: \"{cleaned}\"")
    print(f"[Preprocess] 소요: {(time.time()-t0)*1000:.1f}ms")

    return {"cleaned_text": cleaned}


def analyzer_node(state: AgentState):
    """
    멘토의 마지막 발화를 분석하여 주제와 요약을 추출하는 노드
    """

    # 1. 최근 메시지 추출 (preprocess_node가 정제한 텍스트 우선 사용)
    if not state["messages"]:
        return {"current_topic": "대화 시작 전", "context_summary": "대화 없음", "intent": "대기"}

    t0 = time.time()
    cleaned = state.get("cleaned_text", "").strip()
    last_message = cleaned if cleaned else state["messages"][-1].content
    prev_summary = state.get("context_summary", "")
    print(f"[LLM:Analyzer] 입력: \"{last_message}\"")

    # 2. 분석용 프롬프트 구성 (이전 요약을 넣어 누적 컨텍스트 유지)
    user_prompt = f"""
아래는 드라이빙 멘토링 방송 중 멘토의 발화입니다.
분석하여 주제(topic), 누적 요약(summary), 발화 의도(intent)를 추출하세요.

[발화 의도는 아래 중 하나로만 분류]
- 설명: 멘토가 내용을 설명 중
- 질문: 멘토가 직접 질문을 제시
- 질문요청: 멘토가 AI 진행자에게 멘티 질문 전달이나 질문 생성을 요청 (예: "질문 있어요?", "질문 리스트 정리해줘")
- 정리요청: 멘토가 AI 진행자에게 지금까지 내용 요약을 요청 (예: "지금까지 내용 정리해줘", "요약해줘")
- 마무리: 멘토가 방송을 종료하려는 신호 (예: "오늘 방송 마무리", "이상으로 끝")
- 대기: 특별한 의도 없음

{f'[이전까지의 누적 요약]: {prev_summary}' if prev_summary else ''}

멘토의 발화:
"{last_message}"
"""

    analysis = llm_structured.with_structured_output(AnalysisResult).invoke([
        ("system", SYSTEM_PROMPT),
        ("human", user_prompt)
    ])

    print(f"[LLM:Analyzer] 주제={analysis.topic} | 의도={analysis.intent}")
    print(f"[LLM:Analyzer] 요약: {analysis.summary}")
    print(f"[LLM:Analyzer] 총 소요: {time.time()-t0:.2f}s")

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

    t0 = time.time()
    print(f"[LLM:Search] 검색 주제: \"{topic}\"")
    docs = _vector_db.similarity_search(topic, k=2)
    retrieved_docs = [doc.page_content for doc in docs]
    print(f"[LLM:Search] 검색 결과 {len(retrieved_docs)}건")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"[LLM:Search]   [{i}] {doc[:80]}{'...' if len(doc) > 80 else ''}")
    print(f"[LLM:Search] 총 소요: {time.time()-t0:.2f}s")

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
    elif intent in ("질문", "질문요청", "정리요청", "마무리"):  # 멘토 명시적 신호
        should_intervene = True
    elif q_count >= 3:  # 질문이 너무 많이 쌓임
        should_intervene = True

    # 3. 결과 반환
    # 이 결과는 LangGraph의 'Conditional Edge'에서 경로를 정하는 기준이 됨
    decision = "speak" if should_intervene else "wait"
    print(f"[LLM:Decision] silence={silence:.1f}s | intent=\"{intent}\" | q_count={q_count} → {decision} (즉시 판단)")
    return decision

def script_writer_node(state: AgentState):
    """
    streaming_stage와 intent를 기준으로 4가지 모드 중 하나를 선택해 MC 멘트를 작성합니다.
    - Intro      : 방송 오프닝
    - 질문요청/질문 : 질문 전달 또는 질문 직접 생성
    - 정리요청     : 지금까지 내용 중간 요약
    - 마무리/Outro : 전체 클로징 요약
    - 기본(bridge) : 멘토 발화 흐름 연결
    """
    summary        = state.get("context_summary", "")
    topic          = state.get("current_topic", "")
    knowledge      = "\n".join(state.get("retrieved_info", []))
    intent         = state.get("intent", "")
    stage          = state.get("streaming_stage", "Main")
    question_queue = state.get("question_queue", [])
    last_mentor_msg = state["messages"][-1].content if state["messages"] else ""

    # 모드 선택
    if stage == "Intro":
        writer_prompt = _prompt_intro(topic)
        mode_label = "Intro"
    elif stage == "Outro" or intent == "마무리":
        writer_prompt = _prompt_outro(summary)
        mode_label = "Outro"
    elif intent == "정리요청":
        writer_prompt = _prompt_mid_summary(summary, topic)
        mode_label = "MidSummary"
    elif intent in ("질문", "질문요청") or question_queue:
        writer_prompt = _prompt_question(summary, question_queue, topic)
        mode_label = "Question"
    else:
        writer_prompt = _prompt_bridge(summary, last_mentor_msg, knowledge, topic)
        mode_label = "Bridge"

    t0 = time.time()
    print(f"[LLM:Writer] 모드={mode_label} | LLM 호출 시작")

    response = llm.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", writer_prompt)
    ])

    print(f"[LLM:Writer] 총 소요: {time.time()-t0:.2f}s")
    print(f"[LLM:Writer] 출력:\n{response.content}")

    return {
        "messages": [AIMessage(content=response.content)],
        "streaming_stage": "Output_Ready",
    }


def _prompt_intro(topic: str) -> str:
    return f"""오늘 방송의 주제는 '{topic}'입니다.
청취자들에게 오늘 방송을 소개하는 오프닝 멘트를 1~2문장으로 작성하세요.
멘토를 따뜻하게 소개하고, 오늘 다룰 주제에 대한 기대감을 자연스럽게 전달합니다."""


def _prompt_question(summary: str, question_queue: list, topic: str) -> str:
    q_list = "\n".join(f"- {q}" for q in question_queue) if question_queue else "없음"
    return f"""[지금까지의 내용]: {summary if summary else '방송 초반'}
[현재 주제]: {topic}
[대기 중인 멘티 질문]:
{q_list}

다음 두 가지를 모두 고려하여 지금 멘토에게 전달할 질문 1개를 결정하세요.
1. 대기 중인 멘티 질문 중 현재 대화 흐름과 가장 자연스럽게 이어지는 것
2. 지금까지의 내용을 들은 멘티라면 궁금해할 만한 질문 (직접 생성)

위 두 후보 중 지금 이 시점에 더 적절한 것 하나를 골라 멘토에게 1~2문장으로 전달하세요.
선택한 질문이 멘티 질문이면 "멘티 분이 여쭤봤는데요"와 같이 자연스럽게 소개하고,
직접 생성한 질문이면 "청취자분들이 궁금해하실 것 같은데요"처럼 자연스럽게 연결하세요."""


def _prompt_mid_summary(summary: str, topic: str) -> str:
    return f"""[지금까지의 누적 내용]: {summary if summary else '아직 내용 없음'}
[현재 주제]: {topic}

지금까지의 방송 내용을 청취자를 위해 2~3문장으로 자연스럽게 정리해주세요.
방금 합류한 청취자도 이해할 수 있도록 핵심만 담아 정리합니다."""


def _prompt_outro(summary: str) -> str:
    return f"""[오늘 방송 전체 내용]: {summary if summary else '다양한 주제로 멘토링이 진행되었습니다'}

오늘 방송 전체를 마무리하는 클로징 멘트를 3~4문장으로 작성하세요.
오늘 다룬 핵심 내용을 간략히 짚고, 멘토에게 감사 인사, 청취자에게 마무리 인사를 자연스럽게 포함합니다."""


def _prompt_bridge(summary: str, last_mentor_msg: str, knowledge: str, topic: str) -> str:
    knowledge_line = f"\n[참고 지식]: {knowledge}" if knowledge else ""
    return f"""[현재까지의 내용]: {summary if summary else '방송 진행 중'}
[현재 주제]: {topic}
[멘토의 마지막 발화]: "{last_mentor_msg}"{knowledge_line}

멘토의 발화가 자연스럽게 이어지도록 짧은 브릿지 멘트를 1~2문장으로 작성하세요.
참고 지식이 있다면 청취자 이해를 위해 한 줄 보충 설명을 추가합니다."""

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