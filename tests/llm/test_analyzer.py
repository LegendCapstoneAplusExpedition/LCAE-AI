import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from pipeline.llm.chain.nodes import analyzer_node
from pipeline.llm.chain.state import AgentState

# 1. 환경 변수 로드 (LLM 호출을 위해 필수)
load_dotenv()

def test_analyzer():
    print("=== Analyzer 노드 테스트 시작 ===")

    # 2. 가짜 상태(Mock State) 생성
    # 멘토가 가우시안 스플래팅에 대해 설명하는 상황을 가정합니다.
    mock_state: AgentState = {
        "messages": [
            HumanMessage(content="자, 오늘은 가우시안 스플래팅의 렌더링 파이프라인에 대해 설명해줄게요. 핵심은 포인트 클라우드를 가우시안 형태로 표현하는 겁니다.")
        ],
        "is_speaking": True,
        "silence_duration": 0.0,
        "question_queue": [],
        "current_topic": None,
        "context_summary": "",
        "retrieved_info": [],
        "streaming_stage": "Main"
    }

    # 3. 노드 함수 직접 실행
    try:
        result = analyzer_node(mock_state)

        # 4. 결과 확인
        print(f"\n[분석 결과]")
        print(f"추출된 주제(Topic): {result.get('current_topic')}")
        print(f"내용 요약(Summary): {result.get('context_summary')}")

    except Exception as e:
        print(f"\n에러 발생: {e}")

if __name__ == "__main__":
    test_analyzer()
