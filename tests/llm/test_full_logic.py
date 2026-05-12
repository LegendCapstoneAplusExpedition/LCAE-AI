from pipeline.llm.chain.nodes import script_writer_node

def test_logic_flow():
    # 1. 시뮬레이션용 가짜 데이터 준비
    mock_state = {
        "context_summary": "가우시안 스플래팅의 렌더링 방식에 대한 설명 중",
        "retrieved_info": ["가우시안 스플래팅은 타일 기반 래스터화를 사용하여 실시간 100FPS 이상 구현 가능함"],
        "messages": []
    }

    # 2. 노드 실행
    result = script_writer_node(mock_state)

    # 3. 결과 출력 (TTS 대신 눈으로 확인)
    print("\n[AI MC의 최종 멘트]")
    print(result["messages"][0].content)

test_logic_flow()
