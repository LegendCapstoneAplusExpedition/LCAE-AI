import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from src.chain.nodes import analyzer_node, knowledge_search_node, script_writer_node

load_dotenv()

def run_integrated_test():
    print("AI MC 지능 파이프라인 통합 테스트 시작\n")

    # [Step 1] 가짜 멘토의 발화 입력
    state = {
        "messages": [HumanMessage(content="가우시안 스플래팅은 결국 수많은 타원체들을 쌓아서 장면을 렌더링하는 방식이에요.")],
        "retrieved_info": [],
        "current_topic": "",
        "context_summary": ""
    }
    print(f"멘토: {state['messages'][0].content}")

    # [Step 2] Analyzer 실행
    print("\n단계 1: 상황 분석 중...")
    analysis_update = analyzer_node(state)
    state.update(analysis_update)
    print(f"주제 추출: {state['current_topic']}")

    # [Step 3] Knowledge Search 실행
    print("\n단계 2: 관련 지식 검색 중...")
    search_update = knowledge_search_node(state)
    state.update(search_update)
    print(f"검색된 지식: {state['retrieved_info'][0][:50]}...")

    # [Step 4] Script Writer 실행
    print("\n단계 3: MC 대사 작성 중...")
    script_update = script_writer_node(state)
    final_mc_ment = script_update["messages"][0].content

    print("\n" + "="*50)
    print(f"AI MC 최종 멘트:\n\n{final_mc_ment}")
    print("="*50)

if __name__ == "__main__":
    run_integrated_test()