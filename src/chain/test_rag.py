import os
from dotenv import load_dotenv
from src.chain.nodes import knowledge_search_node
from src.chain.state import AgentState

# 1. 환경 변수 로드
load_dotenv()

def test_rag_retrieval():
    print("=== Knowledge Search (RAG) 노드 테스트 시작 ===")
    
    # 2. 가짜 상태(Mock State) 생성
    # Analyzer 노드가 "가우시안 스플래팅 렌더링"이라는 주제를 뽑았다고 가정
    mock_state: AgentState = {
        "messages": [],
        "is_speaking": False,
        "silence_duration": 0.0,
        "question_queue": [],
        "current_topic": "가우시안 스플래팅 렌더링 파이프라인", # 테스트할 검색어
        "context_summary": "",
        "retrieved_info": [],
        "streaming_stage": "Main"
    }

    # 3. 노드 함수 실행
    try:
        result = knowledge_search_node(mock_state)
        
        # 4. 결과 확인
        retrieved = result.get("retrieved_info", [])
        
        print(f"\n검색 키워드: {mock_state['current_topic']}")
        print(f"검색된 지식 조각 개수: {len(retrieved)}")
        
        for i, doc in enumerate(retrieved):
            print(f"\n--- 지식 조각 {i+1} ---")
            print(doc[:200] + "...") # 너무 길면 잘라서 출력
            
    except Exception as e:
        print(f"\n에러 발생: {e}")

if __name__ == "__main__":
    test_rag_retrieval()