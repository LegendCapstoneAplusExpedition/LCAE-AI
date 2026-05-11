import time
from langchain_core.messages import HumanMessage
from src.chain.graph import app

def run_realtime_loop():
    print("AI MC 실시간 루프 가동 (종료하려면 'exit' 입력)")
    
    # 세션 동안 유지될 초기 상태
    state = {
        "messages": [],
        "silence_duration": 0.0,
        "question_queue": [],
        "current_topic": "",
        "retrieved_info": [],
        "intent": ""
    }

    while True:
        # 1. 멘토의 발화 입력 받기 (STT 모킹)
        user_input = input("\n🎤 멘토 (또는 'exit'): ")
        if user_input.lower() == 'exit':
            break

        # 2. 상태 업데이트
        state["messages"].append(HumanMessage(content=user_input))
        
        # [테스트용] 입력 후에 6초가 지났다고 가정 (침묵 시간 시뮬레이션)
        state["silence_duration"] = 6.0 

        # 3. 그래프 실행
        print("... AI MC가 상황을 판단 중입니다 ...")
        result = app.invoke(state)
        
        # 4. 결과 출력 및 상태 동기화
        if len(result["messages"]) > len(state["messages"]):
            last_msg = result["messages"][-1]
            print(f"\nAI MC: {last_msg.content}")
            
            # AI의 대답도 메시지 내역에 추가하여 문맥 유지
            state["messages"] = result["messages"]
        else:
            print("\n(AI MC는 현재 경청 중입니다)")
            
        # 5. 다음 턴을 위해 분석 결과 등 초기화 (필요 시)
        state["intent"] = result.get("intent", "")
        state["current_topic"] = result.get("current_topic", "")

if __name__ == "__main__":
    run_realtime_loop()