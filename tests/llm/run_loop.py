"""
AI MC 실시간 루프 테스트 — LLM 파이프라인 단독 실행용
STT/TTS 없이 텍스트 입력으로 파이프라인 동작을 확인합니다.

실행:
    uv run python -m tests.llm.run_loop
"""

from langchain_core.messages import HumanMessage, AIMessage
from pipeline.llm.chain.nodes import (
    preprocess_node, knowledge_search_node, analyze_write_node, decision_node, output_node
)
from pipeline.llm.chain.setup import mentor_setup


def run_realtime_loop():
    topics = ["주니어 성장", "번아웃 예방", "MVP 전략"]
    state = mentor_setup(topics)
    print(f"AI MC 실시간 루프 가동 | topics={topics}")
    print("종료하려면 'exit' 입력\n")

    while True:
        user_input = input("멘토: ").strip()
        if user_input.lower() == "exit":
            break
        if not user_input:
            continue

        state["messages"] = [HumanMessage(content=user_input)]

        state.update(preprocess_node(state))
        state.update(knowledge_search_node(state))
        state.update(analyze_write_node(state))

        decision = decision_node(state)
        if decision == "speak":
            state.update(output_node(state))
            msgs = state.get("messages", [])
            last = msgs[-1] if msgs else None
            if isinstance(last, AIMessage):
                print(f"\nAI MC: {last.content}\n")
        else:
            print("(AI MC 경청 중)\n")


if __name__ == "__main__":
    run_realtime_loop()
