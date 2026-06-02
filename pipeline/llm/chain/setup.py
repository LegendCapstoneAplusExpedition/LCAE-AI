from pipeline.llm.chain.state import AgentState


def mentor_setup(topics: list[str], broadcast_id: str = "") -> AgentState:
    """방송 전 멘토가 주제 키워드를 입력 → graph.invoke()의 초기 state로 사용."""
    return {
        "messages":         [],
        "broadcast_topics": topics,
        "broadcast_id":     broadcast_id,
        "current_topic":    topics[0] if topics else "",
        "retrieved_info":   [],
        "streaming_stage":  "Main",
        "silence_duration": 0.0,
        "intent":             "",
        "context_summary":    "",
        "cleaned_text":       "",
        "mc_script":          "",
        "pending_question":   "",
        "last_ai_speech_ts":  0.0,
    }
