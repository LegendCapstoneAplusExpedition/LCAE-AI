def mentor_setup(topics: list[str]) -> dict:
    """방송 전 멘토가 주제 키워드를 입력 → graph.invoke()의 초기 state로 사용."""
    return {
        "messages":         [],
        "question_queue":   [],
        "broadcast_topics": topics,
        "current_topic":    topics[0] if topics else "",
        "context_summary":  "",
        "retrieved_info":   [],
        "streaming_stage":  "Main",
        "intent":             "",
        "cleaned_text":       "",
        "mc_script":          "",
        "listen_summaries":   [],
    }
