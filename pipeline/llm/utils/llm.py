from langchain_ollama import ChatOllama

llm_structured = ChatOllama(
    model       = "driving-mentor",
    temperature = 0.1,
    format      = "json",
)
llm = ChatOllama(
    model       = "driving-mentor",
    temperature = 0.7,
)
llm_summary = ChatOllama(
    model       = "driving-mentor",
    temperature = 0.0,
)
print(f"[LLM] provider=Ollama  model=driving-mentor")
