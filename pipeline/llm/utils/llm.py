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
print(f"[LLM] provider=Ollama  model=driving-mentor")
