from langchain_ollama import ChatOllama
import os

_model = os.getenv("LLM_MODEL", "driving-mentor")
_temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
_num_ctx = int(os.getenv("LLM_NUM_CTX", "2048"))
_num_predict = int(os.getenv("LLM_NUM_PREDICT", "120"))

llm_structured = ChatOllama(
    model       = _model,
    temperature = _temperature,
    format      = "json",
    num_ctx     = _num_ctx,
    num_predict = _num_predict,
)
llm = ChatOllama(
    model       = _model,
    temperature = 0.7,
    num_ctx     = _num_ctx,
    num_predict = _num_predict,
)
llm_summary = ChatOllama(
    model       = _model,
    temperature = 0.0,
    num_ctx     = _num_ctx,
    num_predict = _num_predict,
)
print(f"[LLM] provider=Ollama  model={_model}")
