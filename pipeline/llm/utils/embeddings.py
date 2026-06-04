from langchain_ollama import OllamaEmbeddings
import os

# DB에 데이터가 있을 때만 실제 호출됨 (nodes.py의 _db_has_data 체크)
# 데이터 인제스트 시점에 더 빠른 임베더로 교체 예정
#
# 타임아웃 주입: 긴 발화에서 이 임베딩(nomic-embed-text)이 GPU에 로드될 때,
# VRAM이 부족하면 Ollama가 메인 LLM 모델을 밀어내고 다시 올리는 스래싱이 일어나
# 임베딩/후속 LLM 호출이 무한정 멈출 수 있다. 타임아웃으로 동결을 막는다.
_request_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT_S", "60"))
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    client_kwargs={"timeout": _request_timeout},
)
