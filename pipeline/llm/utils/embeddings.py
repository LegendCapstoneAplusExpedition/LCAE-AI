from langchain_ollama import OllamaEmbeddings

# DB에 데이터가 있을 때만 실제 호출됨 (nodes.py의 _db_has_data 체크)
# 데이터 인제스트 시점에 더 빠른 임베더로 교체 예정
embeddings = OllamaEmbeddings(model="nomic-embed-text")
