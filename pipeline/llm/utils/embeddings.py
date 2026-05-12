# pipeline/llm/utils/embeddings.py
from langchain_openai import OpenAIEmbeddings

# OpenAI의 최신 임베딩 모델 사용
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
