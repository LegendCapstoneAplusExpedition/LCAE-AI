# pipeline/llm/utils/embeddings.py
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

env_path = Path(__file__).resolve().parent.parent.parent.parent / '.env.local'
load_dotenv(dotenv_path=env_path)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("LLM_API_KEY"),
)
