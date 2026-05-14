import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# pipeline/llm/utils/llm.py → 4단계 상위가 프로젝트 루트
env_path = Path(__file__).resolve().parent.parent.parent.parent / '.env.local'
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("LLM_API_KEY")
# LangChain 내부에서 OPENAI_API_KEY를 찾는 모든 곳(embeddings 등)에서 공유되도록 설정
os.environ["OPENAI_API_KEY"] = api_key or ""

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "gpt-5.4-mini"),
    temperature=0.7,
    api_key=api_key,
)
