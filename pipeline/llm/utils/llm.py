import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# pipeline/llm/utils/llm.py → 4단계 상위가 프로젝트 루트
env_path = Path(__file__).resolve().parent.parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# 디버깅
print(f"--- 환경 변수 로드 체크 ---")
print(f"1. .env 파일 경로: {env_path}")
print(f"2. 발견 여부: {env_path.exists()}")
print(f"3. API 키 확인: {str(os.getenv('OPENAI_API_KEY'))[:10]}...")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)
