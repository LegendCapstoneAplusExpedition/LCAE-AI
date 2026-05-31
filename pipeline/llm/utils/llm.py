'''
LLM_PROVIDER 환경변수로 백엔드 전환:
  $env:LLM_PROVIDER="groq"   → Groq API (llama-3.1-8b-instant, ~1~2s)
  $env:LLM_PROVIDER="ollama" → 로컬 Ollama driving-mentor (기본값)

Groq 사용 시 GROQ_API_KEY 환경변수 필요:
  $env:GROQ_API_KEY="gsk_..."
'''

import os

_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

if _PROVIDER == "groq":
    from langchain_groq import ChatGroq

    llm_structured = ChatGroq(
        model       = "llama-3.1-8b-instant",
        temperature = 0.1,
    )
    llm = ChatGroq(
        model       = "llama-3.1-8b-instant",
        temperature = 0.7,
    )
    print(f"[LLM] provider=Groq  model=llama-3.1-8b-instant")

else:
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
