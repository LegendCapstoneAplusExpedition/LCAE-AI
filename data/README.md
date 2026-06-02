# RAG Data

RAG 인덱싱용 텍스트 문서를 두는 폴더입니다.

`pipeline.llm.chain.ingest_data`는 이 폴더의 `*.txt` 파일을 읽어 `chroma_db/`에 인덱싱합니다. LLM 학습 데이터는 `llm_training/datasets/`에서 관리합니다.
