import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src.utils.embeddings import embeddings

def ingest_text_documents():
    # 1. 데이터 경로 설정 (data 폴더 내의 모든 .txt 파일)
    data_path = "./data"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"{data_path} 폴더가 없어 생성했습니다. 여기에 .txt 파일을 넣어주세요.")
        return

    print("데이터 로딩 중...")
    loader = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    if not documents:
        print("data 폴더에 텍스트 파일이 없습니다.")
        return

    # 2. 텍스트 분할 (Chunking)
    # 문맥 유지를 위해 500자 단위로 자르고, 50자씩 겹치게(overlap) 설정합니다.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)
    print(f"총 {len(texts)}개의 텍스트 조각으로 분할되었습니다.")

    # 3. 벡터 DB 생성 및 저장 (Chroma)
    print("벡터 DB 저장 중 (이 작업은 시간이 조금 걸릴 수 있습니다)...")
    vector_db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print("인덱싱 완료! 이제 ./chroma_db 폴더에 데이터가 저장되었습니다.")

if __name__ == "__main__":
    ingest_text_documents()