import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from llm_client import embeddings, llm

DB_PATH = "chroma_db"

def build_vector_store_from_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)

    vectordb = Chroma.from_texts(
        chunks,
        embeddings,
        persist_directory=DB_PATH
    )

    vectordb.persist()
    return True


def get_retriever():
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    ).as_retriever(search_kwargs={"k": 4})


def get_rag_chain():
    retriever = get_retriever()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return chain