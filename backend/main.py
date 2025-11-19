import os
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware

from llm_client import llm
from rag import build_vector_store_from_text, get_rag_chain

from insecure_requests import enable_insecure_requests
enable_insecure_requests()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "Backend is running"}

@app.post("/rag")
async def rag_endpoint(
    file_content: str = Form(None),
    question: str = Form(None)
):
    response = {}

    if file_content:
        build_vector_store_from_text(file_content)
        response["index_status"] = "Text indexed successfully"

    if question:
        chain = get_rag_chain()
        result = chain.invoke(question)
        response["answer"] = result["result"]
        response["sources"] = [
            doc.page_content[:200] for doc in result["source_documents"]
        ]

    if not file_content and not question:
        return {"error": "Send either file_content or question or both."}

    return response