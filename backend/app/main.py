# backend/app/main.py
import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

load_dotenv()

# LangChain & helpers
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAl, OpenAlEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Basic config from env
API_BASE = os.getenv("GENAI_BASE")
API_KEY = os.getenv("GENAI_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "azure_ai/genailab-maas-DeepSeek-V3-0324")
EMB_MODEL = os.getenv("EMB_MODEL", "azure/genailab-maas-text-embedding-3-large")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_index")

# http client: sample code uses verify=False
http_client = httpx.Client(verify=False)

# instantiate LLM & embeddings
llm = ChatOpenAl(
    base_url=API_BASE,
    model=LLM_MODEL,
    api_key=API_KEY,
    http_client=http_client
)

embedding_model = OpenAlEmbeddings(
    base_url=API_BASE,
    model=EMB_MODEL,
    api_key=API_KEY,
    http_client=http_client
)

app = FastAPI(title="RAG Agent Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- helpers ----------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
        tf.write(pdf_bytes)
        tmp_path = tf.name
    text = extract_text(tmp_path)
    return text

def chunk_text(text: str, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks

# ---------- API models ----------
class QueryRequest(BaseModel):
    query: str
    k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list

# ---------- Endpoints ----------
@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads supported.")
    data = await file.read()
    raw_text = extract_text_from_pdf_bytes(data)
    chunks = chunk_text(raw_text, chunk_size=1000, chunk_overlap=200)

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR
    )
    vectordb.persist()
    return {"status": "indexed", "chunks": len(chunks)}

@app.post("/query", response_model=QueryResponse)
async def query_rag(req: QueryRequest):
    # load persisted vectordb
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": req.k})

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # try invoke or run depending on wrapper
    try:
        result = rag_chain.invoke(req.query)
    except Exception:
        result = rag_chain.run(req.query)

    # Extract answer (result may be string or dict depending on wrapper)
    answer = result if isinstance(result, str) else (result.get("result") if isinstance(result, dict) else str(result))

    # Attempt to get source docs: may depend on the LangChain wrapper
    sources = []
    try:
        docs = rag_chain.get_last_source_documents()  # this may not exist; adapt if needed
        for d in docs:
            sources.append({"content": d.page_content[:400]})
    except Exception:
        # fallback: try to load top K from retriever directly
        docs = retriever.get_relevant_documents(req.query) if hasattr(retriever, "get_relevant_documents") else []
        for d in docs[:req.k]:
            sources.append({"content": getattr(d, "page_content", str(d))[:400]})

    return QueryResponse(answer=answer, sources=sources)