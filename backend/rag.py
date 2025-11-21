# rag.py
import os
import hashlib
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

from llm_client import embeddings, llm

ROOT = "vector_stores"
GLOBAL_PATH = os.path.join(ROOT, "global", "chroma_db")
USERS_PATH = os.path.join(ROOT, "users")
SESSIONS_PATH = os.path.join(ROOT, "sessions")

os.makedirs(GLOBAL_PATH, exist_ok=True)
os.makedirs(USERS_PATH, exist_ok=True)
os.makedirs(SESSIONS_PATH, exist_ok=True)

def _chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def _norm_hash(s: str) -> str:
    s2 = " ".join(s.split()).strip().lower()
    return hashlib.sha256(s2.encode('utf-8')).hexdigest()

# ---------------- Global KB ----------------
def build_global_kb_from_texts(texts: List[str]) -> dict:
    if not texts:
        return {"indexed_chunks": 0}
    chunks = []
    for t in texts:
        chunks.extend(_chunk_text(t))
    vectordb = Chroma.from_texts(chunks, embeddings, persist_directory=GLOBAL_PATH)
    vectordb.persist()
    return {"indexed_chunks": len(chunks)}

def global_exists() -> bool:
    return os.path.exists(GLOBAL_PATH) and bool(os.listdir(GLOBAL_PATH))

def get_global_retriever():
    if not global_exists():
        return None
    db = Chroma(persist_directory=GLOBAL_PATH, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k":3})

# ---------------- User KB (dedupe) ----------------
def _user_dir(user_id: str) -> str:
    return os.path.join(USERS_PATH, str(user_id), "chroma_db")

def add_texts_to_user(user_id: str, text: str) -> dict:
    ud = _user_dir(user_id)
    os.makedirs(os.path.dirname(ud), exist_ok=True)
    chunks = _chunk_text(text)
    if not chunks:
        return {"indexed_chunks":0, "status":"no_text"}

    existing_hashes = set()
    if os.path.exists(ud) and os.listdir(ud):
        try:
            vectordb = Chroma(persist_directory=ud, embedding_function=embeddings)
            existing = vectordb.get()
            docs = existing.get("documents", []) if isinstance(existing, dict) else []
            for d in docs:
                txt = d if isinstance(d, str) else (d.get("text") if isinstance(d, dict) else str(d))
                if txt:
                    existing_hashes.add(_norm_hash(txt))
        except Exception:
            existing_hashes = set()

    to_add = [c for c in chunks if _norm_hash(c) not in existing_hashes]
    if not os.path.exists(ud) or not os.listdir(ud):
        if to_add:
            vectordb = Chroma.from_texts(to_add, embeddings, persist_directory=ud)
            vectordb.persist()
            return {"indexed_chunks": len(to_add), "status":"created"}
        else:
            return {"indexed_chunks":0, "status":"nothing_new"}
    else:
        if to_add:
            vectordb = Chroma(persist_directory=ud, embedding_function=embeddings)
            try:
                vectordb.add_texts(to_add)
            except Exception:
                try:
                    existing = vectordb.get()
                    existing_texts = existing.get("documents", []) if isinstance(existing, dict) else []
                except Exception:
                    existing_texts = []
                merged = existing_texts + to_add
                vectordb = Chroma.from_texts(merged, embeddings, persist_directory=ud)
                vectordb.persist()
            return {"indexed_chunks": len(to_add), "status":"appended"}
        else:
            return {"indexed_chunks":0, "status":"nothing_new"}

def get_user_retriever(user_id: str):
    ud = _user_dir(user_id)
    if not os.path.exists(ud) or not os.listdir(ud):
        return None
    db = Chroma(persist_directory=ud, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k":3})

# ---------------- Session KB (append) ----------------
def _session_dir(user_id: str, session_id: str) -> str:
    return os.path.join(SESSIONS_PATH, str(user_id), str(session_id), "chroma_db")

def add_texts_to_session(user_id: str, session_id: str, text: str) -> dict:
    sd = _session_dir(user_id, session_id)
    os.makedirs(os.path.dirname(sd), exist_ok=True)
    chunks = _chunk_text(text)
    print(f"DEBUG: Chunked text into {len(chunks)} chunks for session {session_id}")
    if not chunks:
        return {"indexed_chunks":0, "status":"no_text"}
    if not os.path.exists(sd) or not os.listdir(sd):
        print(f"DEBUG: Creating new session vector store at {sd}")
        vectordb = Chroma.from_texts(chunks, embeddings, persist_directory=sd)
        vectordb.persist()
        return {"indexed_chunks": len(chunks), "status":"created"}
    else:
        print(f"DEBUG: Adding to existing session vector store at {sd}")
        vectordb = Chroma(persist_directory=sd, embedding_function=embeddings)
        try:
            vectordb.add_texts(chunks)
        except Exception as e:
            print(f"DEBUG: Error adding texts: {e}")
            try:
                existing = vectordb.get()
                existing_texts = existing.get("documents", []) if isinstance(existing, dict) else []
            except Exception:
                existing_texts = []
            merged = existing_texts + chunks
            vectordb = Chroma.from_texts(merged, embeddings, persist_directory=sd)
            vectordb.persist()
        return {"indexed_chunks": len(chunks), "status":"appended"}

def get_session_retriever(user_id: str, session_id: str):
    sd = _session_dir(user_id, session_id)
    print(f"DEBUG: Looking for session retriever at {sd}")
    if not os.path.exists(sd) or not os.listdir(sd):
        print(f"DEBUG: Session vector store not found at {sd}")
        return None
    print(f"DEBUG: Session vector store found, creating retriever")
    db = Chroma(persist_directory=sd, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k":4})

# ---------------- Combined retriever ----------------
class CombinedRetriever:
    def __init__(self, retrievers):
        self.retrievers = [r for r in retrievers if r is not None]

    def get_relevant_documents(self, query):
        results = []
        for r in self.retrievers:
            try:
                docs = r.get_relevant_documents(query)
                if isinstance(docs, list):
                    results.extend(docs)
            except Exception:
                continue
        return results

def get_combined_chain(user_id: str, session_id: str):
    rets = []
    sr = get_session_retriever(user_id, session_id)
    ur = get_user_retriever(user_id)
    gr = get_global_retriever()
    if sr: rets.append(sr)
    if ur: rets.append(ur)
    if gr: rets.append(gr)
    if not rets:
        raise FileNotFoundError("No retrievers available")
    combined = CombinedRetriever(rets)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=combined,
        return_source_documents=True
    )
    return chain