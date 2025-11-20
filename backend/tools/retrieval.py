# tools/retrieval.py
from typing import List
from rag import get_session_retriever, get_user_retriever, get_global_retriever

def retrieve_from_all(user_id: str, session_id: str, query: str, k_per_source: int = 4) -> List[object]:
    docs = []
    sr = get_session_retriever(user_id, session_id)
    if sr:
        try:
            docs.extend(sr.get_relevant_documents(query)[:k_per_source])
        except Exception:
            pass

    ur = get_user_retriever(user_id)
    if ur:
        try:
            docs.extend(ur.get_relevant_documents(query)[:k_per_source])
        except Exception:
            pass

    gr = get_global_retriever()
    if gr:
        try:
            docs.extend(gr.get_relevant_documents(query)[:k_per_source])
        except Exception:
            pass

    return docs