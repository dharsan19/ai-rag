# agent/rag_agent_graph.py
from typing import Dict, Any, List
from tools.retrieval import retrieve_from_all
from prompts import ANSWER_SYSTEM_PROMPT
from llm_client import llm
from guardrails import sanitize_answer

def input_guard_node(payload: Dict[str, Any]) -> Dict[str, Any]:
    # pass-through now; place for future checks
    return payload

def retrieve_node(payload: Dict[str, Any]) -> Dict[str, Any]:
    user_id = payload.get("user_id")
    session_id = payload.get("session_id")
    question = payload.get("question", "")
    docs = retrieve_from_all(user_id, session_id, question, k_per_source=3)
    payload["retrieved_docs"] = docs
    return payload

def generate_node(payload: Dict[str, Any]) -> Dict[str, Any]:
    docs: List[object] = payload.get("retrieved_docs", [])
    question: str = payload.get("question", "")

    context_parts = []
    for i, d in enumerate(docs):
        text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        snippet = text[:1200]
        context_parts.append(f"[DOC {i+1}]\n{snippet}")
    context_text = "\n\n".join(context_parts)

    prompt = ANSWER_SYSTEM_PROMPT + "\n\nContext:\n" + context_text + "\n\nUser question:\n" + question + \
             "\n\nInstructions: Answer concisely based only on the context above. If context doesn't contain the answer, say \"I don't have enough information to answer that.\" Also include a short Sources: section with short excerpts."

    try:
        resp = llm.invoke(prompt)
        answer = resp.get("result") if isinstance(resp, dict) and "result" in resp else str(resp)
    except Exception as e:
        answer = f"LLM error: {e}"

    payload["answer_raw"] = answer
    return payload

def output_guard_node(payload: Dict[str, Any]) -> Dict[str, Any]:
    answer = payload.get("answer_raw", "")
    answer = sanitize_answer(answer)
    docs = payload.get("retrieved_docs", [])
    sources = []
    for d in docs:
        txt = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        sources.append(txt[:400])
    return {"answer": answer, "sources": sources}

def run_rag_graph(payload: Dict[str, Any]) -> Dict[str, Any]:
    p = input_guard_node(payload)
    p = retrieve_node(p)
    p = generate_node(p)
    out = output_guard_node(p)
    return out