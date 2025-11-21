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

    # Debug logging
    print(f"DEBUG: Retrieved {len(docs)} documents")

    context_parts = []
    for i, d in enumerate(docs):
        text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        snippet = text[:2000]  # Increased from 1200 to 2000
        print(f"DEBUG: Doc {i+1} snippet (first 200 chars): {snippet[:200]}")
        context_parts.append(f"[DOC {i+1}]\n{snippet}")
    context_text = "\n\n".join(context_parts)

    prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context.

Context:
{context_text}

User Question: {question}

Instructions:
- Answer the question using information from the context above
- If you can provide a helpful answer from the context, do so
- Only say "I don't have enough information" if the context is completely irrelevant to the question
- Be helpful and informative
- Include specific details from the context in your answer"""

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