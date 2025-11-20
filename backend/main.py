# main.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from insecure_requests import enable_insecure_requests
from guardrails import validate_input, sanitize_answer
from rag import add_texts_to_session, add_texts_to_user
from agent.rag_agent_graph import run_rag_graph
from models import ChatRequest, UserKBUpload

enable_insecure_requests()

app = FastAPI(title="Unified RAG + Agent Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "Backend running"}

@app.post("/chat")
async def unified_chat(payload: ChatRequest):
    """
    Unified endpoint:
      - Accepts JSON payload with optional file_contents (list of strings) and/or question
      - Indexes each file content into session KB (and user KB if index_user=True)
      - If question provided: run agent (retrieval + LLM) and return answer + sources
      - Always include user_id and session_id in response
    """
    user_id = payload.user_id
    session_id = payload.session_id
    file_contents = payload.file_contents or []
    question = payload.question
    index_user = bool(payload.index_user)

    ok, err = validate_input(user_id, session_id, question, file_contents)
    if not ok:
        return {"user_id": user_id, "session_id": session_id, "error": err}

    response = {"user_id": user_id, "session_id": session_id}

    # Index each file content to session KB (append)
    session_index_results = []
    user_index_results = []
    if file_contents:
        for content in file_contents:
            if not content or not content.strip():
                continue
            sres = add_texts_to_session(user_id, session_id, content)
            session_index_results.append(sres)
            if index_user:
                ures = add_texts_to_user(user_id, content)
                user_index_results.append(ures)
        response["session_index"] = session_index_results
        if index_user:
            response["user_index"] = user_index_results

    # If question exists → run agent graph
    if question:
        payload_graph = {"user_id": user_id, "session_id": session_id, "question": question}
        result = run_rag_graph(payload_graph)
        answer = sanitize_answer(result.get("answer", ""))
        sources = result.get("sources", [])
        response["answer"] = answer
        response["sources"] = sources

    # If no question and only indexing
    if (not question) and file_contents:
        response.setdefault("message", "Indexed file_contents to session (and user if index_user=true).")

    # If neither question nor file_contents
    if (not question) and (not file_contents):
        return {"user_id": user_id, "session_id": session_id, "error": "Send file_contents or question or both."}

    return response

@app.post("/kb/user/upload")
async def upload_user_kb(payload: UserKBUpload):
    user_id = payload.user_id
    file_contents = payload.file_contents or []
    if not user_id or not file_contents:
        return {"error": "user_id and file_contents are required."}
    res = []
    for content in file_contents:
        if not content or not content.strip():
            continue
        r = add_texts_to_user(user_id, content)
        res.append(r)
    return {"user_id": user_id, "status": "user_kb_indexed", "detail": res}