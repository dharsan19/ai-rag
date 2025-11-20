# guardrails.py
BLOCKED_WORDS = {"kill", "bomb", "suicide", "hack", "attack"}

def validate_input(user_id: str, session_id: str, question: str | None, file_contents: list | None):
    if not user_id or not session_id:
        return False, "Missing user_id or session_id."
    if question and len(question) > 8000:
        return False, "Question too long."
    if file_contents:
        total_len = sum(len(s or "") for s in file_contents)
        if total_len > 2_000_000:
            return False, "Total file_contents too large."
    if question:
        ql = question.lower()
        for b in BLOCKED_WORDS:
            if b in ql:
                return False, "Unsafe or disallowed content in question."
    return True, None

def sanitize_answer(answer: str):
    if not answer:
        return answer
    lower = answer.lower()
    if "i think" in lower or "i may" in lower or "i'm not sure" in lower:
        return answer
    if "always" in lower or "guaranteed" in lower:
        return answer + "\n\nNote: The above may not be universally true; please verify."
    return answer