# tests/test_chat.py
import requests, time

BASE = "http://127.0.0.1:8000"

def test_upload_user_kb():
    r = requests.post(f"{BASE}/kb/user/upload", json={
        "user_id": "user123",
        "file_contents": ["User prefers concise answers.", "User works in finance."]
    })
    print("upload user kb:", r.status_code, r.text)

def test_chat_index_and_query():
    payload = {
        "user_id": "user123",
        "session_id": "sessionA",
        "file_contents": [
            "Python is a programming language used for scripts, web backend, and data science.",
            "It is dynamically typed and has a large ecosystem."
        ],
        "question": "What is Python used for?",
        "index_user": True
    }
    r = requests.post(f"{BASE}/chat", json=payload)
    print("chat response:", r.status_code, r.json())

if __name__ == "__main__":
    test_upload_user_kb()
    time.sleep(1)
    test_chat_index_and_query()