import requests

# # # Upload PDF
# files = {'file': open("AI.pdf", "rb")}
# res = requests.post("http://127.0.0.1:8000/rag/upload", files=files)
# print("Upload Response:", res.json())

# Ask question
# payload = {"message": "what is the model you are using?"}
# res = requests.post("http://127.0.0.1:8000/chat", json=payload)
# print("Ask Response:", res.json())

# Ask question
data = {"question": "what is python?", "file_content": "Python is a easy programming language."}
res = requests.post("http://127.0.0.1:8000/rag", data=data)
print("Ask Response:", res.json())