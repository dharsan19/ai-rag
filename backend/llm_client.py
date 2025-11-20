import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "smollm")
EMBED_MODEL = os.getenv("EMBED_MODEL", "jina-embeddings-v2-small-en")

llm = Ollama(
    base_url="http://localhost:11434",
    model=LLM_MODEL,
)

embeddings = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model=EMBED_MODEL,
)