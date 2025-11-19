import os
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("GENAI_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBED_MODEL = os.getenv("EMBED_MODEL")

client = httpx.Client(verify=False)

llm = ChatOpenAI(
    base_url=BASE_URL,
    model=LLM_MODEL,
    api_key=API_KEY,
    http_client=client
)

embeddings = OpenAIEmbeddings(
    base_url=BASE_URL,
    model=EMBED_MODEL,
    api_key=API_KEY,
    http_client=client
)