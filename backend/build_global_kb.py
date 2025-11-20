# build_global_kb.py
import os
from pathlib import Path
from rag import build_global_kb_from_texts

KB_DIR = Path("knowledge_base/global")

def load_files(kb_dir: Path):
    texts = []
    for p in kb_dir.glob("**/*"):
        if p.is_file():
            try:
                if p.suffix.lower() == ".pdf":
                    from pdfminer.high_level import extract_text
                    txt = extract_text(str(p))
                else:
                    txt = p.read_text(encoding="utf-8", errors="ignore")
                if txt and txt.strip():
                    texts.append(txt)
            except Exception as e:
                print("Failed to load", p, e)
    return texts

if __name__ == "__main__":
    os.makedirs(KB_DIR, exist_ok=True)
    texts = load_files(KB_DIR)
    print(f"Loaded {len(texts)} docs")
    res = build_global_kb_from_texts(texts)
    print("Global KB result:", res)