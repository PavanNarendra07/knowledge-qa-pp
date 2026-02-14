import os
import re
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings



DATA_FOLDER = "data"
VECTOR_FOLDER = "vectorstore"


# ---------- Read files ----------
def read_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ---------- Build Vector Store ----------
def build_vector_store():
    files = os.listdir(DATA_FOLDER)

    texts = []
    metadata = []

    for file in files:
        path = os.path.join(DATA_FOLDER, file)
        content = read_file(path)

        if content.strip():
            texts.append(content)
            metadata.append({"source": file})

    if not texts:
        return None

    splitter = CharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=40,
        separator="\n"
    )

    docs = splitter.create_documents(texts, metadatas=metadata)

    embeddings = FakeEmbeddings(size=384)
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(VECTOR_FOLDER)

    return db


import re

def clean_text(text):
    return re.sub(r"\[\d+\]", "", text)


def best_sentence(text, question):
    text = clean_text(text)

    sentences = re.split(r'(?<=[.!?])\s+', text)
    q_words = question.lower().split()

    candidates = []

    for s in sentences:
        s_low = s.lower()

        score = sum(1 for w in q_words if w in s_low)

        # ignore useless long sentences
        if score > 0 and 20 < len(s) < 250:
            candidates.append((score, len(s), s))

    if not candidates:
        return None

    # highest keyword match + shortest clean sentence
    candidates.sort(key=lambda x: (-x[0], x[1]))

    return candidates[0][2].strip()

def ask_question(question, db):
    docs = db.similarity_search(question, k=8)

    if not docs:
        return "No relevant information found.", []

    q = question.lower()

    # patterns for factual answers
    patterns = [
        r"\(born[^)]*\)",          # born info
        r"born\s+\d{1,2}.*\d{4}",  # born date
        r"\d{4}",                  # year facts fallback
    ]

    for d in docs:
        text = clean_text(d.page_content)

        lines = text.split("\n")

        for line in lines:
            l = line.strip()

            # skip links & junk
            if len(l) < 30 or "http" in l:
                continue

            for p in patterns:
                if re.search(p, l.lower()):
                    src = d.metadata.get("source", "Unknown")
                    return l, [(src, l)]

    # fallback best non-link line
    for d in docs:
        lines = d.page_content.split("\n")
        for l in lines:
            if len(l) > 30 and "http" not in l:
                src = d.metadata.get("source", "Unknown")
                return l.strip(), [(src, l.strip())]

    return "No relevant information found.", []