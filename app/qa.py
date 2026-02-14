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
    main_term = q_words[0] if q_words else ""

    candidates = []

    for s in sentences:
        s_lower = s.lower()

        # must contain main entity
        if main_term not in s_lower:
            continue

        score = sum(1 for w in q_words if w in s_lower)

        if score > 0:
            candidates.append((score, len(s), s))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][2].strip()

def ask_question(question, db):
    docs = db.similarity_search(question, k=10)

    q_words = [w.lower() for w in question.split() if len(w) > 2]

    # keep docs strongly matching the question
    filtered = []
    for d in docs:
        text = d.page_content.lower()
        score = sum(1 for w in q_words if w in text)

        if score >= max(1, len(q_words) // 2):
            filtered.append((score, d))

    # fallback if nothing matched
    if not filtered:
        filtered = [(1, d) for d in docs]

    # sort by keyword match
    filtered.sort(key=lambda x: x[0], reverse=True)

    for _, d in filtered:
        sentence = best_sentence(d.page_content, question)
        if sentence:
            src = d.metadata.get("source", "Unknown")
            return sentence, [(src, sentence)]

    return "No relevant information found.", []