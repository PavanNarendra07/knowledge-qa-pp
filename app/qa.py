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

    if not docs:
        return "No relevant information found.", []

    stop_words = {"when", "what", "who", "is", "was", "the", "a", "an"}
    q_words = [w.lower() for w in question.split() if w.lower() not in stop_words]

    # handle born/birth variation
    variants = set(q_words)
    if "born" in variants:
        variants.add("birth")
    if "birth" in variants:
        variants.add("born")

    for d in docs:
        sentences = re.split(r'(?<=[.!?])\s+', d.page_content)

        for s in sentences:
            s_low = s.lower()

            # require name match + born/birth
            if all(w in s_low for w in variants if len(w) > 3):
                src = d.metadata.get("source", "Unknown")
                return s.strip(), [(src, s.strip())]

    return "No relevant information found.", []