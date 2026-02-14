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
    # remove wiki refs like [1], [2]
    return re.sub(r"\[\d+\]", "", text)

def best_sentence(text, question):
    text = clean_text(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    q_lower = question.lower()

    # Hard rule for birth questions
    if "born" in q_lower:
        for s in sentences:
            if "born" in s.lower():
                return s.strip()

    # fallback scoring
    q_words = q_lower.split()
    best = None
    best_score = 0

    for s in sentences:
        s_lower = s.lower()
        score = sum(1 for w in q_words if w in s_lower)

        if score > best_score:
            best_score = score
            best = s

    return best.strip() if best else None


def ask_question(question, db):
    docs = db.similarity_search(question, k=6)

    if not docs:
        return "No relevant information found.", []

    for d in docs:
        sentence = best_sentence(d.page_content, question)

        if sentence:
            src = d.metadata.get("source", "Unknown")
            return sentence, [(src, sentence)]

    return "No relevant information found.", []