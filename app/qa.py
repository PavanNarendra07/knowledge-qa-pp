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
    docs = db.similarity_search(question, k=6)

    if not docs:
        return "No relevant information found.", []

    best_answer = None
    best_source = None
    best_score = -1

    q_words = question.lower().split()

    for d in docs:
        sentence = best_sentence(d.page_content, question)

        if not sentence:
            continue

        score = sum(w in sentence.lower() for w in q_words)

        if score > best_score:
            best_score = score
            best_answer = sentence
            best_source = d.metadata.get("source", "Unknown")

    if best_answer:
        return best_answer, [(best_source, best_answer)]

    return "No relevant information found.", []