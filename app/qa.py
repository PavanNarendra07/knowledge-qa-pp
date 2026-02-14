import os
import re
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_FOLDER = "data"
VECTOR_FOLDER = "vectorstore"


# ---------- Read files ----------
def read_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ---------- Build Vector Store ----------
def build_vector_store():
    if not os.path.exists(DATA_FOLDER):
        return None

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
        chunk_size=800,
        chunk_overlap=100,
        separator="\n"
    )

    docs = splitter.create_documents(texts, metadatas=metadata)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(VECTOR_FOLDER)

    return db


# ---------- Cleaning ----------
def clean_text(text):
    text = re.sub(r"\[\d+\]", "", text)  # remove wiki refs
    text = re.sub(r"http\S+", "", text)  # remove links
    return text


# ---------- Best Sentence ----------
def best_sentence(text, question):
    text = clean_text(text)

    sentences = re.split(r'(?<=[.!?])\s+', text)
    q_words = question.lower().split()

    best = None
    best_score = 0

    for s in sentences:
        s_low = s.lower()

        score = sum(1 for w in q_words if w in s_low)

        # ignore junk lines
        if len(s) < 40:
            continue

        if score > best_score:
            best_score = score
            best = s

    return best


# ---------- Ask Question ----------
def ask_question(question, db):
    if db is None:
        return "Upload documents first.", []

    docs = db.similarity_search(question, k=6)

    if not docs:
        return "No relevant information found.", []

    for d in docs:
        answer = best_sentence(d.page_content, question)

        if answer:
            src = d.metadata.get("source", "Unknown")
            return answer.strip(), [(src, answer.strip())]

    return "No relevant information found.", []