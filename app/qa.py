from utils import read_file
import os 
import re

from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_FOLDER = "data"
VECTOR_FOLDER = "vectorstore"

def build_vector_store():
    os.makedirs(DATA_FOLDER, exist_ok=True)

    files = os.listdir(DATA_FOLDER)
    if not files:
        return None
    texts = []
    metadata = []

    for file in files:
        path = os.path.join(DATA_FOLDER, file)
        content = read_file(path)
        texts.append(content)
        metadata.append({"source": file})
    
    if not texts:
        return None
    splitter = CharacterTextSplitter(chunk_size= 300, chunk_overlap = 30, separator="\n")
    docs = splitter.create_documents(texts, metadatas = metadata)
    embeddings =  HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    db = FAISS.from_documents(docs,embeddings)
    db.save_local(VECTOR_FOLDER)

    return db

def load_vector_store():
    return build_vector_store()

def select_best_answer(docs, question):
    if not docs:
        return "No relevant information found."

    q = question.lower()

    # prefer doc whose source matches query
    for d in docs:
        src = d.metadata.get("source", "").lower()
        if any(word in src for word in q.split()):
            return (d.page_content or "")[:250]

    # fallback to first
    return (docs[0].page_content or "")[:250]


def clean_text(text):
    return re.sub(r"\[\d+\]", "", text)


def best_sentence(text, question):
    text = clean_text(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    q_words = question.lower().split()
    scored = []

    for s in sentences:
        s_lower = s.lower()
        score = sum(1 for w in q_words if w in s_lower)

        if score > 0:
            scored.append((score, len(s), s))

    if not scored:
        return None

    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][2].strip()


def ask_question(question, db):
    docs = db.similarity_search(question, k=6)

    if not docs:
        return "No relevant information found.", []

    q_lower = question.lower()

    # prioritize docs whose filename matches query words
    prioritized = []
    others = []

    for d in docs:
        src = d.metadata.get("source", "").lower()

        if any(word in src for word in q_lower.split()):
            prioritized.append(d)
        else:
            others.append(d)

    ordered_docs = prioritized + others

    for d in ordered_docs:
        sentence = best_sentence(d.page_content, question)

        if sentence:
            src = d.metadata.get("source", "Unknown")
            return sentence, [(src, sentence)]

    return "No relevant information found.", []