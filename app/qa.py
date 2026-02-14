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
    embeddings =  HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")
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


def extract_best_sentence(text, question):
    # split sentences
    sentences = re.split(r'(?<=[.!?]) +', text)

    q = question.lower()

    # priority: birth questions
    if "born" in q or "birth" in q:
        for s in sentences:
            if "born" in s.lower():
                # clean references like [1][2]
                s = re.sub(r"\[\d+\]", "", s)
                return s.strip()

    # fallback: keyword match
    words = q.split()
    best = None
    best_score = 0

    for s in sentences:
        low = s.lower()
        score = sum(1 for w in words if w in low)

        if score > best_score:
            best_score = score
            best = s

    if best:
        best = re.sub(r"\[\d+\]", "", best)
        return best.strip()

    return None


def ask_question(question, db):
    docs = db.similarity_search(question, k=3)

    if not docs:
        return "No relevant information found.", []

    for d in docs:
        text = d.page_content
        source = d.metadata.get("source", "Unknown")

        answer = extract_best_sentence(text, question)
        if answer:
            return answer, [(source, answer)]

    return "No relevant information found.", []