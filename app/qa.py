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


def ask_question(question, db):
    docs = db.similarity_search(question, k=3)

    if not docs:
        return "No relevant information found.", []

    q_words = question.lower().split()
    best_sentence = None
    sources = []

    for d in docs:
        text = d.page_content
        source = d.metadata.get("source", "Unknown")

        # split into sentences
        sentences = re.split(r'(?<=[.!?]) +', text)

        for sent in sentences:
            low = sent.lower()

            # check keyword overlap
            score = sum(1 for w in q_words if w in low)

            if score >= 2:  # threshold match
                best_sentence = sent.strip()
                sources.append((source, sent.strip()))
                break

        if best_sentence:
            break

    if not best_sentence:
        return "No relevant information found.", []

    return best_sentence, sources