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
    texts = []
    metadata = []

    for file in os.listdir(DATA_FOLDER):
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
    embeddings = HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(VECTOR_FOLDER, embeddings, allow_dangerous_deserialization=True)

def select_best_answer(docs, question):
    if not docs:
        return "No relevant information found."
    text = docs[0].page_content
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return " ".join(lines)[:250]


def ask_question(question, db):

    docs = db.similarity_search(question, k=1)

    response = select_best_answer(docs, question)

    seen =set()
    sources = []

    for d in docs:
        src = d.metadata["source"]
        if src not in seen:
            sources.append((src, d.page_content[:200]))
            seen.add(src)

    return response, sources