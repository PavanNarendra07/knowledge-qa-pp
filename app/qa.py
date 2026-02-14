import os
from utils import read_file

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_FOLDER = "data"
VECTOR_FOLDER = "vectorstore"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


def build_vector_store():
    texts = []
    metadata = []

    if not os.path.exists(DATA_FOLDER):
        return None

    for file in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, file)

        content = read_file(path)

        if content and content.strip():
            texts.append(content)
            metadata.append({"source": file})

    if not texts:
        return None

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.create_documents(
        texts,
        metadatas=metadata
    )

    embeddings = get_embeddings()

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(VECTOR_FOLDER)

    return db


def load_vector_store():
    embeddings = get_embeddings()

    return FAISS.load_local(
        VECTOR_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True
    )


def ask_question(question):
    if not question.strip():
        return "Please enter a valid question.", []

    # Build store if missing
    if not os.path.exists(VECTOR_FOLDER):
        db = build_vector_store()
        if db is None:
            return "No documents available. Please upload documents first.", []
    else:
        db = load_vector_store()

    docs = db.similarity_search(question, k=2)

    if not docs:
        return "No relevant information found.", []

    context = "\n".join([d.page_content for d in docs])

    # Basic retrieval answer
    response = context[:1000]

    sources = [
        (d.metadata.get("source", "Unknown"), d.page_content[:200])
        for d in docs
    ]

    return response, sources