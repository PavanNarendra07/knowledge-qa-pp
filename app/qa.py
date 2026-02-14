import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_FOLDER = "data"
VECTOR_FOLDER = "vectorstore"


# ---------- Read file ----------
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

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(VECTOR_FOLDER)

    return db


# ---------- Pick best answer ----------
def select_best_answer(docs, question):
    if not docs:
        return "No relevant information found.", []

    q_words = question.lower().split()

    scored_docs = []

    for d in docs:
        text = d.page_content
        score = sum(word in text.lower() for word in q_words)
        scored_docs.append((score, d))

    scored_docs.sort(key=lambda x: x[0], reverse=True)

    best_docs = [d for score, d in scored_docs if score > 0]

    if not best_docs:
        best_docs = docs[:1]

    best_doc = best_docs[0]
    text = best_doc.page_content

    # try to return only most relevant sentence
    sentences = text.split(".")
    for s in sentences:
        if any(word in s.lower() for word in q_words):
            answer = s.strip()
            if answer:
                break
    else:
        answer = text[:250]

    answer = answer.strip() + "."

    sources = []
    for d in best_docs[:2]:
        sources.append(
            (d.metadata.get("source", "Unknown"), d.page_content[:250])
        )

    return answer, sources


# ---------- Ask Question ----------
def ask_question(question, db):
    if db is None:
        return "No documents available.", []

    docs = db.similarity_search(question, k=4)

    answer, sources = select_best_answer(docs, question)

    return answer, sources