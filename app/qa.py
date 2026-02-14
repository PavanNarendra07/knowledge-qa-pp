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
        model_name="sentence-transformers/all-MiniLM-L6-v2",
          model_kwargs={"device":"cpu", "trust_remote_code": True},
          encode_kwargs = {"normalize_embeddings": True, "batch_size" : 16}
    )

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(VECTOR_FOLDER)

    return db


# ---------- Clean best sentence ----------
def best_sentence(text, question):
    sentences = re.split(r'(?<=[.!?]) +', text)
    q_words = question.lower().split()

    best_score = 0
    best_line = sentences[0]

    for s in sentences:
        s_low = s.lower()
        score = sum(word in s_low for word in q_words)

        if score > best_score:
            best_score = score
            best_line = s

    return best_line.strip()


# ---------- Select best answer ----------
def select_best_answer(docs, question):
    if not docs:
        return "No relevant information found.", []

    q_words = question.lower().split()

    scored_docs = []
    for d in docs:
        text = d.page_content.lower()
        score = sum(word in text for word in q_words)
        scored_docs.append((score, d))

    scored_docs.sort(key=lambda x: x[0], reverse=True)

    best_doc = scored_docs[0][1]
    answer_line = best_sentence(best_doc.page_content, question)

    sources = []
    for score, d in scored_docs[:2]:
        sources.append(
            (d.metadata.get("source", "Unknown"),
             d.page_content[:250])
        )

    return answer_line, sources


# ---------- Ask Question ----------
def ask_question(question, db):
    if db is None:
        return "No documents available.", []

    docs = db.similarity_search(question, k=4)
    answer, sources = select_best_answer(docs, question)

    return answer, sources