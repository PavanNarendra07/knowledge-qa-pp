# ---- add at top ----
import os
import shutil
import streamlit as st
from qa import build_vector_store, ask_question

@st.cache_resource
def load_db():
    db = build_vector_store()
    return db

DATA_FOLDER = "data"
VECTOR_FOLDER = "vectorstore"

os.makedirs(DATA_FOLDER, exist_ok=True)

# ---- clear data on fresh start ----
if "init_done" not in st.session_state:
    if os.path.exists(DATA_FOLDER):
        shutil.rmtree(DATA_FOLDER, ignore_errors=True)
    if os.path.exists(VECTOR_FOLDER):
        shutil.rmtree(VECTOR_FOLDER, ignore_errors=True)

    os.makedirs(DATA_FOLDER, exist_ok=True)
    st.session_state.init_done = True

# ---- UI ----
st.set_page_config(page_title="Private Knowledge Q&A")
st.title("Private Knowledge Q&A")

uploaded_files = st.file_uploader(
    "Upload text files",
    type=["txt"],
    accept_multiple_files=True
)

os.makedirs(DATA_FOLDER, exist_ok=True)

if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(DATA_FOLDER, file.name), "wb") as f:
            f.write(file.getbuffer())

    if os.path.exists(VECTOR_FOLDER):
        shutil.rmtree(VECTOR_FOLDER, ignore_errors=True)

    st.cache_resource.clear()
    db = load_db()
    st.success("Files uploaded successfully")

st.subheader("Uploaded Documents")
files = os.listdir(DATA_FOLDER)

if files:
    for f in files:
        col1, col2 = st.coloumns([8,1])
        col1.write(f)

        if col2.button("❌", key=f):
            os.remove(os.path.join(DATA_FOLDER, f))

            # rebuild store after delete
            if os.path.exists(VECTOR_FOLDER):
                shutil.rmtree(VECTOR_FOLDER, ignore_errors=True)

            st.cache_resource.clear()

            st.rerun()
else:
    st.write("No documents uploaded yet.")
    

st.subheader("Ask a Question")
question = st.text_input("Enter your question")


if st.button("Get Answer"):
    
    if not os.listdir(DATA_FOLDER):
        st.warning("Upload documents first.")
    elif not question.strip():
        st.warning("Enter a question.")
    else:
        db = load_db()
        if db is None:
            st.error("Vector store not built.")
        else:
            answer, sources = ask_question(question, db)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Sources")
            for src, text in sources:
                st.write(f"Document: {src}")
                st.write(text)