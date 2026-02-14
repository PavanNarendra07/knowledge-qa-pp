import streamlit as st
import os
import shutil
from qa import build_vector_store, ask_question

from dotenv import load_dotenv
load_dotenv()



DATA_FOLDER = "data"
VECTOR_FOLDER = "vectorstore"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

@st.cache_resource
def load_db():
    return build_vector_store()
db = load_db()

st.set_page_config(page_title="Private Knowledge Q&A")
st.title("Private Knowledge Q&A")
st.write("Upload documents and ask questions from them.")

# ===============================
# Upload Files
# ===============================
uploaded_files = st.file_uploader(
    "Upload text files",
    type=["txt"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(DATA_FOLDER, file.name), "wb") as f:
            f.write(file.getbuffer())

    st.success("Files uploaded successfully")
    db = load_db()

# ===============================
# Show Uploaded Files
# ===============================
st.subheader("Uploaded Documents")
import time

def safe_delete_vectorstore():
    if os.path.exists(VECTOR_FOLDER):
        try:
            shutil.rmtree(VECTOR_FOLDER)
        except PermissionError:
            time.sleep(1)
            shutil.rmtree(VECTOR_FOLDER, ignore_errors=True)

files = os.listdir(DATA_FOLDER)
selected_files = []

if files:
    for file in files:
        col1, col2 = st.columns([5, 1])

        # checkbox for multi delete
        with col1:
            if st.checkbox(file, key=file):
                selected_files.append(file)

        # delete icon for single delete
        with col2:
            if st.button("🗑️", key="del_" + file):
                os.remove(os.path.join(DATA_FOLDER, file))

                safe_delete_vectorstore()

                build_vector_store()
                st.rerun()

    # delete multiple files
    if selected_files:
        if st.button("Delete Selected Files"):
            for f in selected_files:
                os.remove(os.path.join(DATA_FOLDER, f))

            if os.path.exists(VECTOR_FOLDER):
                shutil.rmtree(VECTOR_FOLDER)

            build_vector_store()
            st.success("Selected files deleted")
            st.rerun()

else:
    st.write("No documents uploaded yet.")

# ===============================
# Ask Question
# ===============================
st.subheader("Ask a Question")

question = st.text_input("Enter your question")

if st.button("Get Answer"):
    if not os.listdir(DATA_FOLDER):
        st.warning("Upload documents first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        db = load_db() if os.listdir(DATA_FOLDER) else None
        answer, sources = ask_question(question, db)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for src, text in sources:
            st.write(f"**Document:** {src}")
            st.write(text)