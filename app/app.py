import os
import shutil
import streamlit as st
from qa import build_vector_store, ask_question

st.set_page_config(page_title="Private Knowledge Q&A", layout="wide")


# STYLING UI CSS

st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

/* Hide Streamlit header/footer */
header, footer {
    visibility: hidden;
}

/* Main Title */
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    padding: 20px;
    color: white;
}

/* Cards */
.card {
    background: rgba(30, 41, 59, 0.85);
    padding: 25px;
    border-radius: 14px;
    margin-bottom: 25px;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 40px rgba(0,0,0,0.6);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    font-weight: 600;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #1d4ed8, #6d28d9);
}

/* File uploader */
section[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.9);
    padding: 15px;
    border-radius: 12px;
}

/* Text input */
.stTextInput input {
    background-color: #020617;
    color: white;
}

/* Answer text */
.answer-box {
    background: rgba(15,23,42,0.95);
    padding: 18px;
    border-radius: 10px;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_db():
    return build_vector_store()

DATA_FOLDER = "data"
VECTOR_FOLDER = "vectorstore"
os.makedirs(DATA_FOLDER, exist_ok=True)


if "init_done" not in st.session_state:
    shutil.rmtree(DATA_FOLDER, ignore_errors=True)
    shutil.rmtree(VECTOR_FOLDER, ignore_errors=True)
    os.makedirs(DATA_FOLDER, exist_ok=True)
    st.session_state.init_done = True


# TITLE
st.markdown('<div class="main-title">📚 Private Knowledge Q&A Workspace</div>',
            unsafe_allow_html=True)


st.markdown('<div class="card">', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload text files",
    type=["txt"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(DATA_FOLDER, file.name), "wb") as f:
            f.write(file.getbuffer())

    shutil.rmtree(VECTOR_FOLDER, ignore_errors=True)
    st.cache_resource.clear()
    load_db()
    st.success("Files uploaded successfully")

st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Uploaded Documents")

files = os.listdir(DATA_FOLDER)

if files:
    for f in files:
        col1, col2 = st.columns([9,1])
        col1.write(f)

        if col2.button("❌", key=f):
            os.remove(os.path.join(DATA_FOLDER, f))

            shutil.rmtree(VECTOR_FOLDER, ignore_errors=True)
            st.cache_resource.clear()
            load_db()

            st.rerun()
else:
    st.write("No documents uploaded.")

st.markdown('</div>', unsafe_allow_html=True)


# QUESTION SECTION

st.markdown('<div class="card">', unsafe_allow_html=True)

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
            st.markdown(f'<div class="answer-box">{answer}</div>',
                        unsafe_allow_html=True)

            st.subheader("Sources")

            for src, text in sources:
                st.markdown(f"**Document:** {src}")
                st.write(text)

st.markdown('</div>', unsafe_allow_html=True)