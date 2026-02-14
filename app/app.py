import os
import shutil
import streamlit as st
from qa import build_vector_store, ask_question

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Private Knowledge Q&A", layout="wide")

# =========================
# GLOBAL PROFESSIONAL UI CSS
# =========================
st.markdown("""
<style>

/* Global font */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    font-size: 18px !important;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

/* Hide header/footer */
header, footer {
    visibility: hidden;
}

/* Title */
.main-title {
    text-align: center;
    font-size: 44px;
    font-weight: 800;
    padding: 25px;
    color: white;
}

/* Card container */
.card {
    background: rgba(30, 41, 59, 0.9);
    padding: 28px;
    border-radius: 16px;
    margin-bottom: 25px;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.4);
    transition: 0.3s;
}

/* Card hover */
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 18px 45px rgba(0,0,0,0.6);
}

/* Section titles */
h2, h3 {
    font-size: 26px !important;
    font-weight: 700;
}

/* Text input */
.stTextInput input {
    font-size: 18px !important;
    padding: 12px !important;
    border-radius: 10px !important;
    background-color: #020617;
    color: white;
}

/* Buttons */
.stButton > button {
    font-size: 18px !important;
    padding: 12px 22px;
    border-radius: 12px;
    font-weight: 600;
    color: white;
    border: none;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    transition: 0.3s;
}

/* Button hover */
.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #1d4ed8, #6d28d9);
}

/* File uploader */
section[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.9);
    padding: 20px;
    border-radius: 14px;
    font-size: 18px;
}

/* Uploaded file name */
section[data-testid="stFileUploader"] span {
    font-size: 18px !important;
}

/* Uploaded docs list */
.css-1d391kg {
    font-size: 18px !important;
}

/* Answer box */
.answer-box {
    background: rgba(15,23,42,0.95);
    padding: 20px;
    border-radius: 12px;
    font-size: 18px;
}

/* Success / warning text */
.stAlert {
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# VECTOR STORE LOADER
# =========================
@st.cache_resource
def load_db():
    return build_vector_store()

DATA_FOLDER = "data"
VECTOR_FOLDER = "vectorstore"
os.makedirs(DATA_FOLDER, exist_ok=True)

# =========================
# CLEAN INIT
# =========================
if "init_done" not in st.session_state:
    shutil.rmtree(DATA_FOLDER, ignore_errors=True)
    shutil.rmtree(VECTOR_FOLDER, ignore_errors=True)
    os.makedirs(DATA_FOLDER, exist_ok=True)
    st.session_state.init_done = True

# =========================
# TITLE
# =========================
st.markdown(
    '<div class="main-title">📚 Private Knowledge Q&A Workspace</div>',
    unsafe_allow_html=True
)

# =========================
# FILE UPLOAD
# =========================
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

# =========================
# DOCUMENT LIST
# =========================
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

# =========================
# QUESTION SECTION
# =========================
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
            st.markdown(
                f'<div class="answer-box">{answer}</div>',
                unsafe_allow_html=True
            )

            st.subheader("Sources")

            for src, text in sources:
                st.markdown(f"**Document:** {src}")
                st.write(text)

st.markdown('</div>', unsafe_allow_html=True)