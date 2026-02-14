import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FOLDER = "data"

vectorizer = None
sentences = []
sources = []


# ---------- Clean text ----------
def clean_text(text):
    text = re.sub(r"\[\d+\]", "", text)
    return text


# ---------- Sentence splitting ----------
def split_sentences(text):
    text = clean_text(text)
    return re.split(r'(?<=[.!?])\s+', text)


# ---------- Build index ----------
def build_vector_store():
    global vectorizer, sentences, sources

    sentences = []
    sources = []

    if not os.path.exists(DATA_FOLDER):
        return None

    for file in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, file)

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        sents = split_sentences(text)

        for s in sents:
            s = s.strip()
            if len(s) > 20:
                sentences.append(s)
                sources.append(file)

    if not sentences:
        return None

    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit(sentences)

    return True


# ---------- Ask question ----------
def ask_question(question, db=None):
    global vectorizer, sentences, sources

    if not sentences:
        return "No relevant information found.", []

    q_vec = vectorizer.transform([question])
    s_vec = vectorizer.transform(sentences)

    scores = cosine_similarity(q_vec, s_vec)[0]
    best_idx = scores.argmax()

    if scores[best_idx] < 0.1:
        return "No relevant information found.", []

    answer = sentences[best_idx]
    src = sources[best_idx]

    return answer, [(src, answer)]