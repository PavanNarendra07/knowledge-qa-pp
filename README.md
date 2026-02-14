# Private Knowledge Q&A App

This project is a document-based Question Answering web app built using Streamlit and LangChain.  
Users can upload documents and ask questions, and the system retrieves relevant information from uploaded files and shows the source of the answer.

## Features
- Upload PDF, TXT, DOCX, CSV, and Markdown files
- Automatic document text chunking
- Embedding generation for semantic search
- FAISS vector store for document retrieval
- Question answering using an LLM
- Displays document source for answers
- Simple and user-friendly interface

## Tech Stack
- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Sentence Transformers

## How to Run Locally

1. Clone the repository: git clone
2. Create virtual environment : python -m venv venv
3. Activate environment : windows --> venv\Scripts\activate
4. Install dependencies: pip install -r requirements.txt
5. 5.Run the app: streamlit run app/app.py

## Deployment
The application is deployed using Streamlit Cloud and accessible via a live URL.

## Limitations
- Large documents may take time to process.
- Answer quality depends on document content.
- Embedding creation may be slow on low-resource machines.

## Author
 Pavan Narendra Vasamsetti – AI/ML enthusiast building intelligent systems and practical AI applications.
