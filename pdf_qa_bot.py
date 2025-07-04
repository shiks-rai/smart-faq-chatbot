import streamlit as st
import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Title
st.title("📚 Smart College PDF Chatbot")

# Step 1: Load PDFs and extract text
def load_pdfs(folder_path):
    text_chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            pdf = PyPDF2.PdfReader(pdf_path)
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_chunks.append(text)
    return text_chunks

st.info("Loading PDFs...")
docs = load_pdfs("docs")

# Step 2: Embed documents
st.info("Creating embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(docs)

# Step 3: Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

st.success("Ready! Ask your question below:")

# Step 4: Ask question
question = st.text_input("❓ Enter your question:")

if question:
    q_embedding = model.encode([question])
    D, I = index.search(np.array(q_embedding), k=1)  # find top-1 most similar chunk
    answer = docs[I[0][0]]
    st.write("📖 **Answer:**")
    st.write(answer)
