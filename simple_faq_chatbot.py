import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import streamlit as st

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Read PDFs from docs folder
def load_pdfs(folder):
    all_texts = []
    for filename in os.listdir(folder):
        if filename.endswith('.pdf'):
            path = os.path.join(folder, filename)
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() or ''
                all_texts.append({'filename': filename, 'text': text})
    return all_texts

# Embed texts
def embed_texts(texts):
    for item in texts:
        item['embedding'] = model.encode(item['text'])
    return texts

st.title("📚 College PDF FAQ Chatbot")

# Load and embed
pdf_texts = load_pdfs('docs')
pdf_texts = embed_texts(pdf_texts)

# Ask question
query = st.text_input("Ask something:")

if query:
    query_emb = model.encode(query)
    # Simple similarity (dot product)
    scores = [(item['filename'], item['text'][:200], float(query_emb @ item['embedding'])) for item in pdf_texts]
    scores.sort(key=lambda x: x[2], reverse=True)
    
    st.write("### Top result:")
    st.write(f"📄 **File:** {scores[0][0]}")
    st.write(f"📝 **Excerpt:** {scores[0][1]}...")
