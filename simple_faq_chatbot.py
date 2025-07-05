import os
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_and_chunk_pdfs(folder):
    chunks = []
    for filename in os.listdir(folder):
        if filename.endswith('.pdf'):
            path = os.path.join(folder, filename)
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        # break page text into smaller chunks
                        for i in range(0, len(text), 500):
                            chunk = text[i:i+500]
                            chunks.append({'source': f"{filename} p{page_num+1}", 'text': chunk})
    return chunks

def embed_chunks(chunks):
    for chunk in chunks:
        chunk['embedding'] = model.encode(chunk['text'])
    return chunks

st.title("📚 Smart College PDF QA Bot")

with st.spinner("Loading PDFs and creating embeddings..."):
    pdf_chunks = load_and_chunk_pdfs('docs')
    pdf_chunks = embed_chunks(pdf_chunks)

question = st.text_input("Ask your question:")

if question:
    question_emb = model.encode(question)
    # Compute similarity with all chunks
    scores = [
        (chunk['source'], chunk['text'], float(cosine_similarity([question_emb], [chunk['embedding']])[0][0]))
        for chunk in pdf_chunks
    ]
    # Sort by highest score
    scores.sort(key=lambda x: x[2], reverse=True)
    top_source, top_text, top_score = scores[0]

    if top_score > 0.4:  # adjust threshold as needed
        st.subheader(f"📄 Best match (similarity: {top_score:.2f}) from {top_source}:")
        st.write(top_text)
    else:
        st.write("❓ Sorry, I don't know the answer. Try asking differently.")
