import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAQ data
faq = pd.read_csv("faq.csv")

# Compute embeddings for all known questions once
faq['embedding'] = faq['question'].apply(lambda x: model.encode(x))

# Streamlit UI
st.title("🤖 Smart College FAQ Chatbot")

user_question = st.text_input("Ask your question:")

if user_question:
    # Encode user question
    user_embedding = model.encode(user_question)

    # Compute cosine similarities with all FAQ questions
    similarities = faq['embedding'].apply(lambda x: cosine_similarity(
        [user_embedding], [x])[0][0])

    # Find best match
    best_match_idx = similarities.idxmax()
    best_score = similarities[best_match_idx]

    # Set a reasonable threshold
    threshold = 0.6

    if best_score >= threshold:
        answer = faq.loc[best_match_idx, 'answer']
        st.write(f"✅ {answer} (similarity: {best_score:.2f})")
    else:
        st.write("❓ Sorry, I don't know the answer. Try asking differently.")
