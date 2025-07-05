import os
import re
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_subject_chunks(pdf_folder):
    subjects = []
    # Looser regex: match any line containing 'Course' (case-insensitive)
    pattern = re.compile(r'.*Course.*', re.IGNORECASE)

    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            path = os.path.join(pdf_folder, filename)
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                full_text = ''
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        full_text += '\n' + text

                print("=== First 1000 chars of PDF ===")
                print(full_text[:1000])

                lines = full_text.split('\n')
                current_subject = ''
                current_text = ''
                for line in lines:
                    if pattern.match(line):
                        print("✅ Matched header line:", line)
                        if current_subject:
                            subjects.append({
                                'subject': current_subject.strip(),
                                'text': current_text.strip(),
                                'source': filename
                            })
                        current_subject = line
                        current_text = line + '\n'
                    else:
                        current_text += line + '\n'

                if current_subject:
                    subjects.append({
                        'subject': current_subject.strip(),
                        'text': current_text.strip(),
                        'source': filename
                    })
    return subjects

def embed_subjects(subjects):
    for subj in subjects:
        subj['embedding'] = model.encode(subj['text'])
    return subjects

# Streamlit app
st.title("📚 Smart College PDF Subject Syllabus Bot")

with st.spinner("Loading subjects..."):
    subjects = load_subject_chunks('docs')
    st.write(f"📦 Found {len(subjects)} subjects")

    if not subjects:
        st.error("❗ No subjects found. Regex failed or PDFs empty.")
        st.stop()  # stop safely if nothing found

    subjects = embed_subjects(subjects)
    st.success(f"✅ Loaded {len(subjects)} subjects from PDFs.")

query = st.text_input("Ask your question:")

if query:
    if not subjects:
        st.error("❗ Cannot search: no subjects loaded.")
    else:
        query_emb = model.encode(query)
        scores = [
            (
                subj['subject'],
                subj['text'],
                subj['source'],
                float(cosine_similarity([query_emb], [subj['embedding']])[0][0])
            )
            for subj in subjects
        ]

        if not scores:
            st.write("❓ No matches found. Try another question.")
        else:
            scores.sort(key=lambda x: x[3], reverse=True)
            top_subject, top_text, top_file, top_score = scores[0]

            if top_score > 0.4:
                st.subheader(f"✅ Best match (similarity: {top_score:.2f}) from {top_file}")
                st.write(f"### {top_subject}")
                st.write(top_text)
            else:
                st.write("❓ Sorry, I couldn't find that subject in the PDFs.")
