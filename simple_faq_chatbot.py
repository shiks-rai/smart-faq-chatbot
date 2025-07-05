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
    pattern = re.compile(r'.*Course.*', re.IGNORECASE)  # loosen regex

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

                # debug: print what we actually got
                print("=== First 1000 chars of PDF ===")
                print(full_text[:1000])

                lines = full_text.split('\n')
                current_subject = ''
                current_text = ''
                for line in lines:
                    if pattern.match(line):
                        print("✅ Matched header line:", line)  # debug print
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

                # Save last subject
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
    st.write(f"📦 Found {len(subjects)} subjects")  # always show this

    if subjects:
        subjects = embed_subjects(subjects)
        st.success(f"✅ Loaded {len(subjects)} subjects from PDFs.")
    else:
        st.error("❗ No subjects found. Regex failed or PDFs empty.")
        st.stop()  # stop safely if nothing found

query = st.text_input("Ask your question:")

if query:
    if subjects:  # double-check subjects loaded
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

        if scores:
            scores.sort(key=lambda x: x[3], reverse=True)
            top_subject, top_text, top_file, top_score = scores[0]

            if top_score > 0.4:
                st.subheader(f"✅ Best match (similarity: {top_score:.2f}) from {top_file}")
                st.write(f"### {top_subject}")
                st.write(top_text)
            else:
                st.write("❓ Sorry, I couldn't find that subject in the PDFs.")
        else:
            st.write("❓ No matches found.")
    else:
        st.write("❗ Cannot search: no subjects loaded.")
