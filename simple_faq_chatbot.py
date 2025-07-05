import os
import re
import PyPDF2
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_subjects_with_modules(pdf_folder):
    subjects = []
    # Regex: lines starting with "Course:" (case‑insensitive)
    course_pattern = re.compile(r'.*Course.*', re.IGNORECASE)
    # Regex: lines starting with Module
    module_pattern = re.compile(r'\s*Module\s*[-–]?\s*\d+', re.IGNORECASE)

    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            path = os.path.join(pdf_folder, filename)
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ''
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text += '\n' + t

                lines = text.split('\n')
                current_subject = None
                current_modules = []
                current_module = None
                current_text = ''
                for line in lines:
                    if course_pattern.match(line):
                        # save previous subject
                        if current_subject:
                            if current_module:
                                current_modules.append({'module': current_module, 'text': current_text.strip()})
                            subjects.append({'subject': current_subject.strip(), 'modules': current_modules, 'file': filename})
                        current_subject = line
                        current_modules = []
                        current_module = None
                        current_text = ''
                    elif module_pattern.match(line):
                        # save previous module
                        if current_module:
                            current_modules.append({'module': current_module, 'text': current_text.strip()})
                        current_module = line.strip()
                        current_text = ''
                    else:
                        current_text += line.strip() + ' '

                # save last module & subject
                if current_module:
                    current_modules.append({'module': current_module, 'text': current_text.strip()})
                if current_subject:
                    subjects.append({'subject': current_subject.strip(), 'modules': current_modules, 'file': filename})

    return subjects

def embed_subjects(subjects):
    for subj in subjects:
        subj['embedding'] = model.encode(subj['subject'])
    return subjects

st.title("📚 Smart Syllabus Bot with Modules Table")

with st.spinner("Loading PDFs and extracting modules..."):
    subjects = load_subjects_with_modules('docs')
    st.write(f"📦 Found {len(subjects)} subjects from PDFs")
    if not subjects:
        st.error("❗ No subjects found. Check PDF structure or regex.")
        st.stop()
    subjects = embed_subjects(subjects)
    st.success("✅ Loaded subjects & modules.")

query = st.text_input("Ask: e.g. syllabus for full stack development")

if query:
    query_emb = model.encode(query)
    scores = []
    for subj in subjects:
        sim = float(cosine_similarity([query_emb], [subj['embedding']])[0][0])
        scores.append( (subj, sim) )
    scores.sort(key=lambda x: x[1], reverse=True)

    best_subj, best_score = scores[0]
    if best_score > 0.4:
        st.subheader(f"✅ Showing syllabus for: {best_subj['subject']} (score: {best_score:.2f})")

        for mod in best_subj['modules']:
            st.markdown(f"### 📌 {mod['module']}")
            st.write(mod['text'])
    else:
        st.write("❓ Couldn't find a matching subject.")
