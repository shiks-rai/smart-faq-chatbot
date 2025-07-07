import os
import re
import PyPDF2
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def load_subjects_with_modules(pdf_folder):
    subjects = []
    # Match lines containing 'Course' (subject header)
    course_pattern = re.compile(r'.*Course.*', re.IGNORECASE)
    # Match anything starting with "Module" and a number, allowing dashes, colons etc.
    module_pattern = re.compile(r'\s*Module\s*[-–:]?\s*\d+.*', re.IGNORECASE)

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

                # Split into lines
                lines = text.split('\n')

                # ✅ Debug: print first 50 lines
                print(f"\n=== First 50 lines from {filename} ===")
                for line in lines[:50]:
                    print(repr(line))

                current_subject = None
                current_modules = []
                current_module = None
                current_text = ''

                for line in lines:
                    if course_pattern.search(line):
                        print(f"✅ Found subject line: {line}")
                        if current_subject:
                            if current_module:
                                current_modules.append({'module': current_module, 'text': current_text.strip()})
                            subjects.append({'subject': current_subject.strip(), 'modules': current_modules, 'file': filename})
                        current_subject = line
                        current_modules = []
                        current_module = None
                        current_text = ''
                    elif module_pattern.match(line):
                        print(f"🔹 Found module line: {line}")
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

    # ✅ Debug: print summary of subjects & modules
    print("\n=== Summary: Subjects & Modules ===")
    for subj in subjects:
        print(f"Subject: {subj['subject']} (from file: {subj['file']})")
        print(f"Modules found: {len(subj['modules'])}")
        for mod in subj['modules']:
            print(f"- {mod['module']} | text length: {len(mod['text'])}")

    return subjects

def embed_subjects(subjects):
    for subj in subjects:
        subj['embedding'] = model.encode(subj['subject'])
    return subjects

st.title("📚 Smart Syllabus Bot with Modules Table")

with st.spinner("Loading PDFs and extracting modules..."):
    subjects = load_subjects_with_modules('docs')  # your PDFs should be inside a folder called 'docs'
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

        # Show modules table
        table_rows = []
        for mod in best_subj['modules']:
            table_rows.append({
                "Module": mod['module'],
                "Topics (preview)": mod['text'][:200] + "..."
            })

        if table_rows:
            st.table(table_rows)
        else:
            st.warning("⚠ No modules found inside this subject. Check parsing.")
    else:
        st.write("❓ Couldn't find a matching subject.")
