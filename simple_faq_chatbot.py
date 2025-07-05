import os
import re
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_subject_chunks(pdf_folder):
    subjects = []
    pattern = re.compile(r'Course:.*', re.IGNORECASE)

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

                # split text by lines matching 'Course: ...'
                lines = full_text.split('\n')
                current_subject = ''
                current_text = ''
                for line in lines:
                    if pattern.match(line):
                        # save previous subject
                        if current_subject:
                            subjects.append({
                                'subject': current_subject.strip(),
                                'text': current_text.strip(),
