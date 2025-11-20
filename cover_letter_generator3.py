# =========================================================
# ✉️ cover_letter_generator.py — AI Cover Letter Generator
# =========================================================

import os
import re
import json
import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from langchain_openai import ChatOpenAI

# ---------------------------------
# Environment Setup & Model Loading
# ---------------------------------
load_dotenv()

# Initialize embedding + LLM once globally
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# =========================================================
# 🧾 1. CV Extraction & Semantic Profiling
# =========================================================
@st.cache_data(show_spinner=False)
def extract_text_from_cv(pdf_path: str) -> str:
    """Extracts raw text from the uploaded CV PDF."""
    text = ""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"CV not found: {pdf_path}")

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

    return text.strip()

def build_cv_profile(cv_text: str) -> dict:
    """Builds a lightweight semantic profile from CV text."""
    if not cv_text:
        return {"summary": "", "skills": [], "experience": []}

    # Take the top few lines as a quick summary snippet
    lines = [l.strip() for l in cv_text.split("\n") if l.strip()]
    summary = " ".join(lines[:6])

    # Extract known technical keywords
    skills = re.findall(
        r"\b(Python|R|SQL|Azure|AWS|GCP|ML|AI|Statistics|Deep Learning|Causal Inference|Forecasting|Model Validation|Risk)\b",
        cv_text,
        flags=re.IGNORECASE,
    )

    # Identify relevant experience types
    experiences = re.findall(
        r"(Data Scientist|Research|AI|ML|Optimization|Predictive|Model Validation|Risk Modeling)",
        cv_text,
        flags=re.IGNORECASE,
    )

    return {
        "summary": summary,
        "skills": sorted(set([s.title() for s in skills])),
        "experience": sorted(set([e.title() for e in experiences])),
    }

# =========================================================
# 📄 2. Job Parsing
# =========================================================
def extract_job_profile(job_dict: dict) -> dict:
    """Extracts concise info about a job posting for letter generation."""
    desc = job_dict.get("description", "") or ""
    title = job_dict.get("title", "Unknown Role")
    company = job_dict.get("company", "Unknown Company")

    # Extract technical keywords mentioned in the job post
    skills = re.findall(
        r"\b(Python|SQL|R|Azure|AWS|ML|AI|Statistics|Deep Learning|Forecasting|Risk|NLP|GenAI|LLM)\b",
        desc,
        flags=re.IGNORECASE,
    )

    return {
        "title": title,
        "company": company,
        "desc": desc[:600],  # first 600 chars for context
        "skills": sorted(set([s.title() for s in skills])),
    }

# =========================================================
# 🧠 3. Cover Letter Generation (LLM)
# =========================================================
def generate_cover_letter(cv_profile: dict, job_profile: dict) -> str:
    """Generates a human, concise 1-page cover letter using LLM."""
    overlap = set(cv_profile["skills"]).intersection(set(job_profile["skills"]))
    intro_skills = ", ".join(list(overlap)[:5]) if overlap else "data science and AI"

    prompt = f"""
You are a professional career writer. Write a concise, natural 1-page cover letter
for a candidate applying to **{job_profile['title']}** at **{job_profile['company']}**.

Candidate summary:
{cv_profile['summary']}

Relevant experience areas: {', '.join(cv_profile['experience'][:5]) or 'data science, AI, and analytics'}  
Overlapping technical skills: {intro_skills}

Company/job context:
{job_profile['desc']}

Write in a confident, fluent tone — avoid clichés like “fast learner” or “team player.”
Keep it under 250 words, well-structured with:
1. A short intro expressing genuine interest in the company/role.
2. 2 body paragraphs linking achievements to requirements.
3. A closing paragraph inviting next steps politely.
"""

    try:
        response = llm.invoke(prompt)
        text = response.content.strip() if hasattr(response, "content") else str(response)
    except Exception as e:
        text = f"[Error generating letter: {e}]"

    return text

# =========================================================
# 💾 4. Save Result
# =========================================================
def save_cover_letter(company: str, title: str, text: str) -> str:
    """Saves generated cover letter text to /data/cover_letters."""
    os.makedirs("data/cover_letters", exist_ok=True)
    safe_company = re.sub(r"[^a-zA-Z0-9]", "_", company)
    safe_title = re.sub(r"[^a-zA-Z0-9]", "_", title)
    filename = f"data/cover_letters/{safe_company}_{safe_title}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

    return filename

# =========================================================
# 🚀 5. Streamlit Helper Wrapper (Optional)
# =========================================================
def generate_and_display_cover_letters(cv_path: str, ranked_jobs: list, top_n: int = 5):
    """Convenient wrapper to integrate directly into Streamlit UI."""
    cv_text = extract_text_from_cv(cv_path)
    cv_profile = build_cv_profile(cv_text)

    st.info("Generating AI-based tailored cover letters...")
    progress = st.progress(0)
    letters = []

    for i, job in enumerate(ranked_jobs[:top_n], 1):
        job_profile = extract_job_profile(job)
        letter_text = generate_cover_letter(cv_profile, job_profile)
        file_path = save_cover_letter(job_profile["company"], job_profile["title"], letter_text)

        letters.append({"company": job_profile["company"], "title": job_profile["title"], "path": file_path})
        progress.progress(i / top_n)

    progress.empty()
    st.success(f"✅ Generated {len(letters)} cover letters successfully!")

    # Display previews
    for letter in letters:
        st.markdown(f"### ✉️ {letter['company']} — {letter['title']}")
        with open(letter["path"], "r", encoding="utf-8") as f:
            content = f.read()
        st.text_area("Preview", content, height=250, key=letter["title"])
        st.download_button(
            label="⬇️ Download",
            data=content,
            file_name=os.path.basename(letter["path"]),
            mime="text/plain",
        )

    return letters
