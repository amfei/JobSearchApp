# =====================================================
# 🧭 AI-Powered Job Search Ranker (Final Minimal Update)
# =====================================================
import os
import streamlit as st
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import plotly.express as px
import umap.umap_ as umap

# --- Disable telemetry noise ---
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Local imports ---
from style import apply_custom_style
from job_search_rankerv3 import (
    extract_text_from_pdf,
    scrape_linkedin_jobs,
    retrieve_jobs_from_chromadb,
    store_jobs_in_chromadb,
    clear_collection_if_any,
    hybrid_rank,
    send_email,
    visualize_embeddings_plotly,
    JOB_TITLES,
    LOCATIONS,
    EXCLUDED_TITLES,
    TOP_N,
)
from job_search_ranker3 import graph


# =====================================================
# 1️⃣ Page Setup
# =====================================================
def setup_page():
    st.set_page_config(page_title="AI Job Search Ranker", page_icon="🧠", layout="wide")
    apply_custom_style()
    st.title("🧠 AI-Assisted Job Search Ranker")
    st.caption("Two-stage semantic job matcher: BM25 filter + embedding reranker.")


# =====================================================
# 2️⃣ Technical Overview
# =====================================================
def show_technical_overview():
    st.markdown("""
    #### 🧬 Technical Overview
    • **SentenceTransformer embeddings** (`all-mpnet-base-v2`)  
    • **Two-Stage Retrieval:** BM25 lexical recall → cosine-similarity semantic rerank  
    • **Vector Store:** ChromaDB  
    • **Ranking Layer:** Cosine similarity (semantic dominance)  
    • **LangGraph:** LLM-powered orchestration  
    """)


# =====================================================
# 3️⃣ Sidebar Inputs
# =====================================================
def get_sidebar_inputs():
    with st.sidebar:
        st.header("⚙️ Search Parameters")

        job_titles = [t.strip() for t in st.text_input(
            "Job Titles (comma-separated)", ", ".join(JOB_TITLES)
        ).split(",") if t.strip()]

        locations = [l.strip() for l in st.text_input(
            "Locations (comma-separated)", ", ".join(LOCATIONS)
        ).split(",") if l.strip()]

        days_filter = st.slider("Days posted within", 1, 30, 12)
        num_jobs = st.slider("Number of jobs to fetch", 10, 200, 150)
        top_n = st.slider("Top N job matches", 5, 50, TOP_N)

        excluded_titles = [x.strip() for x in st.text_area(
            "Exclude titles containing (comma-separated):",
            ", ".join(EXCLUDED_TITLES),
        ).split(",") if x.strip()]

        st.markdown("---")
        send_email_opt = st.checkbox("📧 Send results via email", value=False)
        recipient_email = st.text_input("Recipient email", "amir.h.feizi@outlook.com")
        generate_cover_opt = st.checkbox("✉️ Generate Cover Letters", value=True)

        return job_titles, locations, days_filter, num_jobs, top_n, excluded_titles, send_email_opt, recipient_email, generate_cover_opt


# =====================================================
# 4️⃣ CV Upload
# =====================================================
def upload_cv():
    st.markdown('<div class="section-title">📄 Step 1 — Upload Your CV</div>', unsafe_allow_html=True)
    cv_file = st.file_uploader("Upload your CV (PDF)", type=["pdf"])
    if not cv_file:
        st.warning("⚠️ Please upload your CV to continue.")
        st.stop()

    os.makedirs("data/uploaded_cvs", exist_ok=True)
    temp_path = os.path.join("data/uploaded_cvs", "current_cv.pdf")
    with open(temp_path, "wb") as f:
        f.write(cv_file.read())

    try:
        cv_text = extract_text_from_pdf(temp_path)
        st.success(f"✅ CV uploaded successfully ({len(cv_text)} chars).")
        st.session_state["uploaded_cv_path"] = temp_path
        st.session_state["cv_text"] = cv_text
        return cv_text
    except Exception as e:
        st.error(f"❌ Failed to read CV: {e}")
        st.stop()


# =====================================================
# 5️⃣ Embedding Model
# =====================================================
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


# =====================================================
# 6️⃣ Action Buttons (email removed)
# =====================================================
def render_action_buttons():
    colA, colB, colC, colD = st.columns(4)
    with colA: scrape_now = st.button("🔎 Fetch Jobs")
    with colB: rank_now = st.button("🏆 Rank Matches")
    with colC: evaluate_now = st.button("🧮 Evaluate Results")
    with colD: visualize_now = st.button("📊 Visualize Embeddings")
    return scrape_now, rank_now, evaluate_now, visualize_now


# =====================================================
# 7️⃣ Fetch and Store Jobs
# =====================================================
def fetch_and_store_jobs(scrape_now, job_titles, locations, num_jobs, days_filter, excluded_titles, embed_model):
    jobs = []
    if scrape_now:
        st.info("🧹 Clearing old jobs before fetching new ones…")
        clear_collection_if_any()
        with st.spinner("🔍 Fetching jobs…"):
            jobs = scrape_linkedin_jobs(job_titles, locations, num_jobs, days_filter, excluded_titles)
        if jobs:
            store_jobs_in_chromadb(jobs, embed_model)
            st.success(f"✅ {len(jobs)} jobs fetched and stored.")
        else:
            st.warning("⚠️ No jobs found.")
    return jobs


# =====================================================
# 8️⃣ Ranking & Cover Letter
# =====================================================
from openai import OpenAI
import re

@st.cache_resource
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ OPENAI_API_KEY missing.")
        st.stop()
    return OpenAI(api_key=api_key)


def extract_contact_info(cv_text):
    lines = [l.strip() for l in cv_text.splitlines() if l.strip()]
    name, city_prov, email, phone, linkedin = "", "", "", "", ""
    if lines:
        name = lines[0]
        if any(x in name.lower() for x in ["curriculum", "resume", "cv"]):
            name = lines[1] if len(lines) > 1 else ""
    for line in lines[:15]:
        m = re.search(r"([A-Za-zÀ-ÿ\s]+,\s?[A-Z]{2})", line)
        if m:
            city_prov = m.group(1).strip()
            break
    m = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", cv_text); email = m.group(0) if m else ""
    m = re.search(r"(\+?\d{1,3}[\s-]?)?(\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4})", cv_text); phone = m.group(0) if m else ""
    m = re.search(r"(https?://)?(www\.)?linkedin\.com/[^\s]+", cv_text); linkedin = m.group(0) if m else ""
    return name.strip(), city_prov.strip(), email.strip(), phone.strip(), linkedin.strip()


def generate_cover_letter(cv_text, job):
    client = get_openai_client()
    name, city_prov, email, phone, linkedin = extract_contact_info(cv_text)
    today = datetime.now().strftime("%B %d, %Y")
    prompt = f"""
    Write a concise, professional cover letter (≈200 words) for this job using the candidate details.

    Date: {today}
    Location: {city_prov or "City, Province"}
    Name: {name or "Candidate Name"}
    Email: {email} Phone: {phone} LinkedIn: {linkedin}

    === CV EXCERPT ===
    {cv_text[:3500]}

    === JOB ===
    Title: {job.get('title')}
    Company: {job.get('company')}
    Description: {job.get('description','')}
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are an expert cover-letter writer."},
                      {"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=600,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error generating cover letter: {e}]"


def rank_and_display_jobs(rank_now, cv_text, stored_jobs, embed_model, top_n, generate_cover_opt):
    if not rank_now:
        return []

    with st.spinner("🏗️ Retrieving and reranking jobs…"):
        ranked = hybrid_rank(cv_text, stored_jobs, embed_model, top_n)

    if not ranked:
        st.warning("⚠️ No ranked jobs found.")
        return []

    # 🧩 Detect output format
    if isinstance(ranked[0], dict):
        # → plain list of job dicts (no scores)
        ranked = [(job.get("score", 0.0), job) for job in ranked]
    elif len(ranked[0]) != 2:
        # → unknown structure (e.g., 3 values)
        #st.warning("⚠️ Unexpected structure returned by hybrid_rank(); attempting to adapt.")
        ranked = [(r[0], r[1]) if len(r) >= 2 else (0.0, r) for r in ranked]

    st.success("🏆 Semantic reranking complete.")
    st.markdown("### 🥇 Top Matched Jobs")

    for i, (score, job) in enumerate(ranked, 1):
        st.markdown(f"**{i}. {job.get('title','N/A')} — {job.get('company','N/A')}**")
        st.caption(job.get("location", "N/A"))
        st.markdown(f"[🔗 View Posting]({job.get('link', '#')}) **Cosine Similarity:** `{score:.3f}`")

        if generate_cover_opt:
            with st.spinner(f"✉️ Generating cover letter for {job.get('company','N/A')}…"):
                letter = generate_cover_letter(cv_text, job)
            with st.expander("📄 Show Cover Letter"):
                st.markdown(
                    f"<div style='white-space:pre-wrap;font-family:monospace;'>{letter}</div>",
                    unsafe_allow_html=True,
                )

    return ranked

# =====================================================
# 🚀 Main Controller
# =====================================================
def main():
    setup_page()
    show_technical_overview()
    job_titles, locations, days_filter, num_jobs, top_n, excluded_titles, send_email_opt, recipient_email, generate_cover_opt = get_sidebar_inputs()

    cv_text = upload_cv()
    embed_model = load_embedding_model()
    scrape_now, rank_now, evaluate_now, visualize_now = render_action_buttons()

    fetch_and_store_jobs(scrape_now, job_titles, locations, num_jobs, days_filter, excluded_titles, embed_model)
    stored_jobs = retrieve_jobs_from_chromadb()
    ranked = rank_and_display_jobs(rank_now, cv_text, stored_jobs, embed_model, top_n, generate_cover_opt)

    # Separate Evaluation Button
    if evaluate_now and "uploaded_cv_path" in st.session_state:
        st.info("🧮 Running LangGraph evaluation…")
        with st.spinner("Executing graph workflow..."):
            initial_state = {
                "cv_path": st.session_state["uploaded_cv_path"],
                "cv_text": cv_text,
                "stored_jobs": stored_jobs,
                "top_jobs": ranked,
                "metrics": {},
                "send_email_opt": send_email_opt,
            }
            final_state = graph.invoke(initial_state)
        metrics = final_state.get("metrics", {})
        if metrics:
            st.markdown("### 📊 Evaluation Metrics (LangGraph)")
            st.json(metrics)
        else:
            st.warning("⚠️ No evaluation metrics found.")

    # Separate Visualization Button
    if visualize_now:
        visualize_embeddings_plotly(cv_text, ranked, embed_model)

    # Email only if chosen from sidebar
    if send_email_opt and ranked:
        lines = [f"Top Job Matches — {datetime.now():%Y-%m-%d %H:%M}"]
        for i, (score, j) in enumerate(ranked, 1):
            lines.append(f"{i}. {j['title']} at {j['company']} ({j['location']}) — Score {score:.4f}")
            lines.append(f"  {j['link']}")
        email_body = "\n".join(lines)
        send_email(recipient_email, "Daily AI-Powered Job Matches", email_body)
        st.success("✅ Email sent successfully.")

    st.markdown("---")
    st.caption("💼 Built by Amir Feizi")


# =====================================================
# 🏁 Run
# =====================================================
if __name__ == "__main__":
    main()
