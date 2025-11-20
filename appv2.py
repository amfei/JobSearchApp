# 🧭 AI-Powered Job Search Ranker (Technical Demo Edition)
# =====================================================
import os
import streamlit as st
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --- Disable telemetry noise ---
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Local imports ---
from style import apply_custom_style  #  modular style injector
from job_search_rankerv2 import (
    extract_text_from_pdf,
    scrape_linkedin_jobs,
    retrieve_jobs_from_chromadb,
    store_jobs_in_chromadb,
    clear_collection_if_any,
    hybrid_rank,
    send_email,
    JOB_TITLES,
    LOCATIONS,
    EXCLUDED_TITLES,
    TOP_N,
)

# =====================================================
# ⚙️ Streamlit Page Setup
# =====================================================
st.set_page_config(page_title="AI Job Search Ranker", page_icon="🧠", layout="wide")
apply_custom_style()

st.title("🧠 AI-Assisted Job Search Ranker")
st.caption(
    "An intelligent information retrieval (IR) system that matches your CV to the most relevant job postings using a **hybrid semantic + lexical ranking pipeline**."
)

# =====================================================
# 🔬 Technical Overview
# =====================================================
st.markdown("""
#### 🧬 Technical Overview
This app demonstrates an **end-to-end NLP-driven job matching pipeline** that integrates:
- **Document Embeddings** via `SentenceTransformer (all-mpnet-base-v2)` → 768-dim dense vectors
- **Hybrid Retrieval**: Combines lexical similarity (**BM25**) and semantic similarity (**cosine distance**)
- **Vector Store**: Persistent embedding storage in `ChromaDB`
- **Ranking Layer**: Weighted fusion of BM25 + cosine similarity
- **Pipeline Orchestration**: Streamlit + LangGraph abstraction layer  
""")

st.caption("🧩 Conceptually similar to a **Retrieval-Augmented Generation (RAG)** system, but optimized for job recommendation instead of text generation.")

# =====================================================
# 🎛 Sidebar Configuration
# =====================================================
with st.sidebar:
    st.header("⚙️ Search Parameters")

    job_titles = st.text_input(
        "Job Titles (comma-separated)", ", ".join(JOB_TITLES)
    ).split(",")
    job_titles = [t.strip() for t in job_titles if t.strip()]

    locations = st.text_input(
        "Locations (comma-separated)", ", ".join(LOCATIONS)
    ).split(",")
    locations = [l.strip() for l in locations if l.strip()]

    days_filter = st.slider("Days posted within", 1, 30, 12)
    num_jobs = st.slider("Number of jobs to fetch", 10, 200, 150)
    top_n = st.slider("Top N job matches", 5, 50, TOP_N)

    st.markdown("### 🚫 Excluded Job Titles")
    excluded_titles_input = st.text_area(
        "Exclude titles containing any of these words (comma-separated):",
        ", ".join(EXCLUDED_TITLES),
        help="Jobs containing these terms in title will be ignored.",
    )
    excluded_titles = [x.strip() for x in excluded_titles_input.split(",") if x.strip()]

    st.markdown("---")
    send_email_opt = st.checkbox("Send results via email", value=False)
    recipient_email = st.text_input("Recipient email", "amir.h.feizi@outlook.com")

    st.markdown("---")
    st.caption("📎 Upload your CV to extract text and compute vector embeddings.")

# =====================================================
# 📄 Step 1 — Upload CV
# =====================================================
st.markdown('<div class="section-title">📄 Step 1 — Upload Your CV (Document Corpus Input)</div>', unsafe_allow_html=True)
cv_file = st.file_uploader(
    "Upload your CV (PDF)",
    type=["pdf"],
    help="Text will be extracted and transformed into vector embeddings.",
)

if cv_file:
    os.makedirs("data", exist_ok=True)
    temp_path = os.path.join("data", f"cv_{datetime.now():%Y%m%d_%H%M%S}.pdf")
    with open(temp_path, "wb") as f:
        f.write(cv_file.read())
    try:
        cv_text = extract_text_from_pdf(temp_path)
        st.success(f"✅ CV uploaded and text extracted successfully ({len(cv_text)} characters).")
    except Exception as e:
        st.error(f"❌ Failed to read CV: {e}")
        st.stop()
else:
    st.warning("⚠️ Please upload your CV to continue.")
    st.stop()

# =====================================================
# 🤖 Step 2 — Load Embedding Model
# =====================================================
st.markdown('<div class="section-title">🤖 Embedding Model — SentenceTransformer</div>', unsafe_allow_html=True)
st.info("""
Model: **all-mpnet-base-v2** — optimized for semantic similarity tasks.  
Transforms each job description and CV into a **768-dimensional dense vector**.  
Similarity is measured using **cosine distance** in this latent space.
""")

@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

embed_model = load_model()

# =====================================================
# 🧩 Step 3 — Actions
# =====================================================
colA, colB, colC, colD = st.columns(4)
with colA:
    scrape_now = st.button("🔎 Fetch Jobs")
with colB:
    rank_now = st.button("🏆 Rank Matches")
with colC:
    email_now = st.button("📧 Email Results")
with colD:
    reset_now = st.button("🧹 Clear Database")

# =====================================================
# 🌐 Step 4 — Fetch LinkedIn Jobs
# =====================================================
jobs = []
if scrape_now:
    st.info("🔍 Fetching LinkedIn job postings...")
    with st.spinner("⏳ Running web scraping + normalization..."):
        jobs = scrape_linkedin_jobs(job_titles, locations, num_jobs, days_filter, excluded_titles)
    if jobs:
        clear_collection_if_any()
        store_jobs_in_chromadb(jobs, embed_model)
        st.success(f"✅ {len(jobs)} jobs fetched and stored in ChromaDB.")
    else:
        st.warning("⚠️ No jobs found or scraping failed.")

# =====================================================
# 📂 Step 5 — Retrieve Stored Jobs
# =====================================================
stored_jobs = retrieve_jobs_from_chromadb()
st.markdown(f"### 🗂️ Stored Jobs in Database: `{len(stored_jobs)}`")

# =====================================================
# 🧠 Step 6 — Rank Jobs (Hybrid Model)
# =====================================================
ranked = []
if rank_now:
    st.info("Computing hybrid semantic + lexical relevance scores...")
    with st.spinner("⚙️ Running BM25 + embedding similarity..."):
        ranked = hybrid_rank(cv_text, stored_jobs, embed_model, top_n)

    if not ranked:
        st.warning("⚠️ No ranked results found.")
    else:
        st.success("🏆 Ranking complete.")
        st.markdown('<div class="section-title">🥇 Top Ranked Jobs</div>', unsafe_allow_html=True)

        for i, (score, job) in enumerate(ranked, 1):
            st.markdown(
                f"""
                <div class="card">
                    <b>{i}. {job['title']}</b> — {job['company']} ({job['location']})<br>
                    <span style='color:#90A4AE;'>Score: {score:.4f}</span><br>
                    <a href="{job['link']}" target="_blank">🔗 View Posting</a>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("🧠 Technical Details: Hybrid Ranking Function"):
            st.markdown("""
            **Formula:**  
            f(job, CV) = 0.5 × *BM25(job, CV)* + 0.5 × *cosine_similarity(embedding(job), embedding(CV))*  

            - **BM25:** Classical probabilistic retrieval model capturing keyword relevance  
            - **Cosine Similarity:** Measures semantic alignment in embedding space  
            - **Hybrid Fusion:** Balances lexical precision and semantic recall  
            """)

# =====================================================
# 📈 Step 7 — 3D Embedding Visualization (PCA)
# =====================================================
if rank_now and ranked:
    with st.expander("📈 Visualize Semantic Space (3D PCA Projection)"):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

        job_texts = [f"{j['title']} - {j['company']}" for _, j in ranked]
        job_embs = [embed_model.encode(jt) for jt in job_texts]
        cv_emb = embed_model.encode(cv_text)

        # Stack and normalize embeddings before PCA (better for cosine-based geometry)
        X = np.vstack([cv_emb] + job_embs)
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        pca = PCA(n_components=3)
        coords = pca.fit_transform(X)
        df = pd.DataFrame(coords, columns=["x", "y", "z"])
        df["label"] = ["CV"] + [f"Job {i+1}" for i in range(len(job_texts))]

        # Compute cosine similarities for color intensity
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity([cv_emb], job_embs).flatten()
        df["similarity"] = np.insert(sims, 0, 1.0)  # insert CV as 1.0

        # --- Plot in 3D ---
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Jobs: color mapped by cosine similarity
        sc = ax.scatter(
            df["x"][1:], df["y"][1:], df["z"][1:],
            c=df["similarity"][1:], cmap="Blues", s=60, alpha=0.8, label="Jobs"
        )

        # CV: highlighted in orange
        ax.scatter(
            df["x"][0], df["y"][0], df["z"][0],
            c="#FFA726", s=160, edgecolors="k", label="CV", marker="o"
        )

        for i, label in enumerate(df["label"]):
            ax.text(df["x"][i], df["y"][i], df["z"][i], label, fontsize=8)

        ax.set_title("3D PCA Projection of Embeddings", fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.legend()
        fig.colorbar(sc, ax=ax, shrink=0.6, label="Cosine Similarity")

        st.pyplot(fig)

        st.markdown("""
        **Interpretation:**  
        - Each point represents an embedding projected from 768D → 3D via PCA.  
        - The orange dot is your CV; blue intensity reflects cosine similarity to the CV vector.  
        - Closer points (especially in direction, not just distance) indicate higher semantic alignment.  
        - Unlike 2D, this 3D projection preserves more of the true angular geometry.
        """)


# =====================================================
# 📧 Step 8 — Email Results
# =====================================================
if email_now:
    if not ranked:
        ranked = hybrid_rank(cv_text, stored_jobs, embed_model, top_n)

    lines = [f"Top Job Matches — {datetime.now():%Y-%m-%d %H:%M}"]
    for i, (score, j) in enumerate(ranked, 1):
        tag = " (High Priority)" if score > 0.7 else ""
        lines.append(f"{i}. {j['title']} at {j['company']} ({j['location']}) — Score {score:.4f}{tag}")
        lines.append(f"   {j['link']}")

    email_body = "\n".join(lines)
    st.text_area("📨 Email Preview", email_body, height=250)

    if send_email_opt:
        send_email(recipient_email, "Daily AI-Powered Job Matches", email_body)
        st.success("✅ Email sent successfully.")
    else:
        st.info("📭 Email not sent (preview mode only).")

# =====================================================
# 🧹 Step 9 — Clear Database
# =====================================================
if reset_now:
    with st.spinner("🧹 Clearing all stored jobs..."):
        clear_collection_if_any()
        st.success("✅ Database cleared.")

# =====================================================
# 📊 Step 10 — Pipeline Summary
# =====================================================
st.markdown("---")
st.subheader("📊 End-to-End Pipeline Summary")
st.markdown("""
**1️⃣ Data Ingestion:** LinkedIn job listings scraped via HTTP requests + BeautifulSoup.  
**2️⃣ Text Preprocessing:** Normalization, tokenization, and minimal cleaning.  
**3️⃣ Vectorization:** Each document encoded into dense vectors (768-dim).  
**4️⃣ Storage Layer:** ChromaDB manages persistent vector store.  
**5️⃣ Retrieval:** Hybrid (BM25 + dense embeddings).  
**6️⃣ Ranking:** Feature fusion via weighted combination.  
**7️⃣ Evaluation:** Cosine scores + visualization.  
**8️⃣ Automation:** Modular LangGraph workflow integrated with Streamlit UI.  
""")

# =====================================================
# 💬 Footer
# =====================================================
st.markdown("---")
st.caption("💼 Built by Amir Feizi | Powered by SentenceTransformers · BM25 · ChromaDB · LangGraph · Streamlit") 