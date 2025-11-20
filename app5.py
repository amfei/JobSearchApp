# =====================================================
# 🧭 AI-Powered Job Search Ranker (v5, modular LangGraph)
# =====================================================
import os
import logging
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import umap.umap_ as umap
import plotly.express as px

from sentence_transformers import SentenceTransformer, CrossEncoder

# --- Disable telemetry noise ---
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Mute residual Chroma logs ---
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.CRITICAL)

# --- Local imports (STYLE stays, FUNCTIONS moved to job_ranker5) ---
from style import apply_custom_style

from job_ranker5 import (
    DEFAULTS,
    graph,
    visualize_embeddings_plotly,
    run_segment_scan_store,
    run_segment_rank,
    run_segment_evaluate,
    run_segment_cover_letters,
    run_segment_email,
)

# =====================================================
# 1️⃣ Page Setup
# =====================================================
def setup_page():
    st.set_page_config(page_title="AI Job Search Ranker", page_icon="🧠", layout="wide")
    apply_custom_style()
    st.title("🧠 AI-Assisted Job Match Engine")

# =====================================================
# 2️⃣ Technical Overview
# =====================================================
def show_technical_overview():
    st.markdown("""
    #### 🧬 Technical Overview
    • **SentenceTransformer embeddings** (configurable via env)  
    • **Three-Stage Scoring:** BM25 → Cosine → Cross-Encoder fusion  
    • **Vector Store:** ChromaDB (persistent)  
    • **Orchestration:** **LangGraph** modular nodes (scan/store → rank → evaluate → cover-letter → email)  
    """)

# =====================================================
# 3️⃣ Sidebar Inputs
# =====================================================
def get_sidebar_inputs():
    with st.sidebar:
        st.header("⚙️ Search Parameters")

        job_titles = [t.strip() for t in st.text_input(
            "Job Titles (comma-separated)", ", ".join(DEFAULTS["JOB_TITLES"])
        ).split(",") if t.strip()]

        locations = [l.strip() for l in st.text_input(
            "Locations (comma-separated)", ", ".join(DEFAULTS["LOCATIONS"])
        ).split(",") if l.strip()]

        days_filter = st.slider("Days posted within", 1, 30, DEFAULTS["DAYS_FILTER"])
        num_jobs = st.slider("Number of jobs to fetch", 10, 150, DEFAULTS["NUM_JOBS"])
        top_n = st.slider("Top N job matches", 5, 30, DEFAULTS["TOP_N"])

        excluded_titles = [x.strip() for x in st.text_area(
            "Exclude titles containing (comma-separated):",
            ", ".join(DEFAULTS["EXCLUDED_TITLES"]),
        ).split(",") if x.strip()]

        st.markdown("---")
        alpha = st.slider("α (BM25 weight)", 0.0, 1.0, DEFAULTS["ALPHA"])
        beta = st.slider("β (CE vs Cosine)", 0.0, 1.0, DEFAULTS["BETA"])
        bm25_top_k = st.slider("BM25 Top-K candidates", 20, 300, DEFAULTS["BM25_TOP_K"])
        scrape_always = st.checkbox("Always scrape (ignore cache)", value=DEFAULTS["SCRAPE_ALWAYS"])

        st.markdown("---")
        send_email_opt = st.checkbox("📧 Send results via email", value=False)
        recipient_email = st.text_input("Recipient email")
        generate_cover_opt = st.checkbox("✉️ Generate Cover Letters", value=False)

        return (
            job_titles, locations, days_filter, num_jobs, top_n,
            excluded_titles, alpha, beta, bm25_top_k,
            scrape_always, send_email_opt, recipient_email, generate_cover_opt
        )

# =====================================================
# 4️⃣ CV Upload
# =====================================================
def upload_cv():
    st.markdown('<div class="section-title">📄 Upload Your CV</div>', unsafe_allow_html=True)
    cv_file = st.file_uploader("Upload your CV (PDF)", type=["pdf"])
    if not cv_file:
        st.warning("⚠️ Please upload your CV to continue.")
        st.stop()

    os.makedirs("data/uploaded_cvs", exist_ok=True)
    temp_path = os.path.join("data/uploaded_cvs", "current_cv.pdf")
    with open(temp_path, "wb") as f:
        f.write(cv_file.read())

    # Store only the path here; graph loads text via node_load_cv
    st.success("✅ CV uploaded successfully.")
    st.session_state["uploaded_cv_path"] = temp_path
    return temp_path

# =====================================================
# 5️⃣ Models
# =====================================================
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model():
    name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    return SentenceTransformer(name), name

@st.cache_resource(show_spinner="Loading cross-encoder model…")
def load_cross_encoder_model():
    name = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    return CrossEncoder(name), name

@st.cache_resource(show_spinner="Preparing LLM (for cover letters)…")
def load_llm():
    from langchain_openai import ChatOpenAI
    name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    return ChatOpenAI(model=name, temperature=0.3), name

# =====================================================
# 6️⃣ Action Buttons (unchanged layout)
# =====================================================
def render_action_buttons():
    if "open_section" not in st.session_state:
        st.session_state["open_section"] = None

    def open_only(section_name):
        st.session_state["open_section"] = section_name

    colA, colB, colC, colD = st.columns(4)

    with colA:
        scan_now = st.button(
            "🔍 Job Scan & Store",
            help=(
                "• Clears old job data from ChromaDB (if scraping)  \n"
                "• Scrapes postings (title, location, full description)  \n"
                "• Embeds & stores in ChromaDB"
            ),
            on_click=lambda: open_only("scan")
        )

    with colB:
        rank_now = st.button(
            "🏆 Rank Matches",
            help=(
                "• Loads CV text  \n"
                "• Retrieves/uses stored jobs  \n"
                "• Hybrid re-ranking (BM25 + Cosine + Cross-Encoder)"
            ),
            on_click=lambda: open_only("rank")
        )

    with colC:
        evaluate_now = st.button(
            "🧩 Evaluate Results",
            help=(
                "• NDCG@5 / NDCG@10  \n"
                "• Cosine mean / std  \n"
                "• Coverage (>0.5)"
            ),
            on_click=lambda: open_only("evaluate")
        )

    with colD:
        visualize_now = st.button(
            "📊 Visualize Embeddings",
            help=(
                "• Encodes CV and top job embeddings  \n"
                "• UMAP to 3D  \n"
                "• Plotly scatter: semantic proximity"
            ),
            on_click=lambda: open_only("visualize")
        )

    return scan_now, rank_now, evaluate_now, visualize_now

# =====================================================
# 7️⃣ Persistent Sections (unchanged UI)
# =====================================================
def show_persistent_ranked_jobs():
    ranked_jobs = st.session_state.get("ranked_jobs", [])
    cover_letters = st.session_state.get("cover_letters", [])
    run_time = st.session_state.get("rank_log_time", "")
    is_open = st.session_state.get("open_section") == "rank"

    with st.expander(f"📋 Top Matched Jobs (last updated: {run_time})", expanded=is_open):
        st.markdown("## Top Matched Jobs")
        if not ranked_jobs:
            st.info("ℹ️ No ranked jobs available yet. Press **Rank Matches** to generate them.")
            return

        for i, (cos, job, hybrid, bm25, ce) in enumerate(ranked_jobs, 1):
            title = job.get("title", "N/A")
            company = job.get("company", "N/A")
            location = job.get("location", "N/A")
            link = job.get("link", "#")

            st.markdown(f"**{i}. {title} — {company}**")
            st.caption(location)
            st.markdown(
                f"[🔗 View Posting]({link})&nbsp;&nbsp;"
                f"<span class='score-badge bm25-badge'>BM25: {bm25:.3f}</span>"
                f"&nbsp;&nbsp;"
                f"<span class='score-badge cos-badge'>Cosine: {cos:.3f}</span>"
                f"&nbsp;&nbsp;"
                f"<span class='score-badge cross-badge'>Cross-Enc: {ce:.3f}</span>"
                f"&nbsp;&nbsp;"
                f"<span class='score-badge hyb-badge'>Hybrid: {hybrid:.3f}</span>",
                unsafe_allow_html=True,
            )

            if cover_letters:
                match = next(
                    (cl for cl in cover_letters if cl["title"] == title and cl["company"] == company),
                    None,
                )
                if match:
                    with st.expander(f"📄 View Cover Letter — {title}"):
                        st.write(match["cover_letter"])
            st.divider()

def show_persistent_evaluation():
    metrics = st.session_state.get("last_metrics")
    run_time = st.session_state.get("eval_log_time", "")
    is_open = st.session_state.get("open_section") == "evaluate"

    with st.expander(f"📊 Evaluation Metrics (last updated: {run_time})", expanded=is_open):
        st.markdown("### 📊 Evaluation Metrics")
        if not metrics:
            st.info("ℹ️ No evaluation metrics available yet. Run the evaluation first.")
            return
        st.json(metrics)
        st.markdown("""
        **Interpretation Guide**
        - **NDCG@5 / NDCG@10:** Ranking precision (1 = perfect, 0 = random)  
        - **Cosine Mean / Std:** Embedding similarity consistency  
        - **Coverage:** Share of jobs semantically related to your CV  
        """)

def show_persistent_visualization():
    fig = st.session_state.get("viz_figure")
    run_time = st.session_state.get("viz_log_time", "")
    is_open = st.session_state.get("open_section") == "visualize"

    with st.expander(f"🌐 Embedding Visualization (last updated: {run_time})", expanded=is_open):
        st.markdown("### 🌐 Embedding Plot")
        if fig is None:
            st.info("ℹ️ No visualization available yet. Generate it by pressing **📊 Visualize Embeddings**.")
            return
        st.markdown("""
        - **UMAP** reduces **high-dimensional embeddings** → **3D space**.  
        - 🟠 CV vector, 🔵 Job vectors.  
        - Color intensity reflects **semantic closeness** (brighter = closer).  
        """)
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 8️⃣ Main
# =====================================================
def main():
    setup_page()
    show_technical_overview()

    (
        job_titles, locations, days_filter, num_jobs, top_n,
        excluded_titles, alpha, beta, bm25_top_k,
        scrape_always, send_email_opt, recipient_email, generate_cover_opt
    ) = get_sidebar_inputs()

    cv_path = upload_cv()

    # Load models once
    embed_model, embed_name = load_embedding_model()
    cross_encoder_model, ce_name = load_cross_encoder_model()
    llm, llm_name = load_llm()

    # Buttons
    scan_now, rank_now, evaluate_now, visualize_now = render_action_buttons()

    # --- Build initial state for LangGraph ---
    # --- Build initial state for LangGraph ---
    base_state = {
        "cv_path": cv_path,
        "job_titles": job_titles,
        "force_refresh": scan_now,  # ✅ corrected variable name
        "locations": locations,
        "days_filter": days_filter,
        "num_jobs": num_jobs,
        "top_n": top_n,
        "excluded_titles": excluded_titles,
        "alpha": float(alpha),
        "beta": float(beta),
        "bm25_top_k": int(bm25_top_k),
        "scrape_always": bool(scrape_always),
        "send_email_opt": bool(send_email_opt),
        "recipient_email": recipient_email or "",
        "embed_model": embed_model,
        "cross_encoder_model": cross_encoder_model,
        "llm": llm,
        "embed_model_name": embed_name,
        "cross_encoder_name": ce_name,
        "llm_model_name": llm_name
        }
    # --- Segment: Scan & Store ---
    if scan_now:
        with st.expander("🔍 Job Scan & Store", expanded=True):
            st.info("🧹 Clearing / scraping / embedding / storing…")
            try:
                final_state = run_segment_scan_store(graph, base_state)
                stored = final_state.get("stored_jobs", [])
                st.write(f"🧩 Debug: state keys = {list(final_state.keys())}")
                st.write(f"🧩 Debug: type of stored_jobs = {type(stored)} len = {len(stored)}")
                st.success(f"✅ Scan & Store complete. Stored jobs available: {len(stored)}")
            except Exception as e:
                st.error(f"❌ Scan & Store failed: {e}")

    # --- Retrieve current stored jobs info (optional log) ---
    # (Graph retrieval happens inside scan-store; for rank-only runs, node_rank will re-use existing store.)

    # --- Segment: Rank Matches ---
    if rank_now:
        st.markdown("## Hybrid Scoring Formula")
        st.markdown("> **Hybrid = α × BM25 + (1–α) × [ (1–β) × Cosine + β × Cross-Encoder ]**")
        try:
            st.info("🏗️ Computing hybrid ranking…")
            final_state = run_segment_rank(graph, base_state)
            ranked = final_state.get("top_jobs", [])
            if not ranked:
                st.warning("⚠️ No ranked jobs found. Try scanning first or adjust filters.")
            else:
                # Optionally generate cover letters via graph
                if generate_cover_opt:
                    st.markdown("## ✉️ Generating Cover Letters (via graph nodes)")
                    final_state = run_segment_cover_letters(graph, final_state)
                    st.session_state["cover_letters"] = final_state.get("cover_letters", [])
                else:
                    st.session_state.pop("cover_letters", None)

                st.session_state["ranked_jobs"] = ranked
                st.session_state["rank_log_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success("🏆 Semantic reranking complete.")
        except Exception as e:
            st.error(f"❌ Ranking failed: {e}")

    # --- Segment: Evaluate ---
    if evaluate_now:
        st.markdown("## 📊 Evaluation Metrics")
        st.markdown("""
        These indicators help interpret **how well your Hybrid Ranker performs**:
        - **NDCG@5 / NDCG@10**  
        - **Cosine Mean / Std**  
        - **Coverage** (>0.5)
        """)
        try:
            st.info("🧮 Evaluating current ranking results ...")
            # carry ranked jobs in state if present
            base_state_eval = {**base_state, "top_jobs": st.session_state.get("ranked_jobs", [])}
            final_state = run_segment_evaluate(graph, base_state_eval)
            metrics = final_state.get("metrics", {})
            if metrics:
                st.session_state["last_metrics"] = metrics
                st.session_state["eval_log_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            else:
                st.warning("⚠️ No evaluation metrics produced.")
        except Exception as e:
            st.error(f"❌ Evaluation failed: {e}")

    # --- Segment: Visualization (computed in-app using helper) ---
    if visualize_now:
        ranked_jobs = st.session_state.get("ranked_jobs", [])
        if not ranked_jobs:
            st.warning("⚠️ Please run ranking first.")
        else:
            try:
                cv_text = None  # Fetch from graph by re-running load_cv minimally:
                # Use a small run to ensure cv_text is available (no full pipeline)
                tmp_state = run_segment_rank(graph, base_state)
                cv_text = tmp_state.get("cv_text", "")
                fig = visualize_embeddings_plotly(cv_text, ranked_jobs, embed_model)
                st.session_state["viz_figure"] = fig
                st.session_state["viz_log_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["visualization_rendered"] = True
            except Exception as e:
                st.error(f"❌ Visualization failed: {e}")

    # --- Optional Email (via graph) ---
    if send_email_opt:
        ranked_jobs = st.session_state.get("ranked_jobs", [])
        if ranked_jobs:
            try:
                # Ensure email_body is present by running format_email and send nodes
                state_for_email = {**base_state, "top_jobs": ranked_jobs}
                state_for_email = run_segment_cover_letters(graph, state_for_email)  # ensures format_email
                state_for_email = run_segment_email(graph, state_for_email)
                st.success("✅ Email sent (or previewed if credentials missing).")
            except Exception as e:
                st.error(f"❌ Email flow failed: {e}")
        else:
            st.warning("⚠️ No ranked jobs to email. Please run ranking first.")

    # --- Persistent sections (fixed order) ---
    show_persistent_ranked_jobs()
    show_persistent_evaluation()
    show_persistent_visualization()

    st.markdown("---")
    st.caption("Built by Amir Feizi")

# =====================================================
# 🏁 Run
# =====================================================
if __name__ == "__main__":
    main()
