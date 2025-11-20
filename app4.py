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
import logging
from openai import OpenAI
import re

# --- Disable telemetry noise ---
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Mute residual Chroma logs ---
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.CRITICAL)


# --- Local imports ---
from style import apply_custom_style
from job_ranker4 import (
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
from job_ranker4 import graph


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

        days_filter = st.slider("Days posted within", 1, 30, 7)
        num_jobs = st.slider("Number of jobs to fetch", 10, 300, 200)
        top_n = st.slider("Top N job matches", 5, 30, TOP_N)    

        excluded_titles = [x.strip() for x in st.text_area(
            "Exclude titles containing (comma-separated):",
            ", ".join(EXCLUDED_TITLES),
        ).split(",") if x.strip()]

        st.markdown("---")
        send_email_opt = st.checkbox("📧 Send results via email", value=False)
        recipient_email = st.text_input("Recipient email")
        generate_cover_opt = st.checkbox("✉️ Generate Cover Letters", value=False)

        return job_titles, locations, days_filter, num_jobs, top_n, excluded_titles, send_email_opt, recipient_email, generate_cover_opt


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
    # --- Initialize global open section state ---
    if "open_section" not in st.session_state:
        st.session_state["open_section"] = None

    def open_only(section_name):
        """Close all other expanders and keep only the active one open."""
        st.session_state["open_section"] = section_name

    # --- Button layout ---
    colA, colB, colC, colD = st.columns(4)

    with colA:
        scrape_now = st.button(
            "🔍 Job Scan & Store",
            help=(
                "• Clears old job data from ChromaDB  \n"
                "• Scrapes new job postings (titles, Location, full job descriptions, apply filters)  \n"
                "• Embeds each job using your selected model(gte-large)  \n"
                "• Stores them in ChromaDB for semantic retrieval"
            ),
            on_click=lambda: open_only("scan")
        )

    with colB:
        rank_now = st.button(
            "🏆 Rank Matches",
            help=(
                "• Embeds your uploaded CV text  \n"
                "• Retrieves semantically similar jobs from ChromaDB  \n"
                "• Computes hybrid scores (BM25 + cosine similarity)  \n"
                "• Displays top-ranked job matches"
            ),
            on_click=lambda: open_only("rank")
        )

    with colC:
        evaluate_now = st.button(
            "🧩 Evaluate Results",
            help=(
                "• Measures ranking quality using NDCG@5 / NDCG@10  \n"
                "• Computes cosine mean, std, and coverage  \n"
                "• Shows retrieval effectiveness of your hybrid scoring"
            ),
            on_click=lambda: open_only("evaluate")
        )

    with colD:
        visualize_now = st.button(
            "📊 Visualize Embeddings",
            help=(
                "• Encodes CV and top job embeddings  \n"
                "• Projects them into 3D space using UMAP  \n"
                "• Visualizes semantic distances with color and proximity"
            ),
            on_click=lambda: open_only("visualize")
        )

    return scrape_now, rank_now, evaluate_now, visualize_now

# =====================================================
# 7️⃣ Fetch and Store Jobs
# =====================================================


def fetch_and_store_jobs(
    scrape_now,
    job_titles,
    locations,
    num_jobs,
    days_filter,
    excluded_titles,
    embed_model,
):
    """Fetch, clean, and store jobs — only stays open while active button pressed."""
    is_open = st.session_state.get("open_section") == "scan"

    with st.expander("🔍 Job Scan & Store", expanded=is_open):
        # ✅ Initialize persistent job logs if not exist
        if "job_logs" not in st.session_state:
            st.session_state["job_logs"] = []

        jobs = []
        if scrape_now:
            # 🧹 Clear history when button pressed
            st.session_state["job_logs"].clear()

            run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_log = {"time": run_time, "messages": []}

            clear_collection_if_any()
            msg1 = "🧹 Old jobs cleared before fetching new ones…"
            current_log["messages"].append(("info", msg1))

            with st.spinner(" 🧹 Clearing / scraping / embedding / storing…(titles, location, descriptions, filters ..)"):
                jobs = scrape_linkedin_jobs(job_titles, locations, num_jobs, days_filter, excluded_titles)
                msg2 = f"✅ Extracted {len(jobs)} jobs successfully (with full descriptions)."
                current_log["messages"].append(("success", msg2))

                msg3 = "💾 Storing jobs in ChromaDB using the selected embedding model (gte-large - 1024D - near GPT performance)"
                current_log["messages"].append(("info", msg3))

                store_jobs_in_chromadb(jobs, embed_model)

                msg4 = f"🟩 {len(jobs)} jobs fetched and stored."
                current_log["messages"].append(("success", msg4))

            # ✅ Store only the new run
            st.session_state["job_logs"].append(current_log)

        # ========================================
        # 📜 Collapsible Job Fetch History
        # ========================================
        job_logs = st.session_state.get("job_logs", [])
        if job_logs:
            st.markdown("### 📂 Job Fetch History")
            for run in job_logs:
                with st.expander(f"🕒 Job Scan on {run['time']}", expanded=False):
                    for level, log in run["messages"]:
                        if level == "success":
                            st.success(log)
                        elif level == "info":
                            st.info(log)
                        else:
                            st.write(log)

        return jobs



# =====================================================
# 8️⃣ Ranking & Cover Letter
# =====================================================


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
            temperature=0.3, max_tokens=400,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error generating cover letter: {e}]"


def rank_and_display_jobs(rank_now, cv_text, stored_jobs, embed_model, top_n, generate_cover_opt):
    """
    Compute hybrid ranking (BM25 + embedding),
    store results in session, and optionally generate & display
    cover letters one-by-one.
    """
    if not rank_now:
        return st.session_state.get("ranked_jobs", [])

    st.info("Computing hybrid ranking…")
    with st.spinner("Retrieving and reranking jobs…"):
        ranked = hybrid_rank(cv_text, stored_jobs, embed_model, top_n=top_n)

    if not ranked:
        st.warning("⚠️ No ranked jobs found.")
        return []

    # --- Normalize structure ---
    parsed = []
    for item in ranked:
        if isinstance(item, tuple) and len(item) == 4:
            cos, job, hybrid, bm25 = item
            parsed.append((cos, job, hybrid, bm25))
        elif isinstance(item, tuple):
            vals = list(item) + [0.0] * (4 - len(item))
            parsed.append(tuple(vals))
        elif isinstance(item, dict):
            parsed.append((item.get("score", 0.0), item, 0.0, 0.0))

    # --- Save ranked jobs ---
    st.session_state["ranked_jobs"] = parsed
    st.session_state["rank_log_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # =====================================================
    # ✉️ Cover Letter Mode
    # =====================================================
    if generate_cover_opt:
        # 🔥 Prevent plain job list from rendering
        st.session_state["suppress_plain_results"] = True

        st.success("🏆 Semantic reranking complete.")
        st.markdown("### ✉️ Generating Cover Letters")

        cover_letters = []
        progress = st.progress(0)
        total = min(top_n, len(parsed))

        for i, (cos, job, hybrid, bm25) in enumerate(parsed[:top_n], 1):
            title = job.get("title", "N/A")
            company = job.get("company", "N/A")
            location = job.get("location", "N/A")
            link = job.get("link", "#")

            # --- Job info with scores ---
            st.markdown(f"**{i}. {title} — {company}**")
            st.caption(location)
            st.markdown(
                f"[🔗 View Posting]({link})&nbsp;&nbsp;"
                f"<span class='score-badge bm25-badge'>BM25: {bm25:.3f}</span>"
                f"&nbsp;&nbsp;"
                f"<span class='score-badge cos-badge'>Cosine: {cos:.3f}</span>"
                f"&nbsp;&nbsp;"
                f"<span class='score-badge hyb-badge'>Hybrid: {hybrid:.3f}</span>",
                unsafe_allow_html=True,
            )

            # --- Generate & show cover letter immediately ---
            with st.spinner(f"Generating personalized cover letter for {company}..."):
                letter = generate_cover_letter(cv_text, job)

            with st.expander(f"📄 View Cover Letter — {title}"):
                st.write(letter)

            cover_letters.append({
                "title": title,
                "company": company,
                "cover_letter": letter
            })

            progress.progress(i / total)
            st.divider()

        st.session_state["cover_letters"] = cover_letters
        st.success(f"✅ Generated {len(cover_letters)} cover letters.")

    else:
        # 🔥 Allow plain results to show
        st.session_state["suppress_plain_results"] = False

        if "cover_letters" in st.session_state:
            del st.session_state["cover_letters"]

        st.success("🏆 Semantic reranking complete.")
        st.info("✉️ Cover letter generation skipped (option unchecked).")

    return parsed


# def show_persistent_ranked_jobs():
#     """Display Top Matched Jobs persistently — auto-collapses when another section is active."""
#     ranked_jobs = st.session_state.get("ranked_jobs", [])
#     cover_letters = st.session_state.get("cover_letters", [])
#     run_time = st.session_state.get("rank_log_time", "")
#     is_open = st.session_state.get("open_section") == "rank"

#     with st.expander(f"📋 Top Matched Jobs (last updated: {run_time})", expanded=is_open):
#         st.markdown("## Top Matched Jobs")

#         if not ranked_jobs:
#             st.info("ℹ️ No ranked jobs available yet. Press **Rank Matches** to generate them.")
#             return

#         for i, (cos, job, hybrid, bm25) in enumerate(ranked_jobs, 1):
#             title = job.get("title", "N/A")
#             company = job.get("company", "N/A")
#             location = job.get("location", "N/A")
#             link = job.get("link", "#")

#             # --- Job summary with metrics ---
#             st.markdown(f"**{i}. {title} — {company}**")
#             st.caption(location)
#             st.markdown(
#                 f"[🔗 View Posting]({link})&nbsp;&nbsp;"
#                 f"<span class='score-badge bm25-badge'>BM25: {bm25:.3f}</span>"
#                 f"&nbsp;&nbsp;"
#                 f"<span class='score-badge cos-badge'>Cosine: {cos:.3f}</span>"
#                 f"&nbsp;&nbsp;"
#                 f"<span class='score-badge hyb-badge'>Hybrid: {hybrid:.3f}</span>",
#                 unsafe_allow_html=True,
#             )

#             # --- Optional Cover Letter ---
#             if cover_letters:
#                 match = next(
#                     (cl for cl in cover_letters if cl["title"] == title and cl["company"] == company),
#                     None,
#                 )
#                 if match:
#                     with st.expander(f"📄 View Cover Letter — {title}"):
#                         st.write(match["cover_letter"])

#             st.divider()

def show_persistent_ranked_jobs():
    """
    Display Top Matched Jobs persistently — but only when cover-letter mode is OFF.
    Auto-collapses when another section is active.
    """

    # 🚫 Do NOT show the plain list if cover letters were generated
    if st.session_state.get("suppress_plain_results", False):
        return

    ranked_jobs = st.session_state.get("ranked_jobs", [])
    cover_letters = st.session_state.get("cover_letters", [])
    run_time = st.session_state.get("rank_log_time", "")

    # Control auto-expansion
    is_open = st.session_state.get("open_section") == "rank"

    with st.expander(f"📋 Top Matched Jobs (last updated: {run_time})", expanded=is_open):
        st.markdown("## Top Matched Jobs")

        # Nothing to show yet
        if not ranked_jobs:
            st.info("ℹ️ No ranked jobs available yet. Press **Rank Matches** to generate them.")
            return

        # Loop through all ranked jobs
        for i, (cos, job, hybrid, bm25) in enumerate(ranked_jobs, 1):
            title = job.get("title", "N/A")
            company = job.get("company", "N/A")
            location = job.get("location", "N/A")
            link = job.get("link", "#")

            # --- Job summary with metrics ---
            st.markdown(f"**{i}. {title} — {company}**")
            st.caption(location)
            st.markdown(
                f"[🔗 View Posting]({link})&nbsp;&nbsp;"
                f"<span class='score-badge bm25-badge'>BM25: {bm25:.3f}</span>"
                f"&nbsp;&nbsp;"
                f"<span class='score-badge cos-badge'>Cosine: {cos:.3f}</span>"
                f"&nbsp;&nbsp;"
                f"<span class='score-badge hyb-badge'>Hybrid: {hybrid:.3f}</span>",
                unsafe_allow_html=True,
            )

            # --- Show cover letters only if they exist in session ---
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
    """Display evaluation metrics persistently and only keep this expander open when active."""
    metrics = st.session_state.get("last_metrics")
    run_time = st.session_state.get("eval_log_time", "")
    is_open = st.session_state.get("open_section") == "evaluate"

    with st.expander(f"📊 Evaluation Metrics (last updated: {run_time})", expanded=is_open):
        st.markdown("### 📊 Evaluation Metrics")

        if not metrics:
            st.info("ℹ️ No evaluation metrics available yet. Run the evaluation first.")
            return

        # --- Display metrics nicely ---
        st.json(metrics)

        # Optional: summary interpretation
        st.markdown("""
        **Interpretation Guide**
        - **NDCG@5 / NDCG@10:** Ranking precision (1 = perfect, 0 = random)  
        - **Cosine Mean / Std:** Embedding similarity consistency  
        - **Coverage:** Share of jobs semantically related to your CV  
        """)



def show_persistent_visualization():
    """Display embedding visualization persistently and auto-collapse when another section is active."""
    fig = st.session_state.get("viz_figure")
    run_time = st.session_state.get("viz_log_time", "")
    is_open = st.session_state.get("open_section") == "visualize"

    with st.expander(f"🌐 Embedding Visualization (last updated: {run_time})", expanded=is_open):
        st.markdown("### 🌐 Embedding Plot")

        if fig is None:
            st.info("ℹ️ No visualization available yet. Generate it by pressing **📊 Visualize Embeddings**.")
            return

        st.markdown("""
        - **UMAP (Uniform Manifold Approximation and Projection)** reduces **1024-dimensional gte-large embeddings** → **3D space**.  
        - 🟠 CV vector, 🔵 Job vectors.  
        - **Color intensity** indicates semantic closeness (🔴 = high similarity).  
        - UMAP preserves **local** and **global** relationships better than PCA, giving clearer cluster separation.
        """)
        st.plotly_chart(fig, use_container_width=True)

def main():
    # ========================================
    # 🏗️ Setup & Sidebar Configuration
    # ========================================
    setup_page()
    (
        job_titles,
        locations,
        days_filter,
        num_jobs,
        top_n,
        excluded_titles,
        send_email_opt,
        recipient_email,
        generate_cover_opt,
    ) = get_sidebar_inputs()

    # ========================================
    # 📄 CV Upload & Models
    # ========================================
    cv_text = upload_cv()
    embed_model = load_embedding_model()
    

    # ========================================
    # 🎛️ Action Buttons
    # ========================================
    scrape_now, rank_now, evaluate_now, visualize_now = render_action_buttons()

    # ========================================
    # 🔍 Job Scanning & Storage
    # ========================================
    fetch_and_store_jobs(
        scrape_now,
        job_titles,
        locations,
        num_jobs,
        days_filter,
        excluded_titles,
        embed_model,
    )

    # ========================================
    # 📦 Retrieve Stored Jobs
    # ========================================
    stored_jobs = retrieve_jobs_from_chromadb()

    # ========================================
    # 🧠 Ranking Logic
    # ========================================
    ranked = st.session_state.get("ranked_jobs", [])

    if ranked:
        first_job = ranked[0][1]  # the job dict
        title = first_job.get("title", "N/A")
        company = first_job.get("company", "N/A")
        description = first_job.get("text") or first_job.get("description", "")

        print("\n===== 🏆 Top Ranked Job =====")
        print(f"Title: {title}")
        print(f"Company: {company}")
        print(f"Description:\n{description}")  # limit to first 1000 chars for readability
        print("==============================\n")
    else:
        print("⚠️ No ranked jobs found.")


    if rank_now:
        st.markdown("### Hybrid Scoring ")
        st.markdown("""
        The system blends **lexical** and **semantic** relevance using a two-level weighted formula:
        > **Hybrid = α × BM25 + (1–α) × Cosine**
        """)

        if not cv_text.strip():
            st.error("⚠️ Please upload your CV before running ranking.")
        elif not stored_jobs:
            st.warning("⚠️ No jobs found. Please scrape or load jobs first.")
        else:
            ranked = rank_and_display_jobs(
                rank_now, cv_text, stored_jobs, embed_model, top_n, generate_cover_opt
            )
            st.session_state["ranked_jobs"] = ranked

    
    # ========================================
    # 🧮 Evaluation (Fast Local Node Only)
    # ========================================
    if evaluate_now:
        if not cv_text.strip():
            st.error("⚠️ Please upload your CV before running evaluation.")
        elif "uploaded_cv_path" not in st.session_state:
            st.warning("⚠️ No CV path found. Please re-upload your CV.")
        else:
            

            from job_ranker4 import node_evaluate
            initial_state = {
                "cv_path": st.session_state["uploaded_cv_path"],
                "cv_text": cv_text,
                "stored_jobs": stored_jobs,
                "top_jobs": st.session_state.get("ranked_jobs", []),
                "metrics": {},
                "send_email_opt": False,
            }

            with st.spinner("Evaluating ranking performance..."):
                try:
                    final_state = node_evaluate(initial_state)
                    metrics = final_state.get("metrics", {})
                    if metrics:
                        st.session_state["last_metrics"] = metrics
                        st.session_state["eval_log_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        st.warning("⚠️ No evaluation metrics found.")
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {e}")




    # ========================================
    # 🌐 Visualization
    # ========================================
    if visualize_now:
        ranked_jobs = st.session_state.get("ranked_jobs", [])
        if not cv_text.strip():
            st.error("⚠️ Please upload your CV first.")
        elif not ranked_jobs:
            st.warning("⚠️ Please run ranking before visualization.")
        else:
            
            try:
                fig = visualize_embeddings_plotly(cv_text, ranked_jobs, embed_model)
                st.session_state["viz_figure"] = fig
                st.session_state["viz_log_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["visualization_rendered"] = True
            except Exception as e:
                st.error(f"❌ Visualization failed: {e}")

    # ========================================
    # ✉️ Email Sending (Optional)
    # ========================================
    if send_email_opt:
        ranked_jobs = st.session_state.get("ranked_jobs", [])
        if ranked_jobs:
            lines = [f"Top Job Matches — {datetime.now():%Y-%m-%d %H:%M}"]
            for i, item in enumerate(ranked_jobs, 1):
                if isinstance(item, tuple):
                    cos, job, hybrid, *rest = item
                elif isinstance(item, dict):
                    job = item
                    hybrid = job.get("score", 0)
                else:
                    continue

                title = job.get("title", "N/A")
                company = job.get("company", "N/A")
                location = job.get("location", "N/A")
                link = job.get("link", "#")

                lines.append(f"{i}. {title} at {company} ({location}) — Hybrid {hybrid:.3f}")
                lines.append(f"   {link}")

            email_body = "\n".join(lines)
            try:
                send_email(recipient_email, "Daily AI-Powered Job Matches", email_body)
                st.success("✅ Email sent successfully.")
            except Exception as e:
                st.error(f"❌ Failed to send email: {e}")
        else:
            st.warning("⚠️ No ranked jobs to email. Please run ranking first.")

    # ========================================
    # 📋 Persistent Section Display (Unified Order)
    # ========================================
    # Ensures UI order stays fixed:
    # 🔍 Scan → 🏆 Rank → 📊 Evaluation → 🌐 Visualization
    
    show_persistent_ranked_jobs()
    show_persistent_evaluation()
    show_persistent_visualization()

    # ========================================
    # 🪶 Footer
    # ========================================
    st.markdown("---")
    st.caption("Built by Amir Feizi")


# =====================================================
# 🏁 Run
# =====================================================
if __name__ == "__main__":
    main()
