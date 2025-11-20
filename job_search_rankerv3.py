# =====================================================
# 🧭 AI-Powered Job Search Ranker — LangGraph Orchestration (v2.1)
# =====================================================
import os
import random
import time
from datetime import datetime, timedelta
from typing import TypedDict
import pdfplumber
import requests
from bs4 import BeautifulSoup
import numpy as np

# ML / NLP
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb

# Email
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# LangGraph Orchestration
from langgraph.graph import StateGraph, END

# LangChain LLM
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv


# =====================================================
# ⚙️ Configuration
# =====================================================
load_dotenv()

os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"

BASE_DIR = os.getcwd()
CHROMA_DIR = os.path.join(BASE_DIR, "chromadb_store")

SEND_EMAIL = True
SCRAPE_ALWAYS = True

JOB_TITLES = ["Data Scientist"]
LOCATIONS = ["Quebec", "Ontario"]
NUM_JOBS = 150
DAYS_FILTER = 12
TOP_N = 20

EXCLUDED_TITLES = [
    "Data Engineer", "Junior", "Owner", "Manager",
    "VP", "AVP", "SVP", "Director", "Vice President",
    "Chief", "Analyst", "Intern", "CO-OP", "Officer"
]

SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL", "amir.h.feizi@outlook.com")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# =====================================================
# 🧩 Helper Functions
# =====================================================
def safe_sleep(a=0.8, b=1.8):
    time.sleep(random.uniform(a, b))


def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ CV not found at {pdf_path}")
    with pdfplumber.open(pdf_path) as pdf:
        text_pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n".join(p.strip() for p in text_pages if p).strip()


# =====================================================
# 🌐 Job Scraper (LinkedIn)
# =====================================================
def scrape_linkedin_jobs(job_titles, locations, num_jobs,
                         days_filter=None, excluded_titles=None):
    jobs = []
    headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}
    excluded_titles = excluded_titles or []

    for location in locations:
        print(f"📍 Location: {location}")
        for job_title in job_titles:
            print(f"🔎 Searching for: {job_title}")
            query = job_title.replace(" ", "%20")
            url = f"https://www.linkedin.com/jobs/search?keywords={query}&location={location}"

            try:
                r = requests.get(url, headers=headers, timeout=20)
                r.raise_for_status()
            except Exception as e:
                print(f"⚠️ Request failed for {job_title} in {location}: {e}")
                continue

            soup = BeautifulSoup(r.text, "html.parser")
            listings = soup.find_all("div", class_="base-card")[:num_jobs]

            for job in listings:
                title_elem = job.find("h3", class_="base-search-card__title")
                company_elem = job.find("h4", class_="base-search-card__subtitle")
                location_elem = job.find("span", class_="job-search-card__location")
                link_elem = job.find("a", class_="base-card__full-link")
                desc_elem = job.find("p", class_="base-search-card__snippet")
                date_elem = job.find("time")

                if not all([title_elem, company_elem, location_elem, link_elem]):
                    continue

                title = title_elem.get_text(strip=True)
                company = company_elem.get_text(strip=True)
                job_location = location_elem.get_text(strip=True)
                link = link_elem.get("href", "").strip()
                desc = desc_elem.get_text(" ", strip=True) if desc_elem else ""

                if any(bad.lower() in title.lower() for bad in excluded_titles):
                    continue

                keep = True
                if date_elem and date_elem.has_attr("datetime"):
                    try:
                        posted = datetime.strptime(date_elem["datetime"], "%Y-%m-%d")
                        if days_filter:
                            cutoff = datetime.now() - timedelta(days=int(days_filter))
                            keep = posted >= cutoff
                    except Exception:
                        pass
                if not keep:
                    continue

                jobs.append({
                    "title": title,
                    "company": company,
                    "location": job_location,
                    "link": link,
                    "description": desc
                })

    print(f"✅ Scraped {len(jobs)} jobs successfully.")
    return jobs


# =====================================================
# 💾 Vector Store (ChromaDB)
# =====================================================
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection("job_descriptions")


def clear_collection_if_any():
    try:
        collection.delete(where={"title": {"$ne": ""}})
        print("🧹 Cleared old records from ChromaDB.")
    except Exception:
        pass


def store_jobs_in_chromadb(jobs, embed_model):
    if not jobs:
        print("⚠️ No jobs to store.")
        return

    ids, embs, metas = [], [], []
    for job in jobs:
        text = " | ".join(filter(None, [
            job.get("title", ""),
            job.get("company", ""),
            job.get("location", ""),
            job.get("description", "")
        ]))
        emb = embed_model.encode(text, normalize_embeddings=True).tolist()
        meta = {**job, "text": text}
        ids.append(job["link"])
        embs.append(emb)
        metas.append(meta)

    collection.add(ids=ids, embeddings=embs, metadatas=metas)
    print(f"💾 Stored {len(ids)} job vectors in ChromaDB.")


def retrieve_jobs_from_chromadb():
    res = collection.get(include=["metadatas"])
    jobs = res.get("metadatas", [])
    print(f"📂 Retrieved {len(jobs)} jobs from ChromaDB.")
    return jobs


# =====================================================
# 🧠 Hybrid Ranking (BM25 + Cosine + Cross-Encoder)
# =====================================================
def hybrid_rank(cv_text, jobs, embed_model, top_n=10, bm25_top_k=100, alpha=0.25, beta=0.6):
    """
    Two-stage hybrid ranking:
      1️⃣ BM25 lexical retrieval for recall
      2️⃣ Cosine similarity + Cross-Encoder for semantic precision
      -> Returns top-N jobs with realistic cosine scores in [0,1]
    """
    if not jobs:
        print("⚠️ No jobs provided.")
        return []

    # --- Stage 1: BM25 Lexical Scoring ---
    docs = [j.get("text", f"{j['title']} {j['company']} {j['location']}").lower() for j in jobs]
    tokenized = [d.split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    cv_tokens = cv_text.lower().split()
    bm25_scores = bm25.get_scores(cv_tokens)

    # Select BM25 top candidates
    top_idx = np.argsort(bm25_scores)[-bm25_top_k:][::-1]
    candidates = [jobs[i] for i in top_idx]
    texts = [docs[i] for i in top_idx]
    bm25_top = np.array([bm25_scores[i] for i in top_idx])

    # --- Stage 2: True Cosine Similarity ---
    # Encode and L2-normalize embeddings
    cv_emb = embed_model.encode(cv_text)
    job_embs = embed_model.encode(texts)

    cv_emb = cv_emb / (np.linalg.norm(cv_emb) + 1e-8)
    job_embs = job_embs / (np.linalg.norm(job_embs, axis=1, keepdims=True) + 1e-8)

    # Compute true cosine similarities (−1 → 1)
    cos = np.dot(job_embs, cv_emb)

    # Scale to [0,1] for nicer interpretation
    cos_scaled = (cos + 1.0) / 2.0
    cos_scaled = np.clip(cos_scaled, 0, 1)

    # --- Stage 3: Score Calibration & Fusion ---
    # Normalize BM25 → [0,1]
    bmin, bmax = bm25_top.min(), bm25_top.max()
    bm25_norm = (bm25_top - bmin) / (bmax - bmin + 1e-8)

    # Calibrate cosine with z-score + sigmoid to widen distribution
    cos_z = (cos_scaled - cos_scaled.mean()) / (cos_scaled.std() + 1e-8)
    cos_cal = 1 / (1 + np.exp(-cos_z / 1.5))

    # Optional Cross-Encoder refinement
    pairs = [(cv_text, t) for t in texts]
    ce_scores = np.array(cross_encoder.predict(pairs))
    ce_norm = (ce_scores - ce_scores.min()) / (ce_scores.max() - ce_scores.min() + 1e-8)

    # Final blended score
    final = (alpha * bm25_norm) + ((1 - alpha) * ((1 - beta) * cos_cal + beta * ce_norm))

    # --- Stage 4: Sort & Deduplicate ---
    ranked = sorted(zip(final, cos_scaled, candidates), key=lambda x: x[0], reverse=True)

    seen, out = set(), []
    for fscore, raw_cos, job in ranked:
        jid = job.get("link")
        if jid in seen:
            continue
        seen.add(jid)
        out.append((round(float(raw_cos), 3), job, round(float(fscore), 3)))
        if len(out) >= top_n:
            break

    # --- Logging for diagnostics ---
    print(f"📚 BM25 candidates: {len(candidates)}")
    print(f"🔢 Cosine (true) range: {cos_scaled.min():.3f} → {cos_scaled.max():.3f}")
    print(f"🎛️ Final hybrid range: {final.min():.3f} → {final.max():.3f}")

    return out


# =====================================================
# 📧 Email Utility
# =====================================================
def send_email(recipient_email, subject, body):
    if not SEND_EMAIL:
        print("[DRY RUN] Email body:\n" + body)
        return

    if not (SENDER_EMAIL and APP_PASSWORD):
        print("⚠️ Missing credentials. Preview:\n", body)
        return

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
        print("✅ Email sent successfully.")
    except Exception as e:
        print("❌ Email failed:", e)


# =====================================================
# 🧩 LangGraph Nodes
# =====================================================
class JobState(TypedDict):
    cv_path: str
    cv_text: str
    jobs: list
    stored_jobs: list
    top_jobs: list
    email_body: str
    cover_letters: list
    metrics: dict
    send_email_opt: bool


def node_load_cv(state):
    cv_path = state.get("cv_path", "")
    if not os.path.exists(cv_path):
        print("⚠️ CV not found.")
        return state
    cv_text = extract_text_from_pdf(cv_path)
    state["cv_text"] = cv_text
    print(f"📄 CV loaded ({len(cv_text)} chars).")
    return state


def node_scrape_or_skip(state):
    existing = retrieve_jobs_from_chromadb()
    if existing and not SCRAPE_ALWAYS:
        print("📦 Using cached jobs.")
        state["stored_jobs"] = existing
        return state
    print("🔍 Scraping new jobs...")
    jobs = scrape_linkedin_jobs(JOB_TITLES, LOCATIONS, NUM_JOBS, DAYS_FILTER, EXCLUDED_TITLES)
    state["jobs"] = jobs
    return state


def node_store(state):
    jobs = state.get("jobs", [])
    if jobs:
        clear_collection_if_any()
        store_jobs_in_chromadb(jobs, embedding_model)
    return state


def node_retrieve(state):
    stored = retrieve_jobs_from_chromadb()
    state["stored_jobs"] = stored
    return state


def node_rank(state):
    cv_text = state.get("cv_text", "")
    jobs = state.get("stored_jobs", [])
    ranked = hybrid_rank(cv_text, jobs, embedding_model, TOP_N)
    state["top_jobs"] = ranked
    return state


def node_generate_cover_letter(state):
    cv_text = state.get("cv_text", "")
    top_jobs = state.get("top_jobs", [])
    if not top_jobs:
        state["cover_letters"] = []
        return state

    letters = []
    for score, job, hybrid in top_jobs[:5]:
        prompt = f"""
        You are a professional career writer.
        Write a concise, confident cover letter (≤250 words)
        for this candidate and job description.

        --- CV ---
        {cv_text[:4000]}

        --- Job ---
        Title: {job['title']}
        Company: {job['company']}
        Location: {job['location']}
        Description: {job.get('description','')}

        Tone: factual, tailored, and confident.
        """
        try:
            resp = llm.invoke(prompt)
            text = getattr(resp, "content", str(resp)).strip()
        except Exception as e:
            text = f"⚠️ LLM generation failed: {e}"

        letters.append({**job, "cover_letter": text})

    state["cover_letters"] = letters
    print(f"✉️ Generated {len(letters)} cover letters.")
    return state


def node_format_email(state):
    ranked = state.get("top_jobs", [])
    if not ranked:
        state["email_body"] = "No job matches found."
        return state
    cos_vals = np.array([r[0] for r in ranked])
    star_cut = float(np.percentile(cos_vals, 75))
    lines = [f"Top Job Matches — {datetime.now():%Y-%m-%d %H:%M}\n"]
    for i, (raw_cos, job, hybrid) in enumerate(ranked, 1):
        tag = " ⭐" if raw_cos >= star_cut else ""
        lines.append(f"{i}. {job['title']} at {job['company']} ({job['location']}) — cos={raw_cos:.3f}, score={hybrid:.3f}{tag}")
        lines.append(f"   {job['link']}")
    state["email_body"] = "\n".join(lines)
    print("📧 Email body formatted.")
    return state


def node_send(state):
    if not state.get("send_email_opt", False):
        print("📭 Skipping email send (user opted out).")
        return state
    send_email(RECIPIENT_EMAIL, "Daily AI-Powered Job Matches", state.get("email_body", ""))
    return state


# =====================================================
# 🚀 LangGraph Workflow
# =====================================================
workflow = StateGraph(JobState)
workflow.add_node("load_cv", node_load_cv)
workflow.add_node("scrape_or_skip", node_scrape_or_skip)
workflow.add_node("store", node_store)
workflow.add_node("retrieve", node_retrieve)
workflow.add_node("rank", node_rank)
workflow.add_node("generate_cover_letter", node_generate_cover_letter)
workflow.add_node("format_email", node_format_email)
workflow.add_node("send", node_send)

workflow.set_entry_point("load_cv")
workflow.add_edge("load_cv", "scrape_or_skip")
workflow.add_edge("scrape_or_skip", "store")
workflow.add_edge("store", "retrieve")
workflow.add_edge("retrieve", "rank")
workflow.add_edge("rank", "generate_cover_letter")
workflow.add_edge("generate_cover_letter", "format_email")
workflow.add_edge("format_email", "send")
workflow.add_edge("send", END)

graph = workflow.compile()


def visualize_embeddings_plotly(cv_text, ranked, embed_model):
    """
    Visualize semantic relationships between CV and top job matches
    using UMAP 3D reduction + Plotly scatter.

    🧭 Interpretation:
    - BM25 used only for recall (context)
    - Cosine similarity drives ranking and color intensity
    """
    if not ranked:
        return

    import umap
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import streamlit as st
    from rank_bm25 import BM25Okapi
    from sentence_transformers import util

    # --- Extract ranked jobs and cosine scores ---
    sim_scores = np.array([score for score, _ in ranked])
    jobs = [job for _, job in ranked]
    job_texts = [f"{job['title']} {job['company']}" for job in jobs]

    # --- Compute CV and job embeddings (normalized) ---
    cv_emb = embed_model.encode(cv_text, normalize_embeddings=True)
    job_embs = [embed_model.encode(jt, normalize_embeddings=True) for jt in job_texts]

    # --- Optional: compute BM25 for interpretability ---
    tokenized_jobs = [jt.lower().split() for jt in job_texts]
    bm25 = BM25Okapi(tokenized_jobs)
    cv_tokens = cv_text.lower().split()
    bm25_scores = bm25.get_scores(cv_tokens)

    # --- Convert similarities to cosine angles (for interpretability) ---
    cosine_angles = np.degrees(np.arccos(np.clip(sim_scores, -1.0, 1.0)))

    # --- Normalize similarity for color scale ---
    normalized_sims = (sim_scores - np.min(sim_scores)) / (np.max(sim_scores) - np.min(sim_scores) + 1e-8)

    # --- Dimensionality reduction (UMAP 3D projection) ---
    X = np.vstack([cv_emb] + job_embs)
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=3, random_state=42)
    coords = reducer.fit_transform(X)

    # --- Build DataFrame for Plotly ---
    df = pd.DataFrame(coords, columns=["x", "y", "z"])
    df["label"] = ["CV"] + [f"Job {i+1}" for i in range(len(jobs))]
    df["cosine_similarity"] = np.insert(sim_scores, 0, 1.0)
    df["cosine_angle_deg"] = np.insert(cosine_angles, 0, 0.0)
    df["bm25_score"] = np.insert(bm25_scores, 0, np.mean(bm25_scores))
    df["color_value"] = np.insert(normalized_sims, 0, 1.0)

    df_cv = df.iloc[[0]]
    df_jobs = df.iloc[1:]

    # --- Explanation section ---
    with st.expander("Embedding Visualization"):
        st.markdown("""
        ### 🧠 What This Graph Shows
        - Each point represents a **semantic embedding** (768-D → 3-D via UMAP).  
        - **Orange point:** your CV embedding.  
        - **Color intensity:** cosine similarity between job and CV (semantic closeness).  
        - Hover to see cosine angle (semantic distance) and BM25 lexical relevance.

        #### Mathematical Intuition
        - **Cosine Similarity:**  
          $$
          \\text{sim}(u,v) = \\frac{u \\cdot v}{\\|u\\| \\|v\\|}
          $$
        - **Cosine Angle:**  
          $$
          \\theta = \\arccos(\\text{sim}(u,v)) \\times \\frac{180}{\\pi}
          $$
        - **BM25 (for context):** lexical keyword matching used only for recall.
        """)

    # --- 3D Scatter Visualization ---
    fig = px.scatter_3d(
        df_jobs,
        x="x", y="y", z="z",
        color="color_value",
        color_continuous_scale="RdBu_r",
        text="label",
        hover_data={
            "cosine_similarity": True,
            "cosine_angle_deg": True,
            "bm25_score": True
        },
        opacity=0.9,
        title="3D Job Embedding Space (Semantic Similarity Color)"
    )

    # --- Highlight CV embedding ---
    fig.add_scatter3d(
        x=df_cv["x"], y=df_cv["y"], z=df_cv["z"],
        mode="markers+text",
        marker=dict(size=14, color="#FFA726", line=dict(width=2, color="white")),
        text=["CV"], textposition="top center",
        name="CV"
    )

    # --- Layout aesthetics ---
    fig.update_layout(
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="UMAP-3",
            xaxis=dict(showgrid=True, zeroline=False),
            yaxis=dict(showgrid=True, zeroline=False),
            zaxis=dict(showgrid=True, zeroline=False),
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False,
        coloraxis_colorbar=dict(title="Cosine Similarity", tickfont=dict(size=10)),
        scene_camera=dict(eye=dict(x=1.6, y=1.6, z=0.9))
    )

    st.plotly_chart(fig, use_container_width=True)




# =====================================================
# 🧭 Run Locally
# =====================================================
if __name__ == "__main__":
    uploaded_cv_path = input("📎 Enter path to your CV PDF: ").strip()
    if not uploaded_cv_path or not os.path.exists(uploaded_cv_path):
        print("⚠️ Invalid CV path.")
    else:
        initial = {
            "cv_path": uploaded_cv_path,
            "cv_text": "",
            "jobs": [],
            "stored_jobs": [],
            "top_jobs": [],
            "email_body": "",
            "cover_letters": [],
            "metrics": {},
            "send_email_opt": False
        }
        final_state = graph.invoke(initial)
        print("\n✅ Workflow completed.")
