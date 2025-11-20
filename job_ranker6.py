# =====================================================
# 🧭 AI-Powered Job Search Ranker — Backend (No LangGraph)
# =====================================================
import os
import random
import time
from datetime import datetime
import pdfplumber
import requests
from bs4 import BeautifulSoup
import numpy as np
import streamlit as st

# ML / NLP
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
from sklearn.metrics import ndcg_score

# Email
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

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
LOCATIONS = ["Ontario"]
NUM_JOBS = 50
DAYS_FILTER = 7
TOP_N = 10
temperature = 0.7

EXCLUDED_TITLES = [
    "Data Engineer",
    "Junior",
    "Owner",
    "Manager",
    "VP",
    "AVP",
    "SVP",
    "Director",
    "Vice President",
    "Chief",
    "Analyst",
    "Intern",
    "CO-OP",
    "Officer",
    "Developer",
    "Lead", 
]

EXCLUDED_COMPANIES =[]
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL", "")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME) if EMBEDDING_MODEL_NAME else None

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



def scrape_linkedin_jobs(
    job_titles,
    locations,
    num_jobs,
    days_filter=None,
    excluded_titles=None,
    excluded_companies=None
):
    """
    Scrape LinkedIn job postings with pagination and full descriptions.
    Company + title filtering now fully normalized and bulletproof.
    """

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import unicodedata

    # ---------------- NORMALIZATION FUNCTION ----------------
    def clean(x):
        if not isinstance(x, str):
            return ""
        x = unicodedata.normalize("NFKD", x)
        x = x.encode("ascii", "ignore").decode()
        return x.lower().strip()

    excluded_titles = [clean(x) for x in (excluded_titles or [])]
    excluded_companies = [clean(x) for x in (excluded_companies or [])]

    # ---------------------------------------------------------
    base_url = "https://www.linkedin.com/jobs/search/"
    all_jobs = []

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    # Helper: fetch full job description safely
    def fetch_description(link):
        try:
            jd_resp = requests.get(link, headers=headers, timeout=5)
            if jd_resp.status_code == 200:
                jd_soup = BeautifulSoup(jd_resp.text, "html.parser")
                desc_elem = jd_soup.find("div", class_="show-more-less-html__markup")
                if desc_elem:
                    return desc_elem.get_text(" ", strip=True)
        except Exception:
            pass
        return ""

    # --------------------- MAIN LOOP ------------------------
    for location in locations:
        for job_title in job_titles:
            print(f"\n📍 Location: {location} | Title: {job_title}")
            start, fetched = 0, 0

            while fetched < num_jobs:
                query = job_title.replace(" ", "%20")
                url = f"{base_url}?keywords={query}&location={location}&start={start}"

                if days_filter:
                    url += f"&f_TPR=r{int(days_filter) * 86400}"

                try:
                    r = requests.get(url, headers=headers, timeout=20)
                    if r.status_code != 200:
                        print(f" Page {start // 25 + 1}: HTTP {r.status_code}")
                        break
                except Exception as e:
                    print(f" Request failed: {e}")
                    break

                soup = BeautifulSoup(r.text, "html.parser")
                listings = soup.find_all("div", class_="base-card")

                if not listings:
                    print(" No more listings found.")
                    break

                page_jobs = []
                for job in listings:
                    title_elem = job.find("h3", class_="base-search-card__title")
                    company_elem = job.find("h4", class_="base-search-card__subtitle")
                    location_elem = job.find("span", class_="job-search-card__location")
                    link_elem = job.find("a", class_="base-card__full-link")

                    if not all([title_elem, company_elem, location_elem, link_elem]):
                        continue

                    title = title_elem.get_text(strip=True)
                    company = company_elem.get_text(strip=True)
                    job_loc = location_elem.get_text(strip=True)
                    link = link_elem.get("href", "").split("?")[0]

                    # ---- NORMALIZED FILTERING ----
                    title_clean = clean(title)
                    company_clean = clean(company)

                    if any(bad in title_clean for bad in excluded_titles):
                        continue

                    if any(bad in company_clean for bad in excluded_companies):
                        continue

                    # Add skeleton job first
                    page_jobs.append({
                        "title": title,
                        "company": company,
                        "location": job_loc,
                        "link": link,
                    })

                # --- Fetch descriptions in parallel ---
                with ThreadPoolExecutor(max_workers=5) as ex:
                    future_to_job = {ex.submit(fetch_description, j["link"]): j for j in page_jobs}

                    for future in as_completed(future_to_job):
                        j = future_to_job[future]
                        desc = future.result()

                        text = f"{j['title']}. {j['company']}. {j['location']}. {desc}".strip()
                        if len(text) < 80:
                            continue

                        j["description"] = desc
                        j["text"] = text
                        all_jobs.append(j)
                        fetched += 1

                        if fetched >= num_jobs:
                            break

                print(f" Page {start // 25 + 1}: {len(page_jobs)} jobs processed (total {fetched})")

                start += 25
                time.sleep(random.uniform(3, 6))

                if fetched >= num_jobs:
                    break

    print(f"\n Scraped {len(all_jobs)} jobs successfully (with full descriptions).")
    return all_jobs


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


# =====================================================
# 💾 Enhanced Vector Store Saver (Full Description v2.1)
# =====================================================

def store_jobs_in_chromadb(jobs, embed_model):
    """
    Store scraped job postings in ChromaDB with high-quality embeddings.

    Includes detailed terminal debugging:
      - Show token count
      - Show truncated examples
      - Show skipped jobs + reasons
      - Show embedding stats
      - Show cosine sanity check
      - Show counters and final summary
    """
    if not jobs:
        print("⚠️ No jobs to store.")
        return

    print("\n==============================")
    print("📦 STORE JOBS → CHROMADB")
    print("==============================")

    ids, embs, metas = [], [], []
    seen_links = set()

    total = len(jobs)
    skipped_short = 0
    skipped_dup = 0
    skipped_fail = 0

    # ----------------------------------------------------------
    # Helper: Safe truncation for embedding model
    # ----------------------------------------------------------
    def truncate_text(text, max_tokens=450):
        tokens = text.split()
        if len(tokens) > max_tokens:
            print(f"🔻 Truncating from {len(tokens)} → {max_tokens} tokens")
            return " ".join(tokens[:max_tokens])
        return text

    # ----------------------------------------------------------
    # Process all jobs
    # ----------------------------------------------------------
    for idx, job in enumerate(jobs, start=1):
        print(f"\n📌 Processing job {idx}/{total}")

        link = job.get("link", "").strip()
        if not link:
            skipped_fail += 1
            print("⛔ Missing link → skipping")
            continue

        if link in seen_links:
            skipped_dup += 1
            print(f"⛔ Duplicate job: {link}")
            continue
        seen_links.add(link)

        # Clean description
        desc = job.get("description", "")
        desc = " ".join(desc.split())

        # Construct semantic text
        title = job.get("title", "")
        company = job.get("company", "")
        location = job.get("location", "")

        text = f"{title} {company} {location} {desc}".strip()
        text = truncate_text(text)

        token_count = len(text.split())
        print(f"🔤 Tokens: {token_count}")

        if token_count < 12:
            skipped_short += 1
            print("⛔ Too short (<12 tokens) → skipping")
            continue

        # Generate embedding
        try:
            emb = embed_model.encode(text, normalize_embeddings=True).tolist()
        except Exception as e:
            skipped_fail += 1
            print(f"⚠️ Embedding failed: {e}")
            continue

        # Debug: Sample text
        print(f"📝 Sample text:\n{text[:250]}...\n")
        print(f"📐 Embedding dimension: {len(emb)}")

        # Debug: embedding variance
        var = np.var(emb)
        print(f"📊 Embedding variance: {var:.6f}")

        # Store metadata
        meta = {**job, "text": text}

        ids.append(link)
        embs.append(emb)
        metas.append(meta)

        print(f"✅ Stored job {idx}/{total}")

    # ----------------------------------------------------------
    # Add to ChromaDB
    # ----------------------------------------------------------
    if ids:
        collection.add(ids=ids, embeddings=embs, metadatas=metas)
        print("\n🌟 DONE — Jobs Stored in ChromaDB 🌟")
        print(f"Total stored: {len(ids)}")
    else:
        print("⚠️ No valid jobs were stored.")

    # ----------------------------------------------------------
    # Summary report
    # ----------------------------------------------------------
    print("\n==============================")
    print("📊 STORAGE SUMMARY")
    print("==============================")
    print(f"Total scraped:         {total}")
    print(f"Stored successfully:   {len(ids)}")
    print(f"Skipped (duplicates):  {skipped_dup}")
    print(f"Skipped (too short):   {skipped_short}")
    print(f"Skipped (failures):    {skipped_fail}")
    print("==============================\n")


def retrieve_jobs_from_chromadb():
    """
    Retrieve all jobs from ChromaDB with full descriptions.
    Ensures 'text' field contains complete job content for embedding or ranking.
    """
    # ✅ Include both metadata and document fields
    res = collection.get(include=["metadatas", "documents"])
    metadatas = res.get("metadatas", [])
    documents = res.get("documents", [])

    jobs = []
    for i, meta in enumerate(metadatas):
        doc_text = ""
        if documents and i < len(documents):
            doc_text = documents[i] or ""

        # ✅ Prefer 'text' if stored in metadata, otherwise reconstruct
        full_text = (
            meta.get("text")
            or doc_text
            or f"{meta.get('title','')} {meta.get('company','')} "
               f"{meta.get('location','')} {meta.get('description','')}"
        ).strip()

        # ✅ Ensure 'text' exists and has minimum length
        if len(full_text) < 50:
            desc = meta.get("description", "")
            full_text = (
                f"{meta.get('title','')} {meta.get('company','')} "
                f"{meta.get('location','')} {desc}"
            ).strip()

        meta["text"] = full_text
        jobs.append(meta)

    print(f"📂 Retrieved {len(jobs)} jobs from ChromaDB (with full descriptions).")
    return jobs


def hybrid_rank(cv_text, jobs, embed_model, top_n=10, bm25_top_k=150, alpha=0.25):
    """
    ⚙️ Hybrid Ranking (BM25 + Cosine only)
    -------------------------------------
    Lightweight and production-ready version
    Uses lexical + semantic fusion for robust retrieval speed and quality.
    """
    if not jobs:
        print("⚠️ No jobs provided.")
        return []

    # --- Ensure text completeness ---
    for j in jobs:
        desc = j.get("description") or j.get("summary") or j.get("snippet") or ""
        j["text"] = (
            j.get("text")
            or f"{j.get('title','')} {j.get('company','')} {j.get('location','')} {desc}".strip()
        )

    # --- Stage 1: BM25 Lexical Ranking ---
    docs = [j["text"].lower() for j in jobs]
    tokenized = [d.split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    cv_tokens = cv_text.lower().split()
    bm25_scores = bm25.get_scores(cv_tokens)

    # Select Top-K candidates
    bm25_top_k = min(bm25_top_k, len(jobs))
    top_idx = np.argsort(bm25_scores)[-bm25_top_k:][::-1]
    candidates = [jobs[i] for i in top_idx]
    bm25_top = np.array([bm25_scores[i] for i in top_idx])

    # --- Stage 2: Semantic (Cosine Similarity) ---
    def _truncate(text, max_tokens=512):
        return " ".join(text.split()[:max_tokens])

    job_texts = [_truncate(c.get("text", "")) for c in candidates]
    cv_emb = embed_model.encode(cv_text, normalize_embeddings=True)
    job_embs = embed_model.encode(job_texts, normalize_embeddings=True)
    cos = np.dot(job_embs, cv_emb)
    cos_scaled = np.clip((cos + 1.0) / 2.0, 0, 1)

    # --- Stage 3: Fusion (BM25 + Cosine) ---
    denom = bm25_top.max() - bm25_top.min()
    bm25_norm = (bm25_top - bm25_top.min()) / (denom + 1e-8)

    cos_z = (cos_scaled - cos_scaled.mean()) / (cos_scaled.std() + 1e-8)
    cos_cal = 1 / (1 + np.exp(-cos_z / 1.5))  # calibrated sigmoid

    final = (alpha * bm25_norm) + ((1 - alpha) * cos_cal)

    # --- Stage 4: Sort & Return ---
    ranked = sorted(
        zip(final, cos_scaled, bm25_norm, candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    seen, out = set(), []
    for fscore, raw_cos, raw_bm25, job in ranked:
        jid = job.get("link")
        if jid in seen:
            continue
        seen.add(jid)
        out.append(
            (
                round(float(raw_cos), 3),
                job,
                round(float(fscore), 3),
                round(float(raw_bm25), 3),
            )
        )
        if len(out) >= top_n:
            break

    print("\n==== ✅ Hybrid Rank Summary ====")
    print(f"📚 Candidates reranked: {len(candidates)}")
    print(f"🔢 Cosine range: {cos_scaled.min():.3f} → {cos_scaled.max():.3f}")
    print(f"🎛️ Hybrid range: {final.min():.3f} → {final.max():.3f}")
    print("=========================\n")

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
# 📊 Evaluation Utilities (Ranking Diagnostics & Metrics)
# =====================================================
def evaluate_ranking(cv_text, ranked_jobs, embed_model):
    """
    Evaluate hybrid ranking quality using proxy relevance.
    Metrics:
      - NDCG@5 / @10
      - Cosine mean & std
      - Coverage: % jobs with cosine > 0.7
    """
    if not ranked_jobs:
        return {"ndcg@5": 0, "ndcg@10": 0, "cos_mean": 0, "cos_std": 0, "coverage": 0}

    job_texts, y_score = [], []

    # --- Parse tuples or dicts gracefully ---
    for item in ranked_jobs:
        if isinstance(item, tuple):
            if len(item) == 4:
                cos, job, hybrid, bm25 = item
            elif len(item) == 3:
                cos, job, hybrid = item
                bm25 = 0
            elif len(item) == 2:
                cos, job = item
                hybrid, bm25 = cos, 0
            else:
                continue
        elif isinstance(item, dict):
            job = item
            cos = job.get("score", 0.0)
            hybrid = cos
            bm25 = 0
        else:
            continue

        title = job.get("title", "")
        company = job.get("company", "")
        desc = job.get("description", "")
        text = f"{title} {company} {desc}".strip()
        if text:
            job_texts.append(text)
            y_score.append(float(hybrid))

    if not job_texts:
        print("⚠️ No valid job texts found for evaluation.")
        return {"ndcg@5": 0, "ndcg@10": 0, "cos_mean": 0, "cos_std": 0, "coverage": 0}

    # --- Compute cosine similarities ---
    cv_emb = embed_model.encode(cv_text, normalize_embeddings=True)
    job_embs = embed_model.encode(job_texts, normalize_embeddings=True)
    cos_scores = np.dot(job_embs, cv_emb)  # shape (n_jobs,)
    cos_norm = np.clip((cos_scores + 1.0) / 2.0, 0, 1)  # map [-1,1] → [0,1]

    # --- Proxy ground truth: top-3 jobs most similar to CV are "relevant" ---
    y_true = np.zeros_like(cos_norm)
    top_k = np.argsort(cos_norm)[-3:]
    y_true[top_k] = 1

    y_score = np.array(y_score)

    try:
        ndcg5 = float(ndcg_score([y_true], [y_score], k=5))
        ndcg10 = float(ndcg_score([y_true], [y_score], k=10))
    except Exception as e:
        print("⚠️ NDCG computation error:", e)
        ndcg5, ndcg10 = 0.0, 0.0

    cos_mean, cos_std = float(cos_norm.mean()), float(cos_norm.std())
    coverage = float((cos_norm > 0.7).mean())

    metrics = {
        "ndcg@5": round(ndcg5, 3),
        "ndcg@10": round(ndcg10, 3),
        "cos_mean": round(cos_mean, 3),
        "cos_std": round(cos_std, 3),
        "coverage": round(coverage, 3),
    }

    print(f"📊 Evaluation Metrics\n{metrics}\n")
    return metrics


# =====================================================
# 🌐 Embedding Visualization (UMAP + Plotly)
# =====================================================
def visualize_embeddings_plotly(cv_text, ranked, embed_model):
    """
    Visualize semantic relationships between CV and top job matches
    using UMAP 3D reduction + Plotly scatter.
    """
    if not ranked:
        st.warning("⚠️ No ranked jobs available for visualization.")
        return

    import umap
    import pandas as pd
    import plotly.express as px

    # --- Extract ranked jobs and cosine scores ---
    sim_scores = np.array([item[0] for item in ranked])
    jobs = [item[1] for item in ranked]
    job_texts = [f"{job['title']} {job['company']}" for job in jobs]

    # --- Compute embeddings ---
    cv_emb = embed_model.encode(cv_text, normalize_embeddings=True)
    job_embs = [embed_model.encode(jt, normalize_embeddings=True) for jt in job_texts]

    # --- Normalize similarity for color scale ---
    normalized_sims = (sim_scores - np.min(sim_scores)) / (
        np.max(sim_scores) - np.min(sim_scores) + 1e-8
    )

    # --- Auto-cap n_neighbors to dataset size ---
    n_neighbors = max(2, min(10, len(jobs) - 1))

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        n_components=3,
        random_state=42,
    )

    X = np.vstack([cv_emb] + job_embs)
    coords = reducer.fit_transform(X)

    df = pd.DataFrame(coords, columns=["x", "y", "z"])
    df["label"] = ["CV"] + [f"Job {i+1}" for i in range(len(jobs))]
    df["similarity"] = np.insert(sim_scores, 0, 1.0)
    df["color_value"] = np.insert(normalized_sims, 0, 1.0)

    df_cv = df.iloc[[0]]
    df_jobs = df.iloc[1:]

    # --- 3D Scatter ---
    fig = px.scatter_3d(
        df_jobs,
        x="x",
        y="y",
        z="z",
        color="color_value",
        color_continuous_scale="RdBu_r",
        text="label",
        hover_data={"similarity": True},
        opacity=0.9,
        title="3D Job Embedding Space (Semantic Similarity Color)",
    )

    # --- Highlight CV point ---
    fig.add_scatter3d(
        x=df_cv["x"],
        y=df_cv["y"],
        z=df_cv["z"],
        mode="markers+text",
        marker=dict(size=14, color="#FFA726", line=dict(width=2, color="white")),
        text=["CV"],
        textposition="top center",
        name="CV",
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="UMAP-3",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False,
        coloraxis_colorbar=dict(
            title="Cosine Similarity",
            tickfont=dict(size=10),
        ),
        scene_camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
    )

    return fig
