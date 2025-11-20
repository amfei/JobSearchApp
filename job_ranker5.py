# =====================================================
# 🧭 AI-Powered Job Search Ranker — LangGraph Orchestration (v5, modular)
# =====================================================
import os
import random
import time
from datetime import datetime, timedelta
from typing import TypedDict, List, Dict, Any, Optional

import numpy as np
import pdfplumber
import requests
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from sklearn.metrics import ndcg_score
import chromadb

# LangGraph
from langgraph.graph import StateGraph, END

# LLM (optional, for cover letters)
from langchain_openai import ChatOpenAI

# Email
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv

# -----------------------------------------------------
# ⚙️ Configuration & Globals
# -----------------------------------------------------
load_dotenv()

os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR = os.getcwd()
CHROMA_DIR = os.path.join(BASE_DIR, "chromadb_store")

DEFAULTS = {
    "JOB_TITLES": ["Data Scientist"],
    "LOCATIONS": ["Ontario"],
    "NUM_JOBS": 50,
    "DAYS_FILTER": 7,
    "TOP_N": 10,
    "EXCLUDED_TITLES": [
        "Data Engineer", "Junior", "Owner", "Manager",
        "VP", "AVP", "SVP", "Director", "Vice President",
        "Chief", "Analyst", "Intern", "CO-OP", "Officer"
    ],
    "ALPHA": 0.25,        # BM25 weight
    "BETA": 0.60,         # CE vs Cosine balance
    "BM25_TOP_K": 150,
    "TEMPERATURE": 0.7,
    "SCRAPE_ALWAYS": True,
    "SEND_EMAIL": False,
}

SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")

# Models are loaded lazily by the app and passed in state
# but we keep names here for CLI or batch use if needed:
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
CROSS_ENCODER_MODEL_NAME = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

# -----------------------------------------------------
# 💾 ChromaDB setup
# -----------------------------------------------------
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection("job_descriptions")

# -----------------------------------------------------
# 🧠 State Definition
# -----------------------------------------------------
class JobState(TypedDict, total=False):
    # Inputs / Config
    cv_path: str
    cv_text: str
    job_titles: List[str]
    locations: List[str]
    days_filter: int
    num_jobs: int
    top_n: int
    excluded_titles: List[str]
    alpha: float
    beta: float
    bm25_top_k: int
    scrape_always: bool
    send_email_opt: bool
    recipient_email: str

    # Models (injected by app)
    embed_model_name: str
    cross_encoder_name: str
    llm_model_name: str
    embed_model: Any
    cross_encoder_model: Any
    llm: Any

    # Working data
    jobs: List[Dict[str, Any]]
    stored_jobs: List[Dict[str, Any]]
    top_jobs: List[Any]        # List[tuple] (cos, job, hybrid, bm25, ce)
    cover_letters: List[Dict[str, str]]
    email_body: str
    metrics: Dict[str, float]

# -----------------------------------------------------
# 🧩 Helpers
# -----------------------------------------------------
def safe_sleep(a=0.8, b=1.8):
    time.sleep(random.uniform(a, b))

def extract_text_from_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ CV not found at {pdf_path}")
    with pdfplumber.open(pdf_path) as pdf:
        text_pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n".join(p.strip() for p in text_pages if p).strip()

def clear_collection_if_any():
    try:
        collection.delete(where={"title": {"$ne": ""}})
        print("🧹 Cleared old records from ChromaDB.")
    except Exception:
        pass

def _truncate(text: str, max_tokens: int = 1024) -> str:
    return " ".join((text or "").split()[:max_tokens])

# -----------------------------------------------------
# 🌐 Scraper
# -----------------------------------------------------
def scrape_linkedin_jobs(job_titles, locations, num_jobs,
                         days_filter=None, excluded_titles=None):
    """
    Scrape LinkedIn job postings for the given titles and locations.
    (HTML structure can change — this is best-effort demo code.)
    """
    jobs = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9"
    }
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

                # Fetch detail page (best-effort)
                full_desc = ""
                try:
                    jd_resp = requests.get(link, headers=headers, timeout=15)
                    jd_resp.raise_for_status()
                    jd_soup = BeautifulSoup(jd_resp.text, "html.parser")
                    full_desc_elem = jd_soup.find("div", class_="show-more-less-html__markup")
                    if full_desc_elem:
                        full_desc = full_desc_elem.get_text(" ", strip=True)
                except Exception as e:
                    print(f"⚠️ Detail fetch failed for {title} ({company}): {e}")

                description = full_desc or desc
                full_text = f"{title} {company} {job_location} {description}".strip()

                jobs.append({
                    "title": title,
                    "company": company,
                    "location": job_location,
                    "link": link,
                    "description": description,
                    "text": full_text
                })

                time.sleep(random.uniform(3, 6))

    print(f"✅ Scraped {len(jobs)} jobs successfully (with full descriptions).")
    return jobs

# -----------------------------------------------------
# 💾 Store / Retrieve
# -----------------------------------------------------
def store_jobs_in_chromadb(jobs, embed_model):
    if not jobs:
        print("⚠️ No jobs to store.")
        return
    ids, embs, metas = [], [], []
    seen_links = set()
    for job in jobs:
        link = job.get("link", "")
        if not link or link in seen_links:
            continue
        seen_links.add(link)

        text = job.get("text") or f"{job.get('title','')} {job.get('company','')} {job.get('location','')} {job.get('description','')}"
        text = _truncate(text, 1024)
        if not text or len(text.split()) < 10:
            continue

        try:
            emb = embed_model.encode(text, normalize_embeddings=True).tolist()
        except Exception as e:
            print(f"⚠️ Embedding failed for {link}: {e}")
            continue

        meta = {**job, "text": text}
        ids.append(link)
        embs.append(emb)
        metas.append(meta)
        time.sleep(random.uniform(0.05, 0.15))

    if ids:
        collection.add(ids=ids, embeddings=embs, metadatas=metas)
        print(f"💾 Stored {len(ids)} job vectors in ChromaDB (full descriptions).")
    else:
        print("⚠️ No valid jobs were stored (check data quality).")

def retrieve_jobs_from_chromadb():
    res = collection.get(include=["metadatas", "documents"])
    metadatas = res.get("metadatas", []) or []
    documents = res.get("documents", []) or []
    jobs = []
    for i, meta in enumerate(metadatas):
        doc_text = documents[i] if i < len(documents) else ""
        full_text = meta.get("text") or doc_text or f"{meta.get('title','')} {meta.get('company','')} {meta.get('location','')} {meta.get('description','')}"
        if len(full_text) < 50:
            desc = meta.get("description", "")
            full_text = f"{meta.get('title','')} {meta.get('company','')} {meta.get('location','')} {desc}".strip()
        meta["text"] = full_text
        jobs.append(meta)
    print(f"📂 Retrieved {len(jobs)} jobs from ChromaDB (with full descriptions).")
    return jobs

# -----------------------------------------------------
# 🏆 Hybrid Rank
# -----------------------------------------------------
def hybrid_rank(cv_text, jobs, embed_model, cross_encoder_model,
                top_n=10, bm25_top_k=150, alpha=0.25, beta=0.6):
    if not jobs:
        print("⚠️ No jobs provided.")
        return []

    # Ensure text
    for j in jobs:
        desc = j.get("description") or j.get("summary") or j.get("snippet") or ""
        j["text"] = j.get("text") or f"{j.get('title','')} {j.get('company','')} {j.get('location','')} {desc}".strip()

    # Stage 1: BM25
    docs = [j["text"].lower() for j in jobs]
    tokenized = [d.split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    cv_tokens = cv_text.lower().split()
    bm25_scores = bm25.get_scores(cv_tokens)
    top_idx = np.argsort(bm25_scores)[-bm25_top_k:][::-1]
    candidates = [jobs[i] for i in top_idx]
    bm25_top = np.array([bm25_scores[i] for i in top_idx])

    # Stage 2: Cosine
    def trunc(t): return _truncate(t, 512)
    job_texts = [trunc(c.get("text", "")) for c in candidates]
    cv_emb = embed_model.encode(cv_text, normalize_embeddings=True)
    job_embs = embed_model.encode(job_texts, normalize_embeddings=True)
    cos = np.dot(job_embs, cv_emb)
    cos_scaled = np.clip((cos + 1.0) / 2.0, 0, 1)

    # Stage 3: Cross-Encoder
    pairs = [(cv_text, jt) for jt in job_texts]
    try:
        ce_scores = np.array(cross_encoder_model.predict(pairs))
        ce_probs = 1 / (1 + np.exp(-ce_scores))
        ce_norm = (ce_probs - ce_probs.min()) / (ce_probs.max() - ce_probs.min() + 1e-8)
    except Exception as e:
        print("⚠️ Cross-Encoder failed:", e)
        ce_norm = np.zeros_like(cos_scaled)

    # Fusion
    denom = bm25_top.max() - bm25_top.min()
    bm25_norm = (bm25_top - bm25_top.min()) / (denom + 1e-8)
    cos_z = (cos_scaled - cos_scaled.mean()) / (cos_scaled.std() + 1e-8)
    cos_cal = 1 / (1 + np.exp(-cos_z / 1.5))
    final = (alpha * bm25_norm) + ((1 - alpha) * ((1 - beta) * cos_cal + beta * ce_norm))

    ranked = sorted(zip(final, cos_scaled, bm25_norm, ce_norm, candidates),
                    key=lambda x: x[0], reverse=True)
    seen, out = set(), []
    for fscore, raw_cos, raw_bm25, raw_ce, job in ranked:
        jid = job.get("link")
        if jid in seen:
            continue
        seen.add(jid)
        out.append((
            round(float(raw_cos), 3),
            job,
            round(float(fscore), 3),
            round(float(raw_bm25), 3),
            round(float(raw_ce), 3)
        ))
        if len(out) >= top_n:
            break
    return out

# -----------------------------------------------------
# ✉️ Email
# -----------------------------------------------------
def send_email(recipient_email, subject, body, allow_send=False):
    if not allow_send:
        print("[DRY RUN] Email body:\n" + body)
        return True
    if not (SENDER_EMAIL and APP_PASSWORD):
        print("⚠️ Missing credentials. Preview:\n", body)
        return False

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
        return True
    except Exception as e:
        print("❌ Email failed:", e)
        return False

# -----------------------------------------------------
# 🧪 Evaluation
# -----------------------------------------------------
def evaluate_ranking(cv_text, ranked_jobs, embed_model):
    if not ranked_jobs:
        return {"ndcg@5":0, "ndcg@10":0, "cos_mean":0, "cos_std":0, "coverage":0}

    job_texts, y_score = [], []
    for item in ranked_jobs:
        if isinstance(item, tuple):
            vals = list(item) + [0.0] * (5 - len(item))
            cos, job, hybrid, bm25, ce = vals[:5]
        elif isinstance(item, dict):
            job = item
            hybrid = job.get("score", 0.0)
        else:
            continue
        text = f"{job.get('title','')} {job.get('company','')} {job.get('description','')}".strip()
        if text:
            job_texts.append(text)
            y_score.append(hybrid)

    if not job_texts:
        return {"ndcg@5":0, "ndcg@10":0, "cos_mean":0, "cos_std":0, "coverage":0}

    cv_emb = embed_model.encode(cv_text, normalize_embeddings=True)
    job_embs = embed_model.encode(job_texts, normalize_embeddings=True)
    cos_scores = np.dot(job_embs, cv_emb)
    cos_norm = np.clip((cos_scores + 1.0) / 2.0, 0, 1)

    y_true = np.zeros_like(cos_norm)
    top_k = np.argsort(cos_norm)[-3:]
    y_true[top_k] = 1
    y_score = np.array(y_score)

    try:
        ndcg5 = ndcg_score([y_true], [y_score], k=5)
        ndcg10 = ndcg_score([y_true], [y_score], k=10)
    except Exception:
        ndcg5, ndcg10 = 0, 0

    metrics = {
        "ndcg@5": round(float(ndcg5), 3),
        "ndcg@10": round(float(ndcg10), 3),
        "cos_mean": round(float(cos_norm.mean()), 3),
        "cos_std": round(float(cos_norm.std()), 3),
        "coverage": round(float((cos_norm > 0.5).mean()), 3),
    }
    print(f"📊 Evaluation Metrics\n{metrics}\n")
    return metrics

# -----------------------------------------------------
# 📝 Cover Letters
# -----------------------------------------------------
def generate_cover_letters(llm, cv_text, ranked, top_k=5):
    letters = []
    for cos, job, hybrid, bm25, ce in ranked[:top_k]:
        prompt = f"""
Write a concise, confident cover letter (≤250 words) tailored to the job.

--- CV (excerpt) ---
{_truncate(cv_text, 3500)}

--- Job ---
Title: {job.get('title')}
Company: {job.get('company')}
Location: {job.get('location')}
Description: {job.get('description','')}
        """.strip()
        try:
            resp = llm.invoke(prompt)
            text = getattr(resp, "content", str(resp)).strip()
        except Exception as e:
            text = f"⚠️ LLM generation failed: {e}"
        letters.append({**job, "cover_letter": text})
    return letters

# -----------------------------------------------------
# 📧 Email Format
# -----------------------------------------------------
def format_email_body(ranked):
    if not ranked:
        return "No job matches found."
    cos_vals = np.array([r[0] for r in ranked])
    star_cut = float(np.percentile(cos_vals, 75)) if len(cos_vals) else 1.0
    lines = [f"Top Job Matches — {datetime.now():%Y-%m-%d %H:%M}\n"]
    for i, (raw_cos, job, hybrid, bm25, ce) in enumerate(ranked, 1):
        tag = " ⭐" if raw_cos >= star_cut else ""
        lines.append(f"{i}. {job['title']} at {job['company']} ({job['location']}) — cos={raw_cos:.3f}, score={hybrid:.3f}{tag}")
        lines.append(f"   {job['link']}")
    return "\n".join(lines)

# -----------------------------------------------------
# 🧩 LangGraph Nodes
# -----------------------------------------------------
def node_load_cv(state: JobState) -> JobState:
    cv_path = state.get("cv_path", "")
    if not cv_path or not os.path.exists(cv_path):
        print("⚠️ CV path missing or not found.")
        return state
    state["cv_text"] = extract_text_from_pdf(cv_path)
    print(f"📄 CV loaded ({len(state['cv_text'])} chars).")
    return state

def node_scrape_or_skip(state: JobState) -> JobState:
    if not state.get("scrape_always", DEFAULTS["SCRAPE_ALWAYS"]):
        existing = retrieve_jobs_from_chromadb()
        if existing:
            print("📦 Using cached jobs.")
            state["stored_jobs"] = existing
            return state
    print("🔍 Scraping new jobs...")
    jobs = scrape_linkedin_jobs(
        state.get("job_titles", DEFAULTS["JOB_TITLES"]),
        state.get("locations", DEFAULTS["LOCATIONS"]),
        state.get("num_jobs", DEFAULTS["NUM_JOBS"]),
        state.get("days_filter", DEFAULTS["DAYS_FILTER"]),
        state.get("excluded_titles", DEFAULTS["EXCLUDED_TITLES"]),
    )
    state["jobs"] = jobs
    return state

def node_store(state: JobState) -> JobState:
    """
    Store embeddings into ChromaDB.
    Clears existing data only if 'force_refresh' flag is True.
    """
    jobs = state.get("jobs", [])
    embed_model = state.get("embed_model")
    if not jobs or embed_model is None:
        return state

    # only clear if explicitly requested
    if state.get("force_refresh", False):
        print("🧹 Force refresh enabled → clearing old records.")
        clear_collection_if_any()

    store_jobs_in_chromadb(jobs, embed_model)
    return state



def node_retrieve(state: JobState) -> JobState:
    state["stored_jobs"] = retrieve_jobs_from_chromadb()
    return state

def node_rank(state: JobState) -> JobState:
    cv_text = (state.get("cv_text") or "").strip()
    jobs = state.get("stored_jobs", [])
    if not cv_text or not jobs:
        state["top_jobs"] = []
        print("⚠️ Missing CV text or stored jobs.")
        return state
    ranked = hybrid_rank(
        cv_text=cv_text,
        jobs=jobs,
        embed_model=state["embed_model"],
        cross_encoder_model=state["cross_encoder_model"],
        top_n=state.get("top_n", DEFAULTS["TOP_N"]),
        bm25_top_k=state.get("bm25_top_k", DEFAULTS["BM25_TOP_K"]),
        alpha=state.get("alpha", DEFAULTS["ALPHA"]),
        beta=state.get("beta", DEFAULTS["BETA"]),
    )
    state["top_jobs"] = ranked
    return state

def node_generate_cover_letter(state: JobState) -> JobState:
    if not state.get("top_jobs"):
        state["cover_letters"] = []
        return state
    llm = state.get("llm")
    cv_text = state.get("cv_text", "")
    if llm is None:
        state["cover_letters"] = []
        print("⚠️ LLM not provided; skipping cover letters.")
        return state
    letters = generate_cover_letters(llm, cv_text, state["top_jobs"], top_k=min(5, len(state["top_jobs"])))
    state["cover_letters"] = letters
    print(f"✉️ Generated {len(letters)} cover letters.")
    return state

def node_format_email(state: JobState) -> JobState:
    state["email_body"] = format_email_body(state.get("top_jobs", []))
    return state

def node_send(state: JobState) -> JobState:
    if not state.get("send_email_opt", False):
        print("📭 Skipping email send (user opted out).")
        return state
    recipient = state.get("recipient_email", "")
    if not recipient:
        print("⚠️ No recipient provided.")
        return state
    ok = send_email(recipient, "Daily AI-Powered Job Matches", state.get("email_body", ""), allow_send=True)
    if not ok:
        print("⚠️ Email send failed.")
    return state

def node_evaluate(state: JobState) -> JobState:
    ranked_jobs = state.get("top_jobs", [])
    cv_text = state.get("cv_text", "")
    if not ranked_jobs or not cv_text:
        state["metrics"] = {}
        print("⚠️ Skipping evaluation (no ranked jobs or no CV).")
        return state
    metrics = evaluate_ranking(cv_text, ranked_jobs, state["embed_model"])
    state["metrics"] = metrics
    print("✅ Evaluation complete.")
    return state

# -----------------------------------------------------
# 🚀 Build Graph
# -----------------------------------------------------
workflow = StateGraph(JobState)
workflow.add_node("load_cv", node_load_cv)
workflow.add_node("scrape_or_skip", node_scrape_or_skip)
workflow.add_node("store", node_store)
workflow.add_node("retrieve", node_retrieve)
workflow.add_node("rank", node_rank)
workflow.add_node("evaluate", node_evaluate)
workflow.add_node("generate_cover_letter", node_generate_cover_letter)
workflow.add_node("format_email", node_format_email)
workflow.add_node("send", node_send)

workflow.set_entry_point("load_cv")
workflow.add_edge("load_cv", "scrape_or_skip")
workflow.add_edge("scrape_or_skip", "store")
workflow.add_edge("store", "retrieve")
workflow.add_edge("retrieve", "rank")
workflow.add_edge("rank", "evaluate")
workflow.add_edge("rank", "generate_cover_letter")
workflow.add_edge("generate_cover_letter", "format_email")
workflow.add_edge("format_email", "send")
workflow.add_edge("send", END)

graph = workflow.compile()

# -----------------------------------------------------
# 🧭 Modular Execution Helpers (interrupt-friendly)
# -----------------------------------------------------
def run_until(graph_exec, state: JobState, target_node: str) -> JobState:
    """
    Run the compiled graph and stop *after* target_node has executed.
    Requires LangGraph runtime that supports streaming with step events.
    """
    out_state = state
    for event in graph_exec.stream(state, interrupt_after=[target_node]):
        # event yields node completions; we update out_state on each yield
        if isinstance(event, dict) and "values" in event:
            out_state = {**out_state, **event["values"]}
    return out_state

def run_segment_scan_store(graph_exec, state: dict) -> dict:
    """
    Run load_cv → scrape_or_skip → store → retrieve in a single stream
    and return the final merged state (including stored_jobs).
    """
    out_state = state.copy()
    for event in graph_exec.stream(state, interrupt_after=["retrieve"]):
        # Each event from LangGraph has a type and may carry "values"
        if isinstance(event, dict):
            if "values" in event:
                out_state.update(event["values"])
            elif "data" in event:
                out_state.update(event["data"])
    # ✅ capture the final output snapshot
    try:
        final_output = graph_exec.get_state()
        if isinstance(final_output, dict):
            out_state.update(final_output.get("values", {}))
    except Exception:
        pass
    return out_state




def run_segment_rank(graph_exec, state: JobState) -> JobState:
    # assumes retrieve already happened
    return run_until(graph_exec, state, "rank")

def run_segment_evaluate(graph_exec, state: JobState) -> JobState:
    # will execute rank if not already done; then evaluate
    st1 = run_until(graph_exec, state, "rank")
    st2 = run_until(graph_exec, st1, "evaluate")
    return st2

def run_segment_cover_letters(graph_exec, state: JobState) -> JobState:
    st1 = run_until(graph_exec, state, "rank")
    st2 = run_until(graph_exec, st1, "generate_cover_letter")
    st3 = run_until(graph_exec, st2, "format_email")
    return st3

def run_segment_email(graph_exec, state: JobState) -> JobState:
    st1 = run_until(graph_exec, state, "format_email")
    st2 = run_until(graph_exec, st1, "send")
    return st2

# -----------------------------------------------------
# 🌐 Optional: Visualization helper (UMAP + Plotly figure builder)
# (Kept here so app can import without duplicating)
# -----------------------------------------------------
def visualize_embeddings_plotly(cv_text, ranked, embed_model):
    import umap
    import pandas as pd
    import plotly.express as px

    if not ranked:
        return None

    sim_scores = np.array([item[0] for item in ranked])
    jobs = [item[1] for item in ranked]
    job_texts = [f"{job['title']} {job['company']}" for job in jobs]

    cv_emb = embed_model.encode(cv_text, normalize_embeddings=True)
    job_embs = [embed_model.encode(jt, normalize_embeddings=True) for jt in job_texts]

    normalized_sims = (sim_scores - np.min(sim_scores)) / (np.max(sim_scores) - np.min(sim_scores) + 1e-8)
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

    fig = px.scatter_3d(
        df_jobs,
        x="x", y="y", z="z",
        color="color_value",
        color_continuous_scale="RdBu_r",
        text="label",
        hover_data={"similarity": True},
        opacity=0.9,
        title="3D Job Embedding Space (Semantic Similarity Color)"
    )

    fig.add_scatter3d(
        x=df_cv["x"], y=df_cv["y"], z=df_cv["z"],
        mode="markers+text",
        marker=dict(size=14, color="#FFA726", line=dict(width=2, color="white")),
        text=["CV"], textposition="top center",
        name="CV"
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="UMAP-3",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False,
        coloraxis_colorbar=dict(title="Cosine Similarity", tickfont=dict(size=10)),
        scene_camera=dict(eye=dict(x=1.6, y=1.6, z=0.9))
    )
    return fig
