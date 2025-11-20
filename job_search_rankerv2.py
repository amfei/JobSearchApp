import os
import random
import time
from datetime import datetime, timedelta
from typing import TypedDict
import pdfplumber
import requests
from bs4 import BeautifulSoup

# Core ML & NLP
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import chromadb

# Email utility
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# LangGraph orchestration
from langgraph.graph import StateGraph, END


# =====================================================
# ⚙️ Configuration
# =====================================================
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


# =====================================================
# 🧩 Helper Functions
# =====================================================
def safe_sleep(a=0.8, b=1.8):
    """Pause randomly to avoid rate limiting."""
    time.sleep(random.uniform(a, b))


def extract_text_from_pdf(pdf_path):
    """Extract text from uploaded PDF CV."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ CV not found at {pdf_path}")
    with pdfplumber.open(pdf_path) as pdf:
        text_pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n".join(p.strip() for p in text_pages if p).strip()


# =====================================================
# 🌐 Job Scraper (LinkedIn)
# =====================================================
def scrape_linkedin_jobs(job_titles, locations, num_jobs,
                         days_filter=None, excluded_titles=None,
                         hours_filter=None):
    """Scrape LinkedIn jobs (titles, companies, links) without Selenium.

    hours_filter (int|None): if set, overrides days_filter.
    """
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
                print(f"⚠️ Failed request for {job_title} in {location}: {e}")
                continue

            soup = BeautifulSoup(r.text, "html.parser")
            listings = soup.find_all("div", class_="base-card")[:num_jobs]

            for job in listings:
                title_elem = job.find("h3", class_="base-search-card__title")
                company_elem = job.find("h4", class_="base-search-card__subtitle")
                location_elem = job.find("span", class_="job-search-card__location")
                link_elem = job.find("a", class_="base-card__full-link")
                date_elem = job.find("time")

                if not all([title_elem, company_elem, location_elem, link_elem]):
                    continue

                title = title_elem.get_text(strip=True)
                company = company_elem.get_text(strip=True)
                job_location = location_elem.get_text(strip=True)
                link = link_elem.get("href", "").strip()

                # Skip excluded titles
                if any(bad.lower() in title.lower() for bad in excluded_titles):
                    continue

                # --- New unified time filter ---
                keep = True
                if date_elem and date_elem.has_attr("datetime"):
                    try:
                        posted = datetime.strptime(date_elem["datetime"], "%Y-%m-%d")
                        now = datetime.now()
                        if hours_filter is not None:
                            cutoff = now - timedelta(hours=int(hours_filter))
                            keep = posted >= cutoff
                        elif days_filter is not None:
                            cutoff = now - timedelta(days=int(days_filter))
                            keep = posted >= cutoff
                        else:
                            keep = True
                    except Exception:
                        pass
                if not keep:
                    continue

                jobs.append({
                    "title": title,
                    "company": company,
                    "location": job_location,
                    "link": link,
                    "description": ""
                })

    print(f"✅ Scraped {len(jobs)} jobs successfully.")
    return jobs


# =====================================================
# 💾 Vector Store (ChromaDB)
# =====================================================
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection("job_descriptions")


def clear_collection_if_any():
    """Clear all stored job embeddings in ChromaDB."""
    try:
        collection.delete(where={"title": {"$ne": ""}})
        print("🧹 Cleared old records from ChromaDB.")
    except Exception:
        pass


def store_jobs_in_chromadb(jobs, embed_model):
    """Embed and store jobs in ChromaDB."""
    if not jobs:
        print("⚠️ No jobs to store.")
        return

    ids, embs, metas = [], [], []
    for job in jobs:
        text = f"{job['title']} {job['company']} {job['location']}"
        emb = embed_model.encode(text, convert_to_tensor=False).tolist()
        meta = {
            "title": job["title"],
            "company": job["company"],
            "location": job["location"],
            "link": job["link"]
        }
        ids.append(job["link"])
        embs.append(emb)
        metas.append(meta)

    collection.add(ids=ids, embeddings=embs, metadatas=metas)
    print(f"💾 Stored {len(ids)} job vectors in ChromaDB.")


def retrieve_jobs_from_chromadb():
    """Retrieve stored job postings."""
    res = collection.get()
    jobs = res.get("metadatas", [])
    print(f"📂 Retrieved {len(jobs)} jobs from ChromaDB.")
    return jobs


# =====================================================
# 🧠 Hybrid Ranking
# =====================================================
def hybrid_rank(cv_text, jobs, embed_model, top_n):
    """Rank jobs using combined BM25 and embedding-based similarity."""
    if not jobs:
        return []

    # Lexical similarity
    tokenized_jobs = [(j["title"] + " " + j["company"]).lower().split() for j in jobs]
    bm25 = BM25Okapi(tokenized_jobs)
    cv_tokens = cv_text.lower().split()
    bm25_scores = bm25.get_scores(cv_tokens)

    # Semantic similarity
    cv_emb = embed_model.encode(cv_text)
    job_embs = [embed_model.encode(j["title"] + " " + j["company"]) for j in jobs]
    sim_scores = [util.cos_sim(cv_emb, e).item() for e in job_embs]

    # Combine
    hybrid = [0.5 * b + 0.5 * s for b, s in zip(bm25_scores, sim_scores)]
    ranked = sorted(zip(hybrid, jobs), key=lambda x: x[0], reverse=True)

    seen, unique = set(), []
    for score, job in ranked:
        if job["link"] in seen:
            continue
        seen.add(job["link"])
        unique.append((score, job))
        if len(unique) >= top_n:
            break

    print(f"🏆 Ranked top {len(unique)} job matches.")
    return unique


# =====================================================
# 📧 Email Utility
# =====================================================
def send_email(recipient_email, subject, body):
    """Send summary email of top job matches."""
    if not SEND_EMAIL:
        print("[DRY RUN] Email body:\n" + body)
        return

    if not (SENDER_EMAIL and APP_PASSWORD):
        print("⚠️ Missing credentials. Previewing email:\n")
        print(body)
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
# 🔄 LangGraph Orchestration
# =====================================================
class JobState(TypedDict):
    cv_path: str
    cv_text: str
    jobs: list
    stored_jobs: list
    top_jobs: list
    email_body: str


def node_load_cv(state):
    """Extract text from uploaded CV."""
    cv_path = state.get("cv_path", "")
    if not cv_path or not os.path.exists(cv_path):
        print("⚠️ Please upload a CV PDF before running the workflow.")
        return state
    cv_text = extract_text_from_pdf(cv_path)
    state["cv_text"] = cv_text
    print(f"📄 CV loaded and text extracted ({len(cv_text)} chars).")
    return state


def node_scrape_or_skip(state):
    """Scrape new jobs or reuse stored ones."""
    existing = retrieve_jobs_from_chromadb()
    if existing and not SCRAPE_ALWAYS:
        print("📦 Using cached jobs from ChromaDB.")
        state["stored_jobs"] = existing
        return state

    print("🔍 Scraping fresh job postings...")
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


def node_format_email(state):
    """Format results into email summary."""
    ranked = state.get("top_jobs", [])
    if not ranked:
        state["email_body"] = "No job matches found."
        return state

    lines = [f"Top Job Matches — {datetime.now():%Y-%m-%d %H:%M}\n"]
    for i, (score, job) in enumerate(ranked, 1):
        tag = " ⭐" if score > 0.7 else ""
        lines.append(f"{i}. {job['title']} at {job['company']} ({job['location']}) — {score:.3f}{tag}")
        lines.append(f"   {job['link']}")
    state["email_body"] = "\n".join(lines)
    print("📧 Email body formatted.")
    return state


def node_send(state):
    send_email(RECIPIENT_EMAIL, "Daily AI-Powered Job Matches", state.get("email_body", ""))
    return state


# =====================================================
# 🚀 Build LangGraph Workflow
# =====================================================
workflow = StateGraph(JobState)
workflow.add_node("load_cv", node_load_cv)
workflow.add_node("scrape_or_skip", node_scrape_or_skip)
workflow.add_node("store", node_store)
workflow.add_node("retrieve", node_retrieve)
workflow.add_node("rank", node_rank)
workflow.add_node("format_email", node_format_email)
workflow.add_node("send", node_send)

workflow.set_entry_point("load_cv")
workflow.add_edge("load_cv", "scrape_or_skip")
workflow.add_edge("scrape_or_skip", "store")
workflow.add_edge("store", "retrieve")
workflow.add_edge("retrieve", "rank")
workflow.add_edge("rank", "format_email")
workflow.add_edge("format_email", "send")
workflow.add_edge("send", END)

graph = workflow.compile()


# =====================================================
# 🧭 Run Locally
# =====================================================
if __name__ == "__main__":
    uploaded_cv_path = input("📎 Enter path to your CV PDF: ").strip()

    if not uploaded_cv_path or not os.path.exists(uploaded_cv_path):
        print("⚠️ Please provide a valid CV path to continue.")
    else:
        initial = {
            "cv_path": uploaded_cv_path,
            "cv_text": "",
            "jobs": [],
            "stored_jobs": [],
            "top_jobs": [],
            "email_body": ""
        }
        graph.invoke(initial)



