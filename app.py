import os
import streamlit as st
from datetime import datetime
from sentence_transformers import SentenceTransformer

# =====================================================
# ⚙️ Environment & Telemetry Configuration
# =====================================================
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "1"   # Increase Hugging Face timeout
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Faster downloads

# =====================================================
# 📦 Local Imports
# =====================================================
from pipeline.config import CFG
from pipeline.cv_reader import extract_text_from_pdf
from pipeline.sources.boards import fetch_linkedin_jobs as fetch_jobs
from pipeline.vectordb import get_collection, store_jobs, read_jobs, clear_all_jobs
from pipeline.ranker import hybrid_rank
from pipeline.email_utils import send_email


# =====================================================
# 🧭 Streamlit Page Configuration
# =====================================================
st.set_page_config(page_title="Job Search Ranker", page_icon="🧭", layout="wide")
st.title("🧭 AI-Assisted Job Search Ranker")


# =====================================================
# 🎛 Sidebar Controls
# =====================================================
with st.sidebar:
    st.subheader("Search Settings")

    titles = [
        t.strip() for t in st.text_input(
            "Job titles (comma-separated)", 
            "Data Scientist"
        ).split(",") if t.strip()
    ]

    locations = [
        l.strip() for l in st.text_input(
            "Locations (comma-separated)", 
            "Quebec, Ontario"
        ).split(",") if l.strip()
    ]

    days_filter = st.number_input("Days posted ≤", 1, 60, 12)
    top_n = st.slider("Top N results", 5, 50, 20)
    scrape_always = st.checkbox("Scrape fresh each run", value=CFG["SCRAPE_ALWAYS"])
    use_selenium = st.checkbox("Use Selenium for full descriptions", value=CFG["USE_SELENIUM"])

    st.markdown("---")
    st.subheader("Email")
    send_mail = st.checkbox("Send email", value=CFG["SEND_EMAIL"])
    recip = st.text_input("Recipient email", value=CFG["RECIPIENT_EMAIL"] or "")

    st.markdown("---")
    st.caption("📎 Upload your CV (PDF). Required for job ranking.")


# =====================================================
# 📄 Step 1: Upload CV
# =====================================================
st.markdown("### 📄 Step 1: Upload Your CV")

cv_file = st.file_uploader(
    "Upload your CV in PDF format",
    type=["pdf"],
    help="The CV text will be extracted automatically for job matching."
)

if not cv_file:
    st.warning("⚠️ Please upload your CV to proceed.")
    st.stop()

os.makedirs("data", exist_ok=True)
temp_path = os.path.join("data", f"cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
with open(temp_path, "wb") as f:
    f.write(cv_file.read())

try:
    cv_text = extract_text_from_pdf(temp_path)
    st.success(f"✅ CV uploaded and text extracted ({len(cv_text)} characters).")
except Exception as e:
    st.error(f"❌ Failed to read CV: {e}")
    st.stop()


# =====================================================
# 🤖 Model & Database Initialization
# =====================================================
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embed(model_name: str):
    return SentenceTransformer(model_name)

embed = get_embed(CFG["EMBEDDING_MODEL"])


@st.cache_resource(show_spinner="Initializing vector database...")
def get_coll(path: str):
    return get_collection(path)

coll = get_coll(CFG["CHROMA_DIR"])


# =====================================================
# 🧩 Action Buttons
# =====================================================
colA, colB, colC = st.columns(3)
with colA:
    do_fetch = st.button("🔎 Fetch Jobs")
with colB:
    do_rank = st.button("🏆 Rank Matches")
with colC:
    do_email = st.button("📧 Email Results")


# =====================================================
# 🌐 Fetch LinkedIn Jobs
# =====================================================
jobs = []
if do_fetch:
    st.info("Fetching jobs from LinkedIn...")
    with st.spinner("⏳ Scraping in progress... this may take a few minutes..."):
        jobs = fetch_jobs(
            job_titles=titles,
            locations=locations,
            days_filter=days_filter,
            num_jobs=150,
            excluded_titles=[
                "Data Engineer", "Junior", "Intern", "Manager", "Head", "Director", "VP"
            ],
            use_selenium=use_selenium,
        )

    if jobs:
        if scrape_always:
            n = store_jobs(coll, jobs, embed)
            st.success(f"✅ {n} jobs stored in vector database.")
        else:
            st.info("ℹ️ Scrape complete — reusing existing database.")
    else:
        st.warning("⚠️ No new jobs found or scraping failed.")


# =====================================================
# 📂 Read Stored Jobs
# =====================================================
stored = read_jobs(coll)
st.write(f"🗂️ Jobs currently in database: **{len(stored)}**")


# =====================================================
# 🧠 Rank Jobs
# =====================================================
ranked = []
if do_rank:
    st.info("Ranking jobs based on CV relevance...")
    with st.spinner("⚙️ Calculating hybrid BM25 + semantic similarity..."):
        ranked = hybrid_rank(cv_text, stored, embed, top_n=top_n)

    if not ranked:
        st.warning("No ranked results found.")
    else:
        st.subheader("🏆 Top Job Matches")
        for i, (score, job) in enumerate(ranked, 1):
            st.markdown(
                f"**{i}. {job['title']}** — {job['company']} · {job['location']}  \n"
                f"Score: `{score:.4f}`  \n"
                f"[View Job Posting]({job['link']})"
            )


# =====================================================
# 📧 Send Email Results
# =====================================================
if do_email:
    if not ranked:
        ranked = hybrid_rank(cv_text, stored, embed, top_n=top_n)

    lines = [f"Top Job Matches — {datetime.now():%Y-%m-%d %H:%M}"]
    for i, (score, j) in enumerate(ranked, 1):
        tag = " (High Priority)" if score > 0.7 else ""
        lines.append(f"{i}. {j['title']} at {j['company']} ({j['location']}) — Score {score:.4f}{tag}")
        lines.append(f"   {j['link']}")
    body = "\n".join(lines)

    st.code(body, language="text")

    ok = send_email(
        sender=CFG["SENDER_EMAIL"],
        app_password=CFG["GMAIL_APP_PASSWORD"],
        recipient=recip or CFG["RECIPIENT_EMAIL"],
        subject="Daily AI-Powered Job Matches",
        body=body,
        enable=send_mail,
    )

    if ok and send_mail:
        st.success("✅ Email sent successfully.")
    else:
        st.info("📭 Dry-run mode — email preview only.")


# =====================================================
# 🧹 Database Maintenance
# =====================================================
st.markdown("---")
st.subheader("🧹 Database Maintenance")

if st.button("Clean Database (Full Reset)"):
    with st.spinner("Deleting all jobs from database..."):
        ok = clear_all_jobs(coll)
        if ok:
            st.success("✅ Database cleared successfully.")
        else:
            st.error("❌ Failed to clean database.")
