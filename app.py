import streamlit as st
from datetime import datetime
from sentence_transformers import SentenceTransformer

# --- Local pipeline imports ---
from pipeline.config import CFG
from pipeline.cv_reader import extract_text_from_pdf
from pipeline.sources.boards import fetch_linkedin_jobs as fetch_jobs  # ✅ unified import
from pipeline.vectordb import get_collection, store_jobs, read_jobs
from pipeline.ranker import hybrid_rank
from pipeline.email_utils import send_email


# ======================
# 🔧 Streamlit Settings
# ======================
st.set_page_config(page_title="Job Search Ranker", page_icon="🧭", layout="wide")
st.title("🧭 AI-Assisted Job Search Ranker")

# ======================
# 🎛️ Sidebar Controls
# ======================
with st.sidebar:
    st.subheader("Search Settings")

    default_titles = ["Data Scientist"]
    titles = st.text_input("Job titles (comma-separated)", ", ".join(default_titles)).split(",")
    titles = [t.strip() for t in titles if t.strip()]

    default_locs = ["Quebec", "Ontario"]
    locations = st.text_input("Locations (comma-separated)", ", ".join(default_locs)).split(",")
    locations = [l.strip() for l in locations if l.strip()]

    days_filter = st.number_input("Days posted ≤", 1, 60, 12)
    top_n = st.slider("Top N results", 5, 50, 20)
    scrape_always = st.checkbox("Scrape fresh each run", value=CFG["SCRAPE_ALWAYS"])
    use_selenium = st.checkbox("Use Selenium for full descriptions", value=CFG["USE_SELENIUM"])

    st.markdown("---")
    st.subheader("Email")
    send_mail = st.checkbox("Send email", value=CFG["SEND_EMAIL"])
    recip = st.text_input("Recipient email", value=CFG["RECIPIENT_EMAIL"] or "")

    st.markdown("---")
    st.caption("📄 Upload your CV (PDF) or use the one from `.env` path.")


# ======================
# 📄 CV Upload & Parsing
# ======================
cv_file = st.file_uploader("Upload CV (PDF)", type=["pdf"])
if cv_file:
    with open("uploaded_cv.pdf", "wb") as f:
        f.write(cv_file.read())
    cv_text = extract_text_from_pdf("uploaded_cv.pdf")
else:
    cv_text = extract_text_from_pdf(CFG["CV_PDF_PATH"])

if cv_text:
    st.success(f"✅ CV loaded successfully ({len(cv_text)} characters)")
else:
    st.warning("⚠️ CV text is empty or invalid PDF provided")

st.divider()


# ======================
# 🧠 Load Models & DB
# ======================
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embed(model_name: str):
    return SentenceTransformer(model_name)

embed = get_embed(CFG["EMBEDDING_MODEL"])


@st.cache_resource(show_spinner="Initializing vector database...")
def get_coll(path: str):
    return get_collection(path)

coll = get_coll(CFG["CHROMA_DIR"])


# ======================
# 🚀 UI Buttons
# ======================
colA, colB, colC = st.columns(3)
with colA:
    do_fetch = st.button("🔎 Fetch Jobs")
with colB:
    do_rank = st.button("🏆 Rank Matches")
with colC:
    do_email = st.button("📧 Email Results")


# ======================
# 🔍 Fetch LinkedIn Jobs
# ======================
jobs = []
if do_fetch:
    st.info("Fetching jobs from LinkedIn...")
    with st.spinner("Scraping in progress... (this may take up to a few minutes)"):
        jobs = fetch_jobs(
            job_titles=titles,
            locations=locations,
            days_filter=days_filter,
            num_jobs=150,
            excluded_titles=[
                "Data Engineer", "Junior", "Intern", "Manager", "Head", "Director", "VP"
            ],
            use_selenium=use_selenium
        )
        if scrape_always and jobs:
            n = store_jobs(coll, jobs, embed)
            st.success(f"✅ {n} jobs stored in vector database")
        else:
            st.warning("No new jobs found or scrape skipped.")


# ======================
# 💾 Load Stored Jobs
# ======================
stored = read_jobs(coll)
st.write(f"📦 Jobs currently in database: **{len(stored)}**")


# ======================
# 🧩 Rank by Relevance
# ======================
ranked = []
if do_rank:
    st.info("Ranking jobs based on CV relevance...")
    with st.spinner("Calculating hybrid BM25 + semantic scores..."):
        ranked = hybrid_rank(cv_text, stored, embed, top_n=top_n)

    if not ranked:
        st.warning("No ranked results found.")
    else:
        st.subheader("🏆 Top Job Matches")
        for i, (score, job) in enumerate(ranked, 1):
            st.markdown(
                f"**{i}. {job['title']}** — {job['company']} · {job['location']}  \n"
                f"💡 Score: `{score:.4f}`  \n"
                f"[🔗 View Job Posting]({job['link']})"
            )


# ======================
# 📧 Send Email Results
# ======================
if do_email:
    if not ranked:
        ranked = hybrid_rank(cv_text, stored, embed, top_n=top_n)

    lines = [f"Top Job Matches — {datetime.now():%Y-%m-%d %H:%M}"]
    for i, (score, j) in enumerate(ranked, 1):
        tag = " ⭐ HIGH PRIORITY" if score > 0.7 else ""
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
        enable=send_mail
    )
    if ok and send_mail:
        st.success("✅ Email sent successfully!")
    else:
        st.info("📭 Dry-run mode — email preview only.")
