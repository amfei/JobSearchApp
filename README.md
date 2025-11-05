🧭 AI-Assisted Job Search Ranker
🚀 Overview

JobSearchApp is an intelligent job-matching tool built with Streamlit, LangGraph-style pipelines, and semantic search to help users automatically find the most relevant job postings based on their CV.

It scrapes job data (from LinkedIn or other boards), extracts job information, ranks postings using both keyword (BM25) and semantic similarity (Sentence-Transformers), and optionally sends a daily summary email with top matches.

🧰 Features
Feature Description
📄 CV Parsing Extracts text automatically from uploaded or pre-defined PDF CVs
🌐 LinkedIn Scraper Uses requests + BeautifulSoup + (optional) Selenium / undetected_chromedriver
🧠 Hybrid Ranking Combines BM25 keyword relevance with Sentence-Transformer embeddings
💾 Vector Database Stores job postings in ChromaDB for re-ranking and historical searches
📧 Email Report Sends your top job matches directly to your inbox using Gmail App Password
🧩 Streamlit UI Interactive sidebar controls and job explorer dashboard
🔐 Secure Secrets Uses .env locally and Streamlit Secrets in production
🗂️ Project Structure
JobSearchApp_final/
├── app.py # Streamlit main app
├── pipeline/
│ ├── config.py # Environment + runtime settings
│ ├── cv_reader.py # PDF CV text extraction
│ ├── email_utils.py # Gmail SMTP email sender
│ ├── ranker.py # BM25 + embeddings hybrid ranking
│ ├── vectordb.py # ChromaDB vector store
│ └── sources/
│ ├── **init**.py
│ └── boards.py # LinkedIn scraper (requests + Selenium)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md

⚙️ Installation (Local / VSCode)

Clone the repo

git clone https://github.com/<your_username>/JobSearchApp_final.git
cd JobSearchApp_final

Create & activate virtual environment

python -m venv .venv
source .venv/bin/activate # macOS/Linux

# .venv\Scripts\activate # Windows

Install dependencies

pip install -U pip
pip install -r requirements.txt

Set up .env

SENDER_EMAIL=your_gmail@gmail.com
GMAIL_APP_PASSWORD=your_16_char_app_password
RECIPIENT_EMAIL=you@outlook.com
CV_PDF_PATH=./CV.pdf
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
USE_SELENIUM=true
SEND_EMAIL=true
SCRAPE_ALWAYS=true
CHROMA_DIR=.chroma

Run the app

streamlit run app.py

☁️ Deployment on Streamlit Cloud

Push your repo to GitHub.

Go to streamlit.io/cloud
→ Deploy New App → select your repo.

Add your secrets under Settings → Secrets in TOML format:

SENDER_EMAIL = "your_gmail@gmail.com"
GMAIL_APP_PASSWORD = "your_16_char_app_password"
RECIPIENT_EMAIL = "you@outlook.com"
CV_PDF_PATH = "./CV.pdf"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
USE_SELENIUM = "false"
SEND_EMAIL = "false"
SCRAPE_ALWAYS = "true"
CHROMA_DIR = ".chroma"

Click Deploy.
Note: Selenium will not run on Streamlit Cloud (set USE_SELENIUM=false).

📧 Gmail App Password Setup

If you get an authentication error (SMTPAuthenticationError 535):

Enable 2-Step Verification in your Google account.

Create an App Password for "Mail" → “Other (Streamlit)” → copy the 16-character code.

Paste that code as GMAIL_APP_PASSWORD in .env or Secrets.

Official Google guide →

🧠 Technical Details
Module Key Libraries
CV extraction pdfplumber
Job scraping requests, BeautifulSoup, selenium, webdriver-manager, undetected-chromedriver
Ranking rank_bm25, sentence-transformers, chromadb
UI streamlit
Email smtplib, email.mime
Workflow Designed modularly for LangGraph / DAG-style orchestration
🔒 Security Notes

Never commit your .env file — it’s in .gitignore.

Use Streamlit Secrets for deployment credentials.

Avoid running Selenium scraping on Streamlit Cloud (it needs a Chrome driver).

Use responsibly — scraping LinkedIn HTML is unofficial and may violate their ToS.

🧩 Example Workflow

Upload your CV (PDF)

Choose job titles & locations

Click Fetch Jobs (scrapes from LinkedIn)

Click Rank Matches (relevance to your CV)

Click Email Results to send a summary

🧭 Future Enhancements

Add progress bar with ETA during scraping

Integrate multiple job sources (Indeed, Glassdoor, Greenhouse)

Add OpenAI summarizer for each job description

Enable scheduling (daily auto-email jobs)

Build REST API endpoint for automation

👨‍💻 Author

Amir Feizi, PhD
AI & Data Science Engineer | Finzzor AI Founder
📍 Montreal, Canada
