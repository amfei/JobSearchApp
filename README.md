AI-Assisted Job Search Ranker
A Streamlit-based intelligent job-matching tool that reads your CV, scrapes job postings (e.g., from LinkedIn), and ranks them by relevance using both keyword and semantic similarity. It can also email you the top matches daily.
Key Features

Automatic CV text extraction (PDF)

Job scraping via Requests / Selenium

Hybrid BM25 + Sentence-Transformer ranking

Vector storage with ChromaDB

Optional daily email reports

Secure configuration via .env or Streamlit Secrets

How It Works
Upload your CV → Choose job titles & locations → Fetch → Rank → Email results.
Author
Amir Feizi, PhD — AI & Data Science Engineer, Montreal, Canada.
