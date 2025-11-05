import os

from dotenv import load_dotenv
load_dotenv()

CFG = {
    "SEND_EMAIL": os.getenv("SEND_EMAIL", "false").lower() == "true",
    "SCRAPE_ALWAYS": os.getenv("SCRAPE_ALWAYS", "true").lower() == "true",
    "USE_SELENIUM": os.getenv("USE_SELENIUM", "false").lower() == "true",  # Cloud=false
    "CHROMA_DIR": os.getenv("CHROMA_DIR", ".chroma"),
    "CV_PDF_PATH": os.getenv("CV_PDF_PATH", "AMIR_FEIZI.pdf"),
    "SENDER_EMAIL": os.getenv("SENDER_EMAIL", ""),
    "GMAIL_APP_PASSWORD": os.getenv("GMAIL_APP_PASSWORD", ""),
    "RECIPIENT_EMAIL": os.getenv("RECIPIENT_EMAIL", ""),
    "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"),
}
