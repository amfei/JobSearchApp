import os, pdfplumber

def extract_text_from_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"CV not found: {pdf_path}")
    with pdfplumber.open(pdf_path) as pdf:
        pages = [(p.extract_text() or "") for p in pdf.pages]
    return "\n".join(p.strip() for p in pages if p).strip()
