"""
vectordb.py — Handles Chroma vector store operations.
Optimized for modular imports and Streamlit caching.
"""

from typing import List, Dict, Any
import os
import chromadb
from sentence_transformers import SentenceTransformer

# ====== Disable noisy telemetry & parallel tokenizer warnings ======
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# =====================================================
# ✅ Cached Initialization (fast re-runs in Streamlit)
# =====================================================
try:
    import streamlit as st

    @st.cache_resource(show_spinner="Initializing Chroma persistent client...")
    def get_client_cached(path: str):
        """Initialize persistent Chroma client (cached across reruns)."""
        return chromadb.PersistentClient(path=path)

except Exception:
    # fallback when Streamlit is not running
    def get_client_cached(path: str):
        return chromadb.PersistentClient(path=path)


def get_collection(path: str):
    """Get or create a single persistent 'jobs' collection."""
    client = get_client_cached(path)
    return client.get_or_create_collection(name="jobs")


# =====================================================
# ✅ Shared Embedding Model (loaded once)
# =====================================================
try:
    import streamlit as st

    @st.cache_resource(show_spinner="Loading embedding model...")
    def get_embed(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        return SentenceTransformer(model_name)

except Exception:
    def get_embed(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        return SentenceTransformer(model_name)


# =====================================================
# ✅ Store / Read / Clear Operations
# =====================================================
def store_jobs(coll, jobs: List[Dict[str, Any]], embed) -> int:
    """Store job postings as embeddings in Chroma."""
    if not jobs:
        return 0

    ids, embs, metas = [], [], []

    for j in jobs:
        text = f"{j['title']} {j['company']} {j['location']} {j.get('description', '')}"
        vec = embed.encode(text, convert_to_tensor=True).tolist()
        jid = j.get("link") or f"{j['title']}-{j['company']}"
        ids.append(jid)
        embs.append(vec)
        metas.append({
            "title": j.get("title", ""),
            "company": j.get("company", ""),
            "location": j.get("location", ""),
            "link": j.get("link", ""),
            "description": j.get("description", "")
        })

    coll.add(ids=ids, embeddings=embs, metadatas=metas)
    return len(ids)


def read_jobs(coll) -> List[Dict[str, Any]]:
    """Retrieve all job entries from the Chroma collection."""
    res = coll.get()
    jobs = []
    for m in (res.get("metadatas") or []):
        jobs.append({
            "title": m.get("title", ""),
            "company": m.get("company", ""),
            "location": m.get("location", ""),
            "link": m.get("link", ""),
            "description": m.get("description", "")
        })
    return jobs


def clear_all_jobs(collection) -> bool:
    """Delete all job entries from the Chroma collection."""
    try:
        collection.delete(where={"title": {"$ne": ""}})
        print("✅ All job entries cleared from Chroma.")
        return True
    except Exception as e:
        print(f"❌ Failed to clear DB: {e}")
        return False
