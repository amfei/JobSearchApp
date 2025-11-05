from typing import List, Dict, Any
import chromadb

def get_collection(path: str):
    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(name="jobs")

def store_jobs(coll, jobs: List[Dict[str, Any]], embed):
    if not jobs: return 0
    ids, embs, metas = [], [], []
    for j in jobs:
        text = f"{j['title']} {j['company']} {j['location']} {j.get('description','')}"
        vec = embed.encode(text, convert_to_tensor=True).tolist()
        jid = j.get("link") or f"{j['title']}-{j['company']}"
        ids.append(jid); embs.append(vec); metas.append(j)
    coll.add(ids=ids, embeddings=embs, metadatas=metas)
    return len(ids)

def read_jobs(coll) -> List[Dict[str, Any]]:
    res = coll.get()
    jobs = []
    for m in (res.get("metadatas") or []):
        jobs.append({
            "title": m.get("title",""),
            "company": m.get("company",""),
            "location": m.get("location",""),
            "link": m.get("link",""),
            "description": m.get("description","")
        })
    return jobs
