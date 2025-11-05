from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

def hybrid_rank(cv_text: str, jobs: List[Dict[str, Any]], embed: SentenceTransformer, top_n=20) -> List[Tuple[float, Dict]]:
    if not jobs:
        return []
    # BM25 on title+company
    corpus = [(j["title"] + " " + j["company"]).lower().split() for j in jobs]
    bm25 = BM25Okapi(corpus)
    tokenized_cv = cv_text.lower().split()
    bm25_scores = bm25.get_scores(tokenized_cv)

    # semantic on title+company
    cv_emb = embed.encode(cv_text, convert_to_tensor=True)
    job_embs = [embed.encode(j["title"] + " at " + j["company"], convert_to_tensor=True) for j in jobs]
    sim = [float(util.pytorch_cos_sim(cv_emb, e)) for e in job_embs]

    combo = [0.5*b + 0.5*s for b,s in zip(bm25_scores, sim)]
    ranked = sorted(zip(combo, jobs), key=lambda x: x[0], reverse=True)

    seen, out = set(), []
    for score, job in ranked:
        lk = job.get("link","")
        if lk in seen: continue
        seen.add(lk); out.append((score, job))
        if len(out) >= top_n: break
    return out
