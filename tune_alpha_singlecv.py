# =====================================================
# 🧭 tune_alpha_singlecv.py — Optimized Alpha Tuning
# =====================================================
import numpy as np, json, random, time
from sentence_transformers import SentenceTransformer
from job_ranker4 import (
    extract_text_from_pdf,
    scrape_linkedin_jobs,
    evaluate_ranking,
)

# =====================================================
# 1️⃣ Load CV and Model
# =====================================================
cv_path = "data/uploaded_cvs/current_cv.pdf"
cv_text = extract_text_from_pdf(cv_path)
print(f"✅ Extracted CV text: {cv_text[:80]}...\n")

# Load embedding model once
embed_model = SentenceTransformer("thenlper/gte-large")


# =====================================================
# 2️⃣ Collect Job Sets (diversify search space)
# =====================================================
titles_variants = [
    ["Data Scientist"],
    ["Machine Learning Engineer"],
    ["AI Specialist"],
]
locations = ["Toronto", "Ontario"]

jobs_sets = []
for titles in titles_variants:
    jobs = scrape_linkedin_jobs(titles, locations, num_jobs=25)
    if jobs:
        jobs_sets.append(jobs)

print(f"✅ Collected {len(jobs_sets)} job sets.\n")

# =====================================================
# 3️⃣ Utility: Normalization Helper
# =====================================================
def norm(x):
    x = np.array(x, dtype=float)
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)


# =====================================================
# 4️⃣ Optimized Single-CV Alpha Tuner
# =====================================================
def tune_alpha_singlecv_fast(cv_text, jobs_sets, embed_model, alphas=None):
    """
    Tune alpha by maximizing mean NDCG@5 across multiple job sets for a single CV.
    Embeddings are computed once per job set (fast).
    """
    if alphas is None:
        alphas = np.linspace(0.1, 0.9, 10)  # 0.1, 0.15, …, 0.9

    results = []
    start_time = time.time()

    for alpha in alphas:
        print(f"🔍 Evaluating α = {alpha:.2f}")
        ndcg_list = []

        for jobs in jobs_sets:
            if not jobs:
                continue

            # --- Precompute only once per job set ---
            bm25_raw = np.array([j.get("bm25", 0.0) for j in jobs])
            bm25 = norm(bm25_raw) if bm25_raw.ptp() > 1e-8 else bm25_raw

            job_texts = [
                (j.get("title", "") + " " + j.get("company", "") + " " + j.get("description", "")).strip()
                for j in jobs
            ]
            # encode only once
            cv_emb = embed_model.encode(cv_text, normalize_embeddings=True)
            job_embs = embed_model.encode(job_texts, normalize_embeddings=True)

            # cosine similarities in [0,1]
            cos = np.dot(job_embs, cv_emb)
            cos01 = np.clip((cos + 1.0) / 2.0, 0, 1)
            cosn = norm(cos01)

            # fuse BM25 and cosine
            hybrid = alpha * bm25 + (1 - alpha) * cosn

            # rank
            ranked_idx = np.argsort(hybrid)[::-1]
            ranked_jobs = [
                (float(cos01[i]), jobs[i], float(hybrid[i]), float(bm25[i]))
                for i in ranked_idx
            ]

            metrics = evaluate_ranking(cv_text, ranked_jobs, embed_model)
            ndcg_list.append(metrics.get("ndcg@5", 0))

        if ndcg_list:
            mean_ndcg = np.mean(ndcg_list)
            results.append((alpha, mean_ndcg))
            print(f"   → mean NDCG@5 = {mean_ndcg:.3f}")

    best_alpha, best_score = max(results, key=lambda x: x[1])
    elapsed = time.time() - start_time
    print(f"\n✅ Finished tuning in {elapsed:.2f} seconds.")
    return best_alpha, results


# =====================================================
# 5️⃣ Run Tuning
# =====================================================
best_alpha, results = tune_alpha_singlecv_fast(cv_text, jobs_sets, embed_model)

print(f"\n🏆 Best α = {best_alpha:.2f}")
print("───────────────────────────────")
for a, score in results:
    print(f"α = {a:.2f} → mean NDCG@5 = {score:.3f}")
print("───────────────────────────────")

# =====================================================
# 6️⃣ Save Result for Your App
# =====================================================
with open("config_alpha.json", "w") as f:
    json.dump({"best_alpha": best_alpha}, f, indent=2)
print("💾 Saved best alpha to config_alpha.json")
