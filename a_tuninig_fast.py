# =====================================================
# ⚡ tune_alpha_singlecv_fast_parallel.py
# Ultra-fast alpha tuning with precomputation + parallelism
# =====================================================
import numpy as np, json, random, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from job_ranker4 import extract_text_from_pdf, scrape_linkedin_jobs, evaluate_ranking

# =====================================================
# 1️⃣ Load CV & Model
# =====================================================
cv_path = "data/uploaded_cvs/current_cv.pdf"
cv_text = extract_text_from_pdf(cv_path)
print(f"✅ Extracted CV text: {cv_text[:80]}...\n")

embed_model = SentenceTransformer("thenlper/gte-large")

# =====================================================
# 2️⃣ Collect Job Sets
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
# 3️⃣ Utility: normalize arrays
# =====================================================
def norm(x):
    x = np.array(x, dtype=float)
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)

# =====================================================
# 4️⃣ Pre-encode embeddings ONCE for each job set
# =====================================================
print("⏳ Precomputing embeddings...")
precomputed = []
cv_emb = embed_model.encode(cv_text, normalize_embeddings=True)

for idx, jobs in enumerate(jobs_sets):
    job_texts = [
        (j.get("title", "") + " " + j.get("company", "") + " " + j.get("description", "")).strip()
        for j in jobs
    ]
    job_embs = embed_model.encode(job_texts, normalize_embeddings=True)
    cos = np.dot(job_embs, cv_emb)
    cos01 = np.clip((cos + 1.0) / 2.0, 0, 1)
    bm25_raw = np.array([j.get("bm25", 0.0) for j in jobs])
    precomputed.append({
        "jobs": jobs,
        "bm25": norm(bm25_raw) if bm25_raw.ptp() > 1e-8 else bm25_raw,
        "cos01": cos01,
        "cosn": norm(cos01)
    })
print(f"✅ Done precomputing for {len(precomputed)} sets.\n")

# =====================================================
# 5️⃣ Parallel Evaluation Function
# =====================================================
def evaluate_alpha(alpha):
    ndcg_list = []
    for data in precomputed:
        jobs, bm25, cos01, cosn = (
            data["jobs"],
            data["bm25"],
            data["cos01"],
            data["cosn"],
        )

        hybrid = alpha * bm25 + (1 - alpha) * cosn
        ranked_idx = np.argsort(hybrid)[::-1]
        ranked_jobs = [
            (float(cos01[i]), jobs[i], float(hybrid[i]), float(bm25[i]))
            for i in ranked_idx
        ]

        metrics = evaluate_ranking(cv_text, ranked_jobs, embed_model)
        ndcg_list.append(metrics.get("ndcg@5", 0))

    return alpha, float(np.mean(ndcg_list))

# =====================================================
# 6️⃣ Random / Parallel Search
# =====================================================
# Random search samples 10–15 alphas between 0.1–0.9
random.seed(42)
alphas = sorted(random.uniform(0.1, 0.9) for _ in range(10))

print(f"🚀 Running parallel tuning on {len(alphas)} alphas...")
start_time = time.time()
results = []

# Use threads (I/O heavy evaluation → good parallel speed-up)
with ThreadPoolExecutor(max_workers=min(8, len(alphas))) as executor:
    futures = [executor.submit(evaluate_alpha, a) for a in alphas]
    for fut in as_completed(futures):
        alpha, score = fut.result()
        results.append((alpha, score))
        print(f"   α={alpha:.2f} → mean NDCG@5={score:.3f}")

# Sort and select best
results.sort(key=lambda x: x[1], reverse=True)
best_alpha, best_score = results[0]

elapsed = time.time() - start_time
print(f"\n🏁 Finished in {elapsed:.2f} sec")
print(f"🏆 Best α = {best_alpha:.2f}  |  NDCG@5 = {best_score:.3f}")

# =====================================================
# 7️⃣ Save Result
# =====================================================
with open("config_alpha.json", "w") as f:
    json.dump({"best_alpha": best_alpha}, f, indent=2)
print("💾 Saved best alpha to config_alpha.json")
