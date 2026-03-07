# day4/eval_clustering.py
"""
Day 4 Task 2: Face Clustering Evaluation
Metrics: NMI, ARI, purity
Uses MiniBatchKMeans to avoid the freeze on k=1807.

Output: results/clustering_results.json
"""

import json
import time
import numpy as np
import pickle
from pathlib import Path
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
)
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR  = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

EMBEDDINGS_NPY = Path("../../embeddings/embeddings.npy")
METADATA_PKL   = Path("../../embeddings/metadata.pkl")

QUICK = __import__("os").environ.get("DAY4_QUICK") == "1"
MAX_SAMPLES = 500 if QUICK else 3000


def cluster_purity(labels_true, labels_pred):
    total = 0
    for cluster_id in set(labels_pred):
        mask = labels_pred == cluster_id
        counts = Counter(labels_true[mask])
        total += counts.most_common(1)[0][1]
    return total / len(labels_true)


def eval_dbscan(embeddings, labels_true, name, eps, min_samples=2):
    print(f"  [{name}]...")
    t0 = time.time()
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1)
    labels_pred = model.fit_predict(embeddings)
    elapsed = time.time() - t0

    n_clusters  = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)
    noise_ratio = round(float(np.sum(labels_pred == -1) / len(labels_pred)), 4)

    mask = labels_pred != -1
    if mask.sum() < 10:
        result = {"method": name, "error": "too many noise points",
                  "n_clusters_found": n_clusters, "noise_ratio": noise_ratio}
        print(f"    ✗ too many noise points ({noise_ratio:.0%})")
        return result

    nmi    = round(float(normalized_mutual_info_score(labels_true[mask], labels_pred[mask])), 4)
    ari    = round(float(adjusted_rand_score(labels_true[mask], labels_pred[mask])), 4)
    purity = round(float(cluster_purity(labels_true[mask], labels_pred[mask])), 4)

    print(f"    NMI={nmi:.4f}, ARI={ari:.4f}, Purity={purity:.4f}, "
          f"clusters={n_clusters}, noise={noise_ratio:.1%}, t={elapsed:.1f}s")
    return {
        "method": name,
        "n_clusters_found": n_clusters,
        "n_clusters_true": int(len(set(labels_true))),
        "nmi": nmi, "ari": ari, "purity": purity,
        "noise_ratio": noise_ratio,
        "time_sec": round(elapsed, 2),
    }


def eval_minibatch_kmeans(embeddings, labels_true, name, n_clusters):
    print(f"  [{name}]...")
    t0 = time.time()
    # MiniBatchKMeans is orders of magnitude faster than KMeans for large k
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=3,
        batch_size=1024,
        max_iter=100,
    )
    labels_pred = model.fit_predict(embeddings)
    elapsed = time.time() - t0

    nmi    = round(float(normalized_mutual_info_score(labels_true, labels_pred)), 4)
    ari    = round(float(adjusted_rand_score(labels_true, labels_pred)), 4)
    purity = round(float(cluster_purity(labels_true, labels_pred)), 4)

    print(f"    NMI={nmi:.4f}, ARI={ari:.4f}, Purity={purity:.4f}, "
          f"k={n_clusters}, t={elapsed:.1f}s")
    return {
        "method": name,
        "n_clusters_found": n_clusters,
        "n_clusters_true": int(len(set(labels_true))),
        "nmi": nmi, "ari": ari, "purity": purity,
        "noise_ratio": 0.0,
        "time_sec": round(elapsed, 2),
    }


def main():
    print("=" * 60)
    print("DAY 4 — Task 2: Face Clustering Evaluation")
    print("=" * 60)

    if not EMBEDDINGS_NPY.exists() or not METADATA_PKL.exists():
        print(f"  ✗ Embeddings not found at {EMBEDDINGS_NPY}")
        result = {"error": "Embeddings not found", "path": str(EMBEDDINGS_NPY)}
        with open(RESULTS_DIR / "clustering_results.json", "w") as f:
            json.dump(result, f, indent=2)
        return

    print("Loading embeddings...")
    embeddings = np.load(str(EMBEDDINGS_NPY)).astype("float32")
    with open(METADATA_PKL, "rb") as f:
        metadata = pickle.load(f)

    # Auto-detect label key
    sample = metadata[0] if metadata else {}
    label_key = next(
        (k for k in ["label", "name", "identity", "person", "class"] if k in sample),
        list(sample.keys())[0] if sample else "label"
    )
    print(f"  ✓ Using label_key='{label_key}'")

    le = LabelEncoder()
    all_labels = np.array([m[label_key] for m in metadata])
    labels_encoded = le.fit_transform(all_labels)

    # Subset
    n = min(MAX_SAMPLES, len(embeddings))
    idx = np.random.RandomState(42).choice(len(embeddings), n, replace=False)
    emb_sub = embeddings[idx]
    lab_sub = labels_encoded[idx]

    # L2 normalize for cosine similarity
    norms = np.linalg.norm(emb_sub, axis=1, keepdims=True)
    emb_sub = emb_sub / (norms + 1e-8)

    n_true = len(set(lab_sub))
    print(f"  Subset: {n} embeddings, {n_true} unique identities\n")

    # Cap k at 200 for MiniBatchKMeans to keep it fast
    k_exact  = min(n_true, 200)
    k_half   = max(k_exact // 2, 2)

    results = []
    results.append(eval_dbscan(emb_sub, lab_sub, "DBSCAN (eps=0.4)", eps=0.4))
    results.append(eval_dbscan(emb_sub, lab_sub, "DBSCAN (eps=0.5)", eps=0.5))
    results.append(eval_minibatch_kmeans(emb_sub, lab_sub, f"MiniBatchKMeans (k={k_exact})", k_exact))
    results.append(eval_minibatch_kmeans(emb_sub, lab_sub, f"MiniBatchKMeans (k={k_half})", k_half))

    with open(RESULTS_DIR / "clustering_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved → results/clustering_results.json")


if __name__ == "__main__":
    main()