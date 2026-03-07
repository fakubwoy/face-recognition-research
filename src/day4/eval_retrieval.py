# day4/eval_retrieval.py
"""
Day 4 Task 1: Retrieval Evaluation
Metrics: Rank-1 accuracy, Rank-5 accuracy, mAP, CMC curve

Output: results/retrieval_results.json
"""

import json
import time
import numpy as np
import faiss
import pickle
import cv2
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR   = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

FAISS_INDEX   = Path("../../embeddings/face_index.faiss")
METADATA_PKL  = Path("../../embeddings/metadata.pkl")
LFW_DIR       = Path("../../datasets/lfw_home/lfw_funneled")

QUICK         = __import__("os").environ.get("DAY4_QUICK") == "1"
MAX_QUERIES   = 30 if QUICK else 200


def load_index_and_metadata():
    print("Loading FAISS index and metadata...")
    index = faiss.read_index(str(FAISS_INDEX))
    with open(METADATA_PKL, "rb") as f:
        metadata = pickle.load(f)
    print(f"  ✓ Index: {index.ntotal} vectors, dim={index.d}")
    print(f"  ✓ Metadata: {len(metadata)} entries")

    # Inspect first entry to find the right keys
    if metadata:
        sample = metadata[0]
        print(f"  ✓ Metadata keys: {list(sample.keys()) if isinstance(sample, dict) else type(sample)}")
    return index, metadata


def get_path_key(metadata):
    """Auto-detect which key holds the image path."""
    if not metadata:
        return None
    sample = metadata[0]
    if not isinstance(sample, dict):
        return None
    for key in ["path", "img_path", "image_path", "file", "filepath", "filename"]:
        if key in sample:
            return key
    # fallback: find any key whose value looks like a file path
    for key, val in sample.items():
        if isinstance(val, str) and ("/" in val or "\\" in val or val.endswith(".jpg")):
            return key
    return None


def get_label_key(metadata):
    """Auto-detect which key holds the identity label."""
    if not metadata:
        return None
    sample = metadata[0]
    if not isinstance(sample, dict):
        return None
    for key in ["label", "name", "identity", "person", "class", "id"]:
        if key in sample:
            return key
    return None


def get_embedding(img_bgr, app):
    faces = app.get(img_bgr)
    if not faces:
        return None
    return faces[0].normed_embedding.astype("float32")


def average_precision(relevant_ranks, total_relevant):
    if total_relevant == 0:
        return 0.0
    hits, running_ap = 0, 0.0
    for i, is_relevant in enumerate(relevant_ranks, 1):
        if is_relevant:
            hits += 1
            running_ap += hits / i
    return running_ap / total_relevant


def main():
    print("=" * 60)
    print("DAY 4 — Task 1: Retrieval Evaluation (Rank-1, mAP, CMC)")
    print("=" * 60)

    if not FAISS_INDEX.exists():
        print(f"  ✗ FAISS index not found at {FAISS_INDEX}")
        result = {"error": "FAISS index not found", "path": str(FAISS_INDEX)}
        with open(RESULTS_DIR / "retrieval_results.json", "w") as f:
            json.dump(result, f, indent=2)
        return

    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
    except Exception as e:
        print(f"  ✗ InsightFace load failed: {e}")
        result = {"error": str(e)}
        with open(RESULTS_DIR / "retrieval_results.json", "w") as f:
            json.dump(result, f, indent=2)
        return

    index, metadata = load_index_and_metadata()

    path_key  = get_path_key(metadata)
    label_key = get_label_key(metadata)
    print(f"  ✓ Using path_key='{path_key}', label_key='{label_key}'")

    if path_key is None or label_key is None:
        print("  ✗ Could not detect path/label keys in metadata.")
        print(f"    Available keys: {list(metadata[0].keys()) if metadata else 'none'}")
        result = {
            "error": "Could not detect metadata keys",
            "available_keys": list(metadata[0].keys()) if metadata else [],
        }
        with open(RESULTS_DIR / "retrieval_results.json", "w") as f:
            json.dump(result, f, indent=2)
        return

    # Build label → list of metadata indices
    label_to_indices = defaultdict(list)
    for i, m in enumerate(metadata):
        label_to_indices[m[label_key]].append(i)

    # Only use people with ≥ 2 images
    query_candidates = [
        (label, idxs) for label, idxs in label_to_indices.items()
        if len(idxs) >= 2
    ]
    np.random.seed(42)
    np.random.shuffle(query_candidates)
    query_candidates = query_candidates[:MAX_QUERIES]
    print(f"\nRunning retrieval on {len(query_candidates)} query persons...")

    rank1_hits = 0
    rank5_hits = 0
    rank10_hits = 0
    aps = []
    cmc_counts = defaultdict(int)
    skipped = 0
    t0 = time.time()

    for person_label, person_idxs in query_candidates:
        query_meta = metadata[person_idxs[0]]
        img_path = Path(query_meta[path_key])

        # Try absolute path first, then relative from LFW_DIR
        if not img_path.exists():
            # Try constructing from LFW dir + label + filename
            fname = img_path.name
            candidate = LFW_DIR / person_label / fname
            if candidate.exists():
                img_path = candidate
            else:
                skipped += 1
                continue

        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        q_emb = get_embedding(img, app)
        if q_emb is None:
            skipped += 1
            continue

        k = min(50, index.ntotal)
        q_vec = q_emb.reshape(1, -1)
        faiss.normalize_L2(q_vec)
        distances, result_indices = index.search(q_vec, k)

        relevant_ranks = []
        found_at_rank = None
        for rank, idx in enumerate(result_indices[0]):
            if idx < 0 or idx >= len(metadata):
                continue
            matched_label = metadata[idx][label_key]
            is_relevant = (matched_label == person_label) and (idx != person_idxs[0])
            relevant_ranks.append(is_relevant)
            if is_relevant and found_at_rank is None:
                found_at_rank = rank

        total_relevant = len(person_idxs) - 1

        if found_at_rank is not None:
            if found_at_rank < 1:  rank1_hits += 1
            if found_at_rank < 5:  rank5_hits += 1
            if found_at_rank < 10: rank10_hits += 1
            for r in range(found_at_rank, 50):
                cmc_counts[r] += 1

        ap = average_precision(relevant_ranks, total_relevant)
        aps.append(ap)

    elapsed = time.time() - t0
    n = len(query_candidates) - skipped

    result = {
        "queries_attempted": len(query_candidates),
        "queries_succeeded": n,
        "queries_skipped": skipped,
        "rank1_accuracy":  round(rank1_hits / max(n, 1), 4),
        "rank5_accuracy":  round(rank5_hits / max(n, 1), 4),
        "rank10_accuracy": round(rank10_hits / max(n, 1), 4),
        "mAP": round(float(np.mean(aps)), 4) if aps else 0,
        "time_sec": round(elapsed, 2),
        "cmc_curve": {
            str(k): round(cmc_counts[k] / max(n, 1), 4)
            for k in sorted(cmc_counts.keys())[:20]
        },
    }

    with open(RESULTS_DIR / "retrieval_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Rank-1  Accuracy : {result['rank1_accuracy']:.4f}")
    print(f"  Rank-5  Accuracy : {result['rank5_accuracy']:.4f}")
    print(f"  Rank-10 Accuracy : {result['rank10_accuracy']:.4f}")
    print(f"  mAP              : {result['mAP']:.4f}")
    print(f"  Skipped          : {skipped}")
    print(f"  Time             : {elapsed:.1f}s")
    print(f"✓ Saved → results/retrieval_results.json")


if __name__ == "__main__":
    main()