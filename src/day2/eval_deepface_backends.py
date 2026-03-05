# day2/eval_deepface_backends.py
"""
Day 2 Task 4: DeepFace Backend Deep Comparison
Side-by-side comparison of ArcFace, Facenet512, VGG-Face, OpenFace
on same pairs — includes timing, memory, similarity distributions.

Output: results/deepface_backends.json
"""

import json
import time
import os
import tempfile
import numpy as np
import cv2
from pathlib import Path
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR   = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
LFW_DATA_HOME = "../../datasets"
MAX_PAIRS     = 80   # keep reasonable — VGG-Face is slow

MODELS_TO_TEST = ["ArcFace", "Facenet512", "VGG-Face", "OpenFace"]


def cosine_sim(a, b):
    a, b = np.array(a, dtype="float32"), np.array(b, dtype="float32")
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def find_best_threshold(sims, labels):
    best_acc, best_t = 0, 0.5
    for t in np.arange(0.1, 1.0, 0.01):
        preds = (np.array(sims) >= t).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return round(best_acc, 4), round(float(best_t), 2)


def load_pairs():
    print("Loading LFW pairs...")
    data = fetch_lfw_pairs(subset="test", data_home=LFW_DATA_HOME,
                           download_if_missing=True, resize=1.0, color=True)
    n = min(MAX_PAIRS, len(data.target))
    pairs = []
    for i in range(n):
        img1 = (data.pairs[i, 0] * 255).astype(np.uint8)
        img2 = (data.pairs[i, 1] * 255).astype(np.uint8)
        pairs.append((
            cv2.cvtColor(img1, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(img2, cv2.COLOR_RGB2BGR),
            int(data.target[i]),
        ))
    print(f"  ✓ {n} pairs — {sum(p[2] for p in pairs)} same, {sum(1-p[2] for p in pairs)} diff")
    return pairs


def eval_model(pairs, model_name):
    print(f"\n  [{model_name}]...")
    from deepface import DeepFace

    sims_same, sims_diff = [], []
    sims_all, labels, skipped = [], [], 0
    t0 = time.time()

    for img1, img2, label in pairs:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
            cv2.imwrite(f1.name, img1); cv2.imwrite(f2.name, img2)
            p1, p2 = f1.name, f2.name
        try:
            r = DeepFace.verify(p1, p2, model_name=model_name,
                                detector_backend="opencv",
                                enforce_detection=False, silent=True)
            sim = 1 - r["distance"]
            sims_all.append(sim)
            labels.append(label)
            if label == 1:
                sims_same.append(sim)
            else:
                sims_diff.append(sim)
        except Exception as ex:
            skipped += 1
        finally:
            os.unlink(p1); os.unlink(p2)

    elapsed = time.time() - t0

    if not sims_all:
        return {"model": model_name, "error": "all pairs failed"}

    acc, thr = find_best_threshold(sims_all, labels)
    auc = round(roc_auc_score(labels, sims_all), 4) if len(set(labels)) > 1 else None

    result = {
        "model": model_name,
        "pairs_tested": len(sims_all),
        "pairs_skipped": skipped,
        "accuracy": acc,
        "best_threshold": thr,
        "auc_roc": auc,
        "time_sec": round(elapsed, 2),
        "sec_per_pair": round(elapsed / max(len(sims_all), 1), 3),
        "same_person_sim": {
            "mean": round(float(np.mean(sims_same)), 4) if sims_same else None,
            "std":  round(float(np.std(sims_same)), 4) if sims_same else None,
            "min":  round(float(np.min(sims_same)), 4) if sims_same else None,
            "max":  round(float(np.max(sims_same)), 4) if sims_same else None,
        },
        "diff_person_sim": {
            "mean": round(float(np.mean(sims_diff)), 4) if sims_diff else None,
            "std":  round(float(np.std(sims_diff)), 4) if sims_diff else None,
            "min":  round(float(np.min(sims_diff)), 4) if sims_diff else None,
            "max":  round(float(np.max(sims_diff)), 4) if sims_diff else None,
        },
        "separability": round(
            (np.mean(sims_same) - np.mean(sims_diff)) if (sims_same and sims_diff) else 0, 4
        ),
    }
    print(f"    acc={acc:.4f}, AUC={auc}, sep={result['separability']:.4f}, t={elapsed:.1f}s")
    return result


def main():
    print("=" * 60)
    print("DAY 2 — Task 4: DeepFace Backend Comparison")
    print("=" * 60)

    pairs = load_pairs()
    results = []

    for model in MODELS_TO_TEST:
        try:
            results.append(eval_model(pairs, model))
        except Exception as e:
            results.append({"model": model, "error": str(e)})

    out = RESULTS_DIR / "deepface_backends.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved → {out}")

    print(f"\n{'Model':<15} {'Acc':>7} {'AUC':>7} {'Sep':>7} {'s/pair':>8}")
    print("-" * 52)
    for r in results:
        if "error" in r:
            print(f"{r['model']:<15} ERROR: {r['error']}")
        else:
            print(f"{r['model']:<15} {r['accuracy']:>7.4f} "
                  f"{str(r['auc_roc']):>7} {r['separability']:>7.4f} "
                  f"{r['sec_per_pair']:>8.3f}")


if __name__ == "__main__":
    main()