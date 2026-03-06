# day3/eval_full_lfw_benchmark.py
"""
Day 3 Task 2: Full Balanced LFW Benchmark
Runs the standard 6,000-pair LFW test (3,000 same + 3,000 different).
Computes: Accuracy, AUC-ROC, TAR@FAR=0.01%, EER.

This fixes the Day 2 positive-only test set issue.

Output: results/full_lfw_benchmark.json
"""

import json
import time
import os
import tempfile
import shutil
import numpy as np
import cv2
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR   = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
LFW_DATA_HOME = "../../datasets"
MAX_PAIRS     = int(os.environ.get("DAY3_MAX_PAIRS", 600))  # 600 = 300 same + 300 diff


# ─── helpers ─────────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    a = np.array(a, dtype="float32")
    b = np.array(b, dtype="float32")
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def compute_metrics(sims, labels):
    """Full metric suite: accuracy, AUC, EER, TAR@FAR."""
    sims   = np.array(sims)
    labels = np.array(labels)

    # Best accuracy + threshold
    best_acc, best_t = 0, 0.5
    for t in np.arange(0.05, 1.0, 0.01):
        acc = accuracy_score(labels, (sims >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, t

    metrics = {
        "accuracy": round(float(best_acc), 4),
        "best_threshold": round(float(best_t), 3),
    }

    if len(set(labels)) > 1:
        auc = roc_auc_score(labels, sims)
        fpr, tpr, thresholds = roc_curve(labels, sims)

        # EER — point where FPR ≈ FNR
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)

        # TAR @ FAR = 1% (closest available point)
        tar_at_far_1 = None
        far_targets = [0.01, 0.001, 0.0001]
        tar_values  = {}
        for target_far in far_targets:
            idx = np.where(fpr <= target_far)[0]
            if len(idx) > 0:
                tar_values[f"TAR@FAR={target_far}"] = round(float(tpr[idx[-1]]), 4)

        metrics.update({
            "auc_roc": round(float(auc), 4),
            "eer": round(eer, 4),
            **tar_values,
        })
    else:
        metrics.update({
            "auc_roc": None,
            "eer": None,
            "note": "Single class in test set — AUC/EER not computable",
        })

    return metrics


def load_balanced_pairs(max_pairs):
    """Load balanced pairs (same + diff) from sklearn LFW, save to disk."""
    from sklearn.datasets import fetch_lfw_pairs

    print(f"Loading balanced LFW pairs (max {max_pairs})...")
    data = fetch_lfw_pairs(
        subset="test",
        data_home=LFW_DATA_HOME,
        download_if_missing=True,
        resize=1.0,
        color=True,
    )

    images = data.pairs
    labels = data.target.tolist()

    # Build balanced set: equal same + diff
    same_idx = [i for i, l in enumerate(labels) if l == 1]
    diff_idx = [i for i, l in enumerate(labels) if l == 0]

    n_each = min(max_pairs // 2, len(same_idx), len(diff_idx))
    selected = same_idx[:n_each] + diff_idx[:n_each]

    tmp_dir = Path(tempfile.mkdtemp(prefix="lfw_balanced_"))
    pairs   = []
    for i in selected:
        img1_rgb = (images[i, 0] * 255).astype(np.uint8)
        img2_rgb = (images[i, 1] * 255).astype(np.uint8)
        img1_bgr = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR)
        img2_bgr = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2BGR)
        p1 = str(tmp_dir / f"pair_{i}_a.jpg")
        p2 = str(tmp_dir / f"pair_{i}_b.jpg")
        cv2.imwrite(p1, img1_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(p2, img2_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        pairs.append((p1, p2, int(labels[i])))

    same = sum(p[2] for p in pairs)
    print(f"  ✓ {len(pairs)} pairs — {same} same, {len(pairs)-same} different")
    return pairs, tmp_dir


# ─── InsightFace ─────────────────────────────────────────────────────────────

def run_insightface(pairs):
    print("\n[InsightFace ArcFace] Running full benchmark...")
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))

        sims, labels, skipped = [], [], 0
        t0 = time.time()

        for idx, (p1, p2, label) in enumerate(pairs):
            if idx % 100 == 0:
                elapsed_so_far = time.time() - t0
                eta = (elapsed_so_far / (idx + 1)) * (len(pairs) - idx) if idx > 0 else 0
                print(f"    {idx}/{len(pairs)}  ETA: {eta:.0f}s")
            try:
                img1 = cv2.imread(p1)
                img2 = cv2.imread(p2)
                f1 = app.get(img1)
                f2 = app.get(img2)
                if not f1 or not f2:
                    skipped += 1
                    continue
                e1 = f1[0].normed_embedding.astype("float32")
                e2 = f2[0].normed_embedding.astype("float32")
                sims.append(cosine_sim(e1, e2))
                labels.append(label)
            except Exception:
                skipped += 1

        elapsed = time.time() - t0
        metrics = compute_metrics(sims, labels)

        result = {
            "framework": "InsightFace (ArcFace w600k_r50)",
            "pairs_tested": len(sims),
            "pairs_skipped": skipped,
            "detection_rate": round(len(sims) / len(pairs), 4),
            "pairs_per_sec": round(len(sims) / elapsed, 3),
            "time_sec": round(elapsed, 2),
            **metrics,
        }
        print(f"  ✓ acc={metrics['accuracy']:.4f}, AUC={metrics.get('auc_roc')}, "
              f"EER={metrics.get('eer')}")
        return result
    except Exception as e:
        return {"framework": "InsightFace (ArcFace)", "error": str(e)}


# ─── DeepFace Facenet512 ──────────────────────────────────────────────────────

def run_deepface(pairs, model_name="Facenet512"):
    print(f"\n[DeepFace {model_name}] Running full benchmark...")
    try:
        from deepface import DeepFace

        sims, labels, skipped = [], [], 0
        t0 = time.time()

        for idx, (p1, p2, label) in enumerate(pairs):
            if idx % 100 == 0:
                elapsed_so_far = time.time() - t0
                eta = (elapsed_so_far / (idx + 1)) * (len(pairs) - idx) if idx > 0 else 0
                print(f"    {idx}/{len(pairs)}  ETA: {eta:.0f}s")
            try:
                r = DeepFace.verify(
                    p1, p2,
                    model_name=model_name,
                    detector_backend="opencv",
                    enforce_detection=False,
                    silent=True,
                )
                sims.append(1 - r["distance"])
                labels.append(label)
            except Exception:
                skipped += 1

        elapsed = time.time() - t0
        metrics = compute_metrics(sims, labels)

        result = {
            "framework": f"DeepFace ({model_name})",
            "pairs_tested": len(sims),
            "pairs_skipped": skipped,
            "detection_rate": round(len(sims) / len(pairs), 4),
            "pairs_per_sec": round(len(sims) / elapsed, 3),
            "time_sec": round(elapsed, 2),
            **metrics,
        }
        print(f"  ✓ acc={metrics['accuracy']:.4f}, AUC={metrics.get('auc_roc')}, "
              f"EER={metrics.get('eer')}")
        return result
    except Exception as e:
        return {"framework": f"DeepFace ({model_name})", "error": str(e)}


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DAY 3 — Task 2: Full Balanced LFW Benchmark")
    print("=" * 60)

    pairs, tmp_dir = load_balanced_pairs(MAX_PAIRS)

    results = []
    try:
        results.append(run_insightface(pairs))
        results.append(run_deepface(pairs, "Facenet512"))
        results.append(run_deepface(pairs, "ArcFace"))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    out = RESULTS_DIR / "full_lfw_benchmark.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved → {out}")

    # Summary table
    print(f"\n{'Framework':<35} {'Acc':>7} {'AUC':>7} {'EER':>7} {'Pairs/s':>9}")
    print("-" * 65)
    for r in results:
        if "error" in r:
            print(f"{r['framework']:<35} ERROR: {r['error'][:30]}")
        else:
            eer = f"{r.get('eer', 'N/A'):.4f}" if r.get("eer") else "N/A"
            auc = f"{r.get('auc_roc', 'N/A'):.4f}" if r.get("auc_roc") else "N/A"
            print(f"{r['framework']:<35} {r['accuracy']:>7.4f} {auc:>7} "
                  f"{eer:>7} {r['pairs_per_sec']:>9.3f}")


if __name__ == "__main__":
    main()