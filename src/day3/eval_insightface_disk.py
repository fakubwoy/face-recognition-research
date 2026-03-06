# day3/eval_insightface_disk.py
"""
Day 3 Task 1: InsightFace Re-Evaluation — Disk-Based Pipeline
Fixes the Day 2 in-memory numpy array issue by loading JPEG files directly.

Uses the actual LFW funneled directory + pairs.txt to run a proper evaluation.

Output: results/insightface_disk_results.json
"""

import json
import time
import os
import numpy as np
import cv2
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR   = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Adjust these paths relative to project root
LFW_FUNNELED  = Path("../../datasets/lfw_home/lfw_funneled")
PAIRS_FILE    = Path("../../datasets/lfw_home/pairs.txt")   # standard LFW pairs file
MAX_PAIRS     = int(os.environ.get("DAY3_MAX_PAIRS", 500))


# ─── helpers ─────────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    a, b = np.array(a, dtype="float32"), np.array(b, dtype="float32")
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (norm + 1e-8))


def find_best_threshold(sims, labels):
    best_acc, best_t = 0, 0.5
    for t in np.arange(0.05, 1.0, 0.01):
        preds = (np.array(sims) >= t).astype(int)
        acc   = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return round(best_acc, 4), round(float(best_t), 3)


def parse_pairs_file(pairs_file, lfw_root, max_pairs):
    """
    Parse standard LFW pairs.txt format.
    Returns list of (path1, path2, label) where label 1=same, 0=diff.
    """
    pairs = []
    if not pairs_file.exists():
        print(f"  ⚠  pairs.txt not found at {pairs_file}")
        print("     Falling back to sklearn LFW pairs loader...")
        return None

    with open(pairs_file) as f:
        lines = f.readlines()

    # First line: "<n_folds> <n_pairs>"
    n_folds, n_per_fold = map(int, lines[0].strip().split())
    same_lines = []
    diff_lines = []
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) == 3:
            same_lines.append(parts)   # name, idx1, idx2
        elif len(parts) == 4:
            diff_lines.append(parts)   # name1, idx1, name2, idx2

    def img_path(name, idx):
        return lfw_root / name / f"{name}_{int(idx):04d}.jpg"

    for name, i1, i2 in same_lines[:max_pairs // 2]:
        p1, p2 = img_path(name, i1), img_path(name, i2)
        if p1.exists() and p2.exists():
            pairs.append((str(p1), str(p2), 1))

    for name1, i1, name2, i2 in diff_lines[:max_pairs // 2]:
        p1, p2 = img_path(name1, i1), img_path(name2, i2)
        if p1.exists() and p2.exists():
            pairs.append((str(p1), str(p2), 0))

    return pairs


def load_pairs_sklearn_fallback(lfw_root, max_pairs):
    """Fallback: use sklearn but save images to temp disk files first."""
    from sklearn.datasets import fetch_lfw_pairs
    import tempfile

    print("  Loading via sklearn + writing to temp JPEG files...")
    data = fetch_lfw_pairs(
        subset="test",
        data_home=str(lfw_root.parent.parent),
        download_if_missing=True,
        resize=1.0,
        color=True,
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="lfw_disk_"))
    pairs   = []
    n       = min(max_pairs, len(data.target))

    for i in range(n):
        img1_rgb = (data.pairs[i, 0] * 255).astype(np.uint8)
        img2_rgb = (data.pairs[i, 1] * 255).astype(np.uint8)
        img1_bgr = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR)
        img2_bgr = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2BGR)

        p1 = str(tmp_dir / f"img_{i}_a.jpg")
        p2 = str(tmp_dir / f"img_{i}_b.jpg")
        cv2.imwrite(p1, img1_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(p2, img2_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        pairs.append((p1, p2, int(data.target[i])))

    same = sum(p[2] for p in pairs)
    print(f"  ✓ Saved {n} pairs to {tmp_dir} — {same} same, {n-same} diff")
    return pairs, tmp_dir


# ─── InsightFace evaluation ───────────────────────────────────────────────────

def eval_insightface_from_disk(pairs):
    """
    Load each image directly from disk via cv2.imread — avoids the
    float32/RGB in-memory array issue that caused 0 detections on Day 2.
    """
    print("\n[InsightFace] Evaluating from disk-loaded JPEG files...")
    try:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))

        sims, labels = [], []
        skipped_no_face = 0
        skipped_error   = 0
        t0 = time.time()

        for idx, (p1, p2, label) in enumerate(pairs):
            if idx % 50 == 0:
                print(f"    {idx}/{len(pairs)} pairs processed...")
            try:
                img1 = cv2.imread(p1)   # BGR uint8 — exactly what InsightFace expects
                img2 = cv2.imread(p2)

                if img1 is None or img2 is None:
                    skipped_error += 1
                    continue

                faces1 = app.get(img1)
                faces2 = app.get(img2)

                if not faces1 or not faces2:
                    skipped_no_face += 1
                    continue

                e1 = faces1[0].normed_embedding.astype("float32")
                e2 = faces2[0].normed_embedding.astype("float32")
                sims.append(cosine_sim(e1, e2))
                labels.append(label)

            except Exception as ex:
                skipped_error += 1

        elapsed = time.time() - t0

        if not sims:
            return {
                "framework": "InsightFace (ArcFace) — disk",
                "error": "No pairs processed. Check LFW path and file permissions.",
                "pairs_attempted": len(pairs),
            }

        acc, threshold = find_best_threshold(sims, labels)
        auc = round(roc_auc_score(labels, sims), 4) if len(set(labels)) > 1 else None

        same_sims = [s for s, l in zip(sims, labels) if l == 1]
        diff_sims = [s for s, l in zip(sims, labels) if l == 0]

        result = {
            "framework": "InsightFace (ArcFace w600k_r50) — disk-loaded",
            "pairs_tested": len(sims),
            "pairs_skipped_no_face": skipped_no_face,
            "pairs_skipped_error": skipped_error,
            "detection_rate": round(len(sims) / len(pairs), 4),
            "accuracy": acc,
            "best_threshold": threshold,
            "auc_roc": auc,
            "time_sec": round(elapsed, 2),
            "pairs_per_sec": round(len(sims) / elapsed, 3),
            "same_person_sim": {
                "mean": round(float(np.mean(same_sims)), 4) if same_sims else None,
                "std":  round(float(np.std(same_sims)), 4)  if same_sims else None,
                "min":  round(float(np.min(same_sims)), 4)  if same_sims else None,
                "max":  round(float(np.max(same_sims)), 4)  if same_sims else None,
            },
            "diff_person_sim": {
                "mean": round(float(np.mean(diff_sims)), 4) if diff_sims else None,
                "std":  round(float(np.std(diff_sims)), 4)  if diff_sims else None,
                "min":  round(float(np.min(diff_sims)), 4)  if diff_sims else None,
                "max":  round(float(np.max(diff_sims)), 4)  if diff_sims else None,
            },
            "separability": round(
                (np.mean(same_sims) - np.mean(diff_sims)) if (same_sims and diff_sims) else 0, 4
            ),
        }

        print(f"  ✓ acc={acc:.4f}, AUC={auc}, sep={result['separability']:.4f}")
        print(f"     Tested={len(sims)}, no_face={skipped_no_face}, error={skipped_error}")
        print(f"     Speed: {result['pairs_per_sec']:.2f} pairs/sec")
        return result

    except ImportError:
        return {"framework": "InsightFace (ArcFace)", "error": "insightface not installed"}
    except Exception as e:
        return {"framework": "InsightFace (ArcFace)", "error": str(e)}


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DAY 3 — Task 1: InsightFace Disk-Based Re-Evaluation")
    print("=" * 60)
    print(f"LFW path : {LFW_FUNNELED.resolve()}")
    print(f"Max pairs: {MAX_PAIRS}")

    tmp_dir = None
    pairs   = parse_pairs_file(PAIRS_FILE, LFW_FUNNELED, MAX_PAIRS)

    if pairs is None:
        pairs, tmp_dir = load_pairs_sklearn_fallback(LFW_FUNNELED, MAX_PAIRS)

    same = sum(p[2] for p in pairs)
    print(f"\nLoaded {len(pairs)} pairs — {same} same-person, {len(pairs)-same} different")

    result = eval_insightface_from_disk(pairs)

    # Cleanup temp files if we created them
    if tmp_dir:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    out = RESULTS_DIR / "insightface_disk_results.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Saved → {out}")
    print("\nKey result:")
    for k in ["pairs_tested", "detection_rate", "accuracy", "auc_roc",
              "pairs_per_sec", "separability"]:
        if k in result:
            print(f"  {k}: {result[k]}")


if __name__ == "__main__":
    main()