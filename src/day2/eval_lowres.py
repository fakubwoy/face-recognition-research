# day2/eval_lowres.py
"""
Day 2 Task 2: Low-Resolution Stress Test
Downscales LFW pairs to simulate CCTV/compressed images and
measures how each framework degrades at lower resolutions.

Resolutions tested: original, 128x128, 64x64, 32x32

Output: results/lowres_results.json
"""

import json
import time
import numpy as np
import cv2
from pathlib import Path
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

LFW_DATA_HOME = "../../datasets"
RESOLUTIONS   = [0, 128, 64, 32]   # 0 = original (~125px)
MAX_PAIRS     = 150


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


def downscale(img_bgr, size):
    """Downscale then upscale back to original size (simulates low-res capture)."""
    if size == 0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    small = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def load_pairs():
    print("Loading LFW pairs...")
    pairs_data = fetch_lfw_pairs(
        subset="test",
        data_home=LFW_DATA_HOME,
        download_if_missing=True,
        resize=1.0,
        color=True,
    )
    images = pairs_data.pairs
    labels = pairs_data.target.tolist()
    n = min(MAX_PAIRS, len(labels))
    pairs = []
    for i in range(n):
        img1 = (images[i, 0] * 255).astype(np.uint8)
        img2 = (images[i, 1] * 255).astype(np.uint8)
        img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        pairs.append((img1_bgr, img2_bgr, labels[i]))
    print(f"  ✓ Loaded {n} pairs")
    return pairs


# ─── InsightFace ─────────────────────────────────────────────────────────────

def test_insightface_lowres(pairs):
    print("\n[InsightFace] Low-resolution test...")
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    rows = []
    for res in RESOLUTIONS:
        label_str = "original" if res == 0 else f"{res}x{res}"
        sims, labels, skipped = [], [], 0
        t0 = time.time()

        for img1, img2, label in pairs:
            d1 = downscale(img1, res)
            d2 = downscale(img2, res)
            f1 = app.get(d1)
            f2 = app.get(d2)
            if not f1 or not f2:
                skipped += 1
                continue
            e1 = f1[0].normed_embedding.astype("float32")
            e2 = f2[0].normed_embedding.astype("float32")
            sims.append(cosine_sim(e1, e2))
            labels.append(label)

        elapsed = time.time() - t0
        if sims:
            acc, thr = find_best_threshold(sims, labels)
            auc = round(roc_auc_score(labels, sims), 4) if len(set(labels)) > 1 else None
            det_rate = round(len(sims) / len(pairs), 3)
        else:
            acc, thr, auc, det_rate = 0, 0, None, 0

        row = {
            "framework": "InsightFace (ArcFace)",
            "resolution": label_str,
            "pairs_tested": len(sims),
            "pairs_skipped": skipped,
            "detection_rate": det_rate,
            "accuracy": acc,
            "auc_roc": auc,
            "time_sec": round(elapsed, 2),
        }
        rows.append(row)
        print(f"  [{label_str:>10}] acc={acc:.4f}, det={det_rate:.2%}, skipped={skipped}")

    return rows


# ─── DeepFace ────────────────────────────────────────────────────────────────

def test_deepface_lowres(pairs, model_name="ArcFace"):
    print(f"\n[DeepFace/{model_name}] Low-resolution test...")
    from deepface import DeepFace
    import tempfile, os

    rows = []
    for res in RESOLUTIONS:
        label_str = "original" if res == 0 else f"{res}x{res}"
        sims, labels, skipped = [], [], 0
        t0 = time.time()

        for img1, img2, label in pairs:
            d1 = downscale(img1, res)
            d2 = downscale(img2, res)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1, \
                 tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
                cv2.imwrite(f1.name, d1); cv2.imwrite(f2.name, d2)
                p1, p2 = f1.name, f2.name
            try:
                r = DeepFace.verify(p1, p2, model_name=model_name,
                                    detector_backend="opencv",
                                    enforce_detection=False, silent=True)
                sims.append(1 - r["distance"])
                labels.append(label)
            except Exception:
                skipped += 1
            finally:
                os.unlink(p1); os.unlink(p2)

        elapsed = time.time() - t0
        if sims:
            acc, thr = find_best_threshold(sims, labels)
            auc = round(roc_auc_score(labels, sims), 4) if len(set(labels)) > 1 else None
            det_rate = round(len(sims) / len(pairs), 3)
        else:
            acc, thr, auc, det_rate = 0, 0, None, 0

        row = {
            "framework": f"DeepFace ({model_name})",
            "resolution": label_str,
            "pairs_tested": len(sims),
            "pairs_skipped": skipped,
            "detection_rate": det_rate,
            "accuracy": acc,
            "auc_roc": auc,
            "time_sec": round(elapsed, 2),
        }
        rows.append(row)
        print(f"  [{label_str:>10}] acc={acc:.4f}, det={det_rate:.2%}, skipped={skipped}")

    return rows


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DAY 2 — Task 2: Low-Resolution Stress Test")
    print("=" * 60)

    pairs = load_pairs()

    all_rows = []
    all_rows.extend(test_insightface_lowres(pairs))
    all_rows.extend(test_deepface_lowres(pairs, model_name="ArcFace"))

    out = RESULTS_DIR / "lowres_results.json"
    with open(out, "w") as f:
        json.dump(all_rows, f, indent=2)

    print(f"\n✓ Saved → {out}")
    print("\nSummary:")
    print(f"{'Framework':<28} {'Resolution':>12} {'Acc':>7} {'Det%':>8}")
    print("-" * 60)
    for r in all_rows:
        print(f"{r['framework']:<28} {r['resolution']:>12} {r['accuracy']:>7.4f} {r['detection_rate']:>8.2%}")


if __name__ == "__main__":
    main()