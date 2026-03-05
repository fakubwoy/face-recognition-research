# day2/eval_occlusion.py
"""
Day 2 Task 3: Partial Occlusion Test
Synthetically occludes faces (eyes, lower face/mouth, glasses bar, random patch)
and measures accuracy drop per framework.

Output: results/occlusion_results.json
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

RESULTS_DIR   = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
LFW_DATA_HOME = "../../datasets"
MAX_PAIRS     = 100


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


# ─── Occlusion functions ──────────────────────────────────────────────────────

def occlude_none(img):
    return img.copy()


def occlude_lower_face(img):
    """Black bar over mouth/chin area (like a mask)."""
    h, w = img.shape[:2]
    out = img.copy()
    y1, y2 = int(h * 0.55), h
    out[y1:y2, :] = 0
    return out


def occlude_eyes(img):
    """Black bar over eye region (like sunglasses)."""
    h, w = img.shape[:2]
    out = img.copy()
    y1, y2 = int(h * 0.25), int(h * 0.50)
    out[y1:y2, :] = 0
    return out


def occlude_random_patch(img):
    """Random 40% patch anywhere on the face."""
    h, w = img.shape[:2]
    out = img.copy()
    ph, pw = int(h * 0.4), int(w * 0.4)
    y = np.random.randint(0, h - ph)
    x = np.random.randint(0, w - pw)
    out[y:y+ph, x:x+pw] = 0
    return out


def occlude_hat(img):
    """Top 30% of image blocked (hat/forehead)."""
    h, w = img.shape[:2]
    out = img.copy()
    out[:int(h * 0.3), :] = 0
    return out


OCCLUSIONS = {
    "none":          occlude_none,
    "lower_face":    occlude_lower_face,
    "eyes":          occlude_eyes,
    "random_patch":  occlude_random_patch,
    "hat_forehead":  occlude_hat,
}


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
    print(f"  ✓ {n} pairs loaded")
    return pairs


# ─── InsightFace ─────────────────────────────────────────────────────────────

def test_insightface_occlusion(pairs):
    print("\n[InsightFace] Occlusion test...")
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    rows = []
    for occ_name, occ_fn in OCCLUSIONS.items():
        sims, labels, skipped = [], [], 0
        np.random.seed(42)
        t0 = time.time()

        for img1, img2, label in pairs:
            # Apply occlusion to BOTH images
            d1 = occ_fn(img1)
            d2 = occ_fn(img2)
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
            "occlusion": occ_name,
            "pairs_tested": len(sims),
            "pairs_skipped": skipped,
            "detection_rate": det_rate,
            "accuracy": acc,
            "auc_roc": auc,
            "time_sec": round(elapsed, 2),
        }
        rows.append(row)
        print(f"  [{occ_name:<15}] acc={acc:.4f}, det={det_rate:.2%}, skip={skipped}")

    return rows


# ─── DeepFace ────────────────────────────────────────────────────────────────

def test_deepface_occlusion(pairs, model_name="ArcFace"):
    print(f"\n[DeepFace/{model_name}] Occlusion test...")
    from deepface import DeepFace
    import tempfile, os

    rows = []
    for occ_name, occ_fn in OCCLUSIONS.items():
        sims, labels, skipped = [], [], 0
        np.random.seed(42)
        t0 = time.time()

        for img1, img2, label in pairs:
            d1, d2 = occ_fn(img1), occ_fn(img2)
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
            "occlusion": occ_name,
            "pairs_tested": len(sims),
            "pairs_skipped": skipped,
            "detection_rate": det_rate,
            "accuracy": acc,
            "auc_roc": auc,
            "time_sec": round(elapsed, 2),
        }
        rows.append(row)
        print(f"  [{occ_name:<15}] acc={acc:.4f}, det={det_rate:.2%}, skip={skipped}")

    return rows


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DAY 2 — Task 3: Occlusion Robustness Test")
    print("=" * 60)
    pairs = load_pairs()

    all_rows = []
    all_rows.extend(test_insightface_occlusion(pairs))
    all_rows.extend(test_deepface_occlusion(pairs, "ArcFace"))

    out = RESULTS_DIR / "occlusion_results.json"
    with open(out, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\n✓ Saved → {out}")

    print(f"\n{'Framework':<28} {'Occlusion':>15} {'Acc':>7} {'Det%':>8}")
    print("-" * 62)
    for r in all_rows:
        print(f"{r['framework']:<28} {r['occlusion']:>15} {r['accuracy']:>7.4f} {r['detection_rate']:>8.2%}")


if __name__ == "__main__":
    main()