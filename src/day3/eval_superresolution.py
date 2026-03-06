# day3/eval_superresolution.py
"""
Day 3 Task 4: Super-Resolution Preprocessing for Low-Res Faces
Evaluates whether upscaling 32x32 / 64x64 CCTV faces before embedding
improves recognition accuracy.

Methods tested:
  - Baseline: OpenCV bicubic (current approach)
  - OpenCV LANCZOS4
  - OpenCV EDSR (DNN super-resolution, if available)
  - Simple sharpening post-upscale

Output: results/superresolution_results.json
"""

import json
import time
import os
import tempfile
import shutil
import numpy as np
import cv2
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR   = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
LFW_DATA_HOME = "../../datasets"
MAX_PAIRS     = int(os.environ.get("DAY3_MAX_PAIRS", 150))
RESOLUTIONS   = [64, 32]   # Focus on the two hardest cases from Day 2


# ─── helpers ─────────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    a, b = np.array(a, dtype="float32"), np.array(b, dtype="float32")
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def find_best_threshold(sims, labels):
    best_acc, best_t = 0, 0.5
    for t in np.arange(0.05, 1.0, 0.01):
        preds = (np.array(sims) >= t).astype(int)
        acc   = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return round(best_acc, 4), round(float(best_t), 3)


def load_pairs():
    from sklearn.datasets import fetch_lfw_pairs
    data = fetch_lfw_pairs(
        subset="test", data_home=LFW_DATA_HOME,
        download_if_missing=True, resize=1.0, color=True,
    )
    n = min(MAX_PAIRS, len(data.target))
    pairs = []
    for i in range(n):
        img1_bgr = cv2.cvtColor((data.pairs[i, 0] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        img2_bgr = cv2.cvtColor((data.pairs[i, 1] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        pairs.append((img1_bgr, img2_bgr, int(data.target[i])))
    same = sum(p[2] for p in pairs)
    print(f"  ✓ {n} pairs — {same} same, {n-same} different")
    return pairs


# ─── upscaling methods ────────────────────────────────────────────────────────

def upscale_bicubic(img, target_size):
    h, w = img.shape[:2]
    small = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def upscale_lanczos(img, target_size):
    h, w = img.shape[:2]
    small = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LANCZOS4)


def upscale_bicubic_sharpen(img, target_size):
    """Bicubic + mild unsharp mask to recover edge detail."""
    upscaled = upscale_bicubic(img, target_size)
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(upscaled, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def upscale_edsr(img, target_size, sr_model=None):
    """
    Uses OpenCV's DNN super-resolution (EDSR x4).
    Requires: pip install opencv-contrib-python
    Model: EDSR_x4.pb from https://github.com/Saafke/EDSR_Tensorflow
    Falls back to bicubic if model unavailable.
    """
    if sr_model is None:
        return upscale_bicubic(img, target_size)
    try:
        h, w = img.shape[:2]
        small = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        upscaled = sr_model.upsample(small)
        return cv2.resize(upscaled, (w, h), interpolation=cv2.INTER_AREA)
    except Exception:
        return upscale_bicubic(img, target_size)


UPSCALE_METHODS = {
    "bicubic":          upscale_bicubic,
    "lanczos4":         upscale_lanczos,
    "bicubic+sharpen":  upscale_bicubic_sharpen,
}


def load_edsr_model():
    """Try to load EDSR x4 model. Return None if unavailable."""
    model_path = Path("models/EDSR_x4.pb")
    if not model_path.exists():
        print("  ℹ  EDSR_x4.pb not found in models/ — skipping EDSR method")
        print("     Download from: https://github.com/Saafke/EDSR_Tensorflow/tree/master/models")
        return None
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(str(model_path))
        sr.setModel("edsr", 4)
        UPSCALE_METHODS["edsr_x4"] = lambda img, sz: upscale_edsr(img, sz, sr)
        print("  ✓ EDSR x4 model loaded")
        return sr
    except AttributeError:
        print("  ℹ  cv2.dnn_superres not available — install opencv-contrib-python")
        return None


# ─── DeepFace evaluation ──────────────────────────────────────────────────────

def run_deepface_on_pairs(pairs_upscaled, tmp_dir):
    from deepface import DeepFace
    sims, labels, skipped = [], [], 0
    for img1, img2, label in pairs_upscaled:
        p1 = str(tmp_dir / f"a_{len(sims)}.jpg")
        p2 = str(tmp_dir / f"b_{len(sims)}.jpg")
        cv2.imwrite(p1, img1)
        cv2.imwrite(p2, img2)
        try:
            r = DeepFace.verify(p1, p2, model_name="ArcFace",
                                detector_backend="opencv",
                                enforce_detection=False, silent=True)
            sims.append(1 - r["distance"])
            labels.append(label)
        except Exception:
            skipped += 1
    return sims, labels, skipped


def eval_sr_method(pairs, method_name, upscale_fn, resolution, tmp_dir):
    t0 = time.time()
    upscaled_pairs = [
        (upscale_fn(img1, resolution), upscale_fn(img2, resolution), label)
        for img1, img2, label in pairs
    ]
    sims, labels, skipped = run_deepface_on_pairs(upscaled_pairs, tmp_dir)
    elapsed = time.time() - t0

    if not sims:
        return {
            "method": method_name,
            "resolution": f"{resolution}x{resolution}",
            "error": "all pairs failed",
        }

    acc, thr = find_best_threshold(sims, labels)
    auc = round(roc_auc_score(labels, sims), 4) if len(set(labels)) > 1 else None

    return {
        "method": method_name,
        "resolution": f"{resolution}x{resolution}",
        "pairs_tested": len(sims),
        "pairs_skipped": skipped,
        "accuracy": acc,
        "best_threshold": thr,
        "auc_roc": auc,
        "time_sec": round(elapsed, 2),
    }


# ─── Baseline (no degradation) ───────────────────────────────────────────────

def eval_baseline(pairs, tmp_dir):
    print("\n[Baseline — no degradation]")
    sims, labels, skipped = run_deepface_on_pairs(pairs, tmp_dir)
    if not sims:
        return {"method": "baseline_original", "error": "all failed"}
    acc, thr = find_best_threshold(sims, labels)
    auc = round(roc_auc_score(labels, sims), 4) if len(set(labels)) > 1 else None
    result = {
        "method": "original_no_degradation",
        "resolution": "original",
        "pairs_tested": len(sims),
        "accuracy": acc,
        "auc_roc": auc,
    }
    print(f"  ✓ acc={acc:.4f}, AUC={auc}")
    return result


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DAY 3 — Task 4: Super-Resolution Preprocessing Evaluation")
    print("=" * 60)

    pairs = load_pairs()
    tmp_dir = Path(tempfile.mkdtemp(prefix="sr_eval_"))
    load_edsr_model()

    results = []
    try:
        results.append(eval_baseline(pairs, tmp_dir))

        for resolution in RESOLUTIONS:
            print(f"\n── Resolution: {resolution}x{resolution} ──")
            for method_name, upscale_fn in UPSCALE_METHODS.items():
                print(f"  [{method_name}]")
                r = eval_sr_method(pairs, method_name, upscale_fn, resolution, tmp_dir)
                results.append(r)
                if "error" not in r:
                    print(f"    acc={r['accuracy']:.4f}, AUC={r.get('auc_roc')}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    out = RESULTS_DIR / "superresolution_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved → {out}")

    print(f"\n{'Method':<25} {'Resolution':>12} {'Accuracy':>10} {'AUC':>8}")
    print("-" * 60)
    for r in results:
        if "error" in r:
            print(f"{r['method']:<25} {r.get('resolution','—'):>12}  ERROR")
        else:
            auc = f"{r['auc_roc']:.4f}" if r.get("auc_roc") else "N/A"
            print(f"{r['method']:<25} {r['resolution']:>12} {r['accuracy']:>10.4f} {auc:>8}")


if __name__ == "__main__":
    main()