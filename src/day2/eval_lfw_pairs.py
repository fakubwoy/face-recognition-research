# day2/eval_lfw_pairs.py
"""
Day 2 Task 1: Formal LFW Pairs Evaluation
Tests InsightFace (ArcFace), DeepFace backends, and Dlib
on the standard LFW 6,000-pair verification benchmark.

Output: results/lfw_pairs_results.json
"""

import json
import time
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve
)
import cv2
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

LFW_DATA_HOME = "../../datasets"   # adjust if your LFW is elsewhere


# ─── helpers ────────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    a, b = np.array(a, dtype="float32"), np.array(b, dtype="float32")
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def find_best_threshold(sims, labels):
    """Sweep thresholds, return (best_acc, best_threshold)."""
    best_acc, best_t = 0, 0.5
    for t in np.arange(0.1, 1.0, 0.01):
        preds = (np.array(sims) >= t).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return round(best_acc, 4), round(float(best_t), 2)


def load_lfw_pairs_images(max_pairs=300):
    """
    Returns list of (img1_bgr, img2_bgr, label) from sklearn LFW pairs.
    label=1 means same person, 0 means different.
    """
    print(f"Loading LFW pairs (max {max_pairs})...")
    pairs = fetch_lfw_pairs(
        subset="test",
        data_home=LFW_DATA_HOME,
        download_if_missing=True,
        resize=1.0,
        color=True,
    )
    images = pairs.pairs          # shape: (N, 2, H, W, 3)  float 0-1
    labels = pairs.target.tolist() # 1=same, 0=diff

    n = min(max_pairs, len(labels))
    pairs_out = []
    for i in range(n):
        img1 = (images[i, 0] * 255).astype(np.uint8)
        img2 = (images[i, 1] * 255).astype(np.uint8)
        img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        pairs_out.append((img1_bgr, img2_bgr, labels[i]))

    same = sum(labels[:n])
    print(f"  Loaded {n} pairs — {same} same-person, {n-same} different")
    return pairs_out


# ─── InsightFace ArcFace ─────────────────────────────────────────────────────

def eval_insightface(pairs):
    print("\n[1/3] Evaluating InsightFace (ArcFace)...")
    try:
        from insightface.app import FaceAnalysis
        import faiss

        app = FaceAnalysis(providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))

        sims, labels, skipped = [], [], 0
        t0 = time.time()

        for img1, img2, label in pairs:
            f1 = app.get(img1)
            f2 = app.get(img2)
            if not f1 or not f2:
                skipped += 1
                continue
            e1 = f1[0].normed_embedding.astype("float32")
            e2 = f2[0].normed_embedding.astype("float32")
            sims.append(cosine_sim(e1, e2))
            labels.append(label)

        elapsed = time.time() - t0
        acc, threshold = find_best_threshold(sims, labels)
        auc = round(roc_auc_score(labels, sims), 4)

        result = {
            "framework": "InsightFace (ArcFace w600k_r50)",
            "pairs_tested": len(labels),
            "pairs_skipped": skipped,
            "accuracy": acc,
            "best_threshold": threshold,
            "auc_roc": auc,
            "time_sec": round(elapsed, 2),
            "pairs_per_sec": round(len(labels) / elapsed, 2),
        }
        print(f"  ✓ acc={acc:.4f}, AUC={auc:.4f}, time={elapsed:.1f}s, skipped={skipped}")
        return result

    except Exception as e:
        print(f"  ✗ InsightFace failed: {e}")
        return {"framework": "InsightFace (ArcFace)", "error": str(e)}


# ─── DeepFace backends ───────────────────────────────────────────────────────

def eval_deepface_backend(pairs, model_name, detector="opencv"):
    print(f"\n  DeepFace [{model_name}]...")
    try:
        from deepface import DeepFace
        import tempfile, os

        sims, labels, skipped = [], [], 0
        t0 = time.time()

        for img1, img2, label in pairs:
            # DeepFace needs file paths, save temp
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1, \
                 tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
                cv2.imwrite(f1.name, img1)
                cv2.imwrite(f2.name, img2)
                p1, p2 = f1.name, f2.name

            try:
                r = DeepFace.verify(
                    p1, p2,
                    model_name=model_name,
                    detector_backend=detector,
                    enforce_detection=False,
                    silent=True,
                )
                sims.append(1 - r["distance"])   # convert distance → similarity
                labels.append(label)
            except Exception:
                skipped += 1
            finally:
                os.unlink(p1); os.unlink(p2)

        elapsed = time.time() - t0
        if not sims:
            return {"framework": f"DeepFace ({model_name})", "error": "all pairs failed"}

        acc, threshold = find_best_threshold(sims, labels)
        auc = round(roc_auc_score(labels, sims), 4)

        result = {
            "framework": f"DeepFace ({model_name})",
            "pairs_tested": len(labels),
            "pairs_skipped": skipped,
            "accuracy": acc,
            "best_threshold": threshold,
            "auc_roc": auc,
            "time_sec": round(elapsed, 2),
            "pairs_per_sec": round(len(labels) / elapsed, 2),
        }
        print(f"    ✓ acc={acc:.4f}, AUC={auc:.4f}, time={elapsed:.1f}s")
        return result

    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return {"framework": f"DeepFace ({model_name})", "error": str(e)}


def eval_deepface(pairs):
    print("\n[2/3] Evaluating DeepFace backends...")
    results = []
    for model in ["ArcFace", "Facenet512", "VGG-Face"]:
        results.append(eval_deepface_backend(pairs, model))
    return results


# ─── Dlib ────────────────────────────────────────────────────────────────────

def eval_dlib(pairs):
    print("\n[3/3] Evaluating Dlib (ResNet)...")
    try:
        import dlib
        from pathlib import Path

        # Dlib needs model files — download if missing
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        shape_model = models_dir / "shape_predictor_68_face_landmarks.dat"
        face_model  = models_dir / "dlib_face_recognition_resnet_model_v1.dat"

        if not shape_model.exists() or not face_model.exists():
            print("  Downloading Dlib models (one-time ~100MB)...")
            import urllib.request, bz2
            for url, dest in [
                ("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2", shape_model),
                ("http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2", face_model),
            ]:
                bz_path = str(dest) + ".bz2"
                urllib.request.urlretrieve(url, bz_path)
                with bz2.open(bz_path) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                Path(bz_path).unlink()
            print("  ✓ Dlib models downloaded")

        detector   = dlib.get_frontal_face_detector()
        sp         = dlib.shape_predictor(str(shape_model))
        facerec    = dlib.face_recognition_model_v1(str(face_model))

        sims, labels, skipped = [], [], 0
        t0 = time.time()

        for img1_bgr, img2_bgr, label in pairs:
            try:
                embs = []
                for img_bgr in [img1_bgr, img2_bgr]:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    dets = detector(img_rgb, 1)
                    if not dets:
                        embs.append(None)
                        continue
                    shape = sp(img_rgb, dets[0])
                    emb   = np.array(facerec.compute_face_descriptor(img_rgb, shape))
                    embs.append(emb)

                if embs[0] is None or embs[1] is None:
                    skipped += 1
                    continue

                sims.append(cosine_sim(embs[0], embs[1]))
                labels.append(label)
            except Exception:
                skipped += 1

        elapsed = time.time() - t0
        acc, threshold = find_best_threshold(sims, labels)
        auc = round(roc_auc_score(labels, sims), 4)

        result = {
            "framework": "Dlib (ResNet-128)",
            "pairs_tested": len(labels),
            "pairs_skipped": skipped,
            "accuracy": acc,
            "best_threshold": threshold,
            "auc_roc": auc,
            "time_sec": round(elapsed, 2),
            "pairs_per_sec": round(len(labels) / elapsed, 2),
        }
        print(f"  ✓ acc={acc:.4f}, AUC={auc:.4f}, time={elapsed:.1f}s, skipped={skipped}")
        return result

    except ImportError:
        print("  ⚠ Dlib not installed — pip install dlib")
        return {"framework": "Dlib (ResNet-128)", "error": "not installed"}
    except Exception as e:
        print(f"  ✗ Dlib failed: {e}")
        return {"framework": "Dlib (ResNet-128)", "error": str(e)}


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DAY 2 — Task 1: LFW Pairs Formal Evaluation")
    print("=" * 60)

    MAX_PAIRS = 300   # increase to 1000+ for full benchmark (slow)
    pairs = load_lfw_pairs_images(max_pairs=MAX_PAIRS)

    all_results = []

    all_results.append(eval_insightface(pairs))
    all_results.extend(eval_deepface(pairs))
    all_results.append(eval_dlib(pairs))

    # Save
    out = RESULTS_DIR / "lfw_pairs_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"✓ Results saved → {out}")
    print(f"{'='*60}\n")

    # Quick summary table
    print(f"{'Framework':<35} {'Acc':>7} {'AUC':>7} {'Pairs/s':>9}")
    print("-" * 62)
    for r in all_results:
        if "error" in r:
            print(f"{r['framework']:<35} {'ERROR':>7}")
        else:
            print(f"{r['framework']:<35} {r['accuracy']:>7.4f} {r['auc_roc']:>7.4f} {r['pairs_per_sec']:>9.2f}")


if __name__ == "__main__":
    main()