# day4/eval_final_benchmark.py
"""
Day 4 Task 3: Final Head-to-Head Benchmark
InsightFace vs DeepFace (Facenet512 & ArcFace) vs Dlib
on 500 balanced LFW pairs — accuracy, AUC, EER, TAR@FAR.

Fix applied: InsightFace receives disk-reloaded images (cv2.imwrite → cv2.imread)
to avoid the in-memory float32/BGR pipeline bug identified in Day 2 and Day 3.

Output: results/final_benchmark_results.json
"""

import json
import time
import os
import tempfile
import numpy as np
import cv2
from pathlib import Path
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR   = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
LFW_DATA_HOME = "../../datasets"

QUICK     = os.environ.get("DAY4_QUICK") == "1"
MAX_PAIRS = 100 if QUICK else 500


def cosine_sim(a, b):
    a, b = np.array(a, dtype="float32"), np.array(b, dtype="float32")
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def find_best_threshold(sims, labels):
    best_acc, best_t = 0, 0.5
    for t in np.arange(0.0, 1.01, 0.005):
        preds = (np.array(sims) >= t).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return round(best_acc, 4), round(float(best_t), 4)


def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = float(fpr[np.nanargmin(np.abs(fnr - fpr))])
    return round(eer, 4), round(float(eer_threshold), 4)


def tar_at_far(labels, scores, far_target=0.01):
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(fpr, far_target)
    if idx >= len(tpr):
        return 0.0
    return round(float(tpr[idx]), 4)


def load_balanced_pairs():
    print(f"Loading {MAX_PAIRS} balanced LFW pairs...")
    data = fetch_lfw_pairs(
        subset="test", data_home=LFW_DATA_HOME,
        download_if_missing=True, resize=1.0, color=True,
    )
    all_pairs = list(zip(data.pairs, data.target))
    same  = [(p, l) for p, l in all_pairs if l == 1]
    diff  = [(p, l) for p, l in all_pairs if l == 0]
    n = min(MAX_PAIRS // 2, len(same), len(diff))
    chosen = same[:n] + diff[:n]
    np.random.RandomState(42).shuffle(chosen)

    pairs = []
    for (imgs, label) in chosen:
        img1 = cv2.cvtColor((imgs[0] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor((imgs[1] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        pairs.append((img1, img2, int(label)))

    same_n = sum(1 for _, _, l in pairs if l == 1)
    print(f"  ✓ {len(pairs)} pairs — {same_n} same, {len(pairs)-same_n} diff")
    return pairs


def _write_reload(img_bgr, upscale=True):
    """
    Write image to a temp JPEG and reload it.

    Why upscale: fetch_lfw_pairs at resize=1.0 returns 125x94px images.
    InsightFace RetinaFace with det_size=(640,640) fails to find faces that
    small. Upscaling 2x to ~250x188px gives reliable detection.
    det_size=(320,320) is also set in eval_insightface() to match.

    Why mkstemp: on Linux, NamedTemporaryFile keeps the fd open, causing
    cv2.imwrite to produce a silent zero-byte file.
    """
    if upscale:
        h, w = img_bgr.shape[:2]
        img_bgr = cv2.resize(img_bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    cv2.imwrite(path, img_bgr)
    reloaded = cv2.imread(path)
    os.unlink(path)
    return reloaded


# ─── InsightFace ─────────────────────────────────────────────────────────────

def eval_insightface(pairs):
    print("\n[InsightFace ArcFace]...")
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(320, 320))  # 125x94 LFW images need smaller det_size

        sims, labels, skipped = [], [], 0
        t0 = time.time()
        for img1, img2, label in pairs:
            # Reload from disk to fix in-memory detection bug (confirmed fix in Day 3 Task 1)
            r1 = _write_reload(img1)
            r2 = _write_reload(img2)
            f1, f2 = app.get(r1), app.get(r2)
            if not f1 or not f2:
                skipped += 1
                continue
            e1 = f1[0].normed_embedding.astype("float32")
            e2 = f2[0].normed_embedding.astype("float32")
            sims.append(cosine_sim(e1, e2))
            labels.append(label)
        elapsed = time.time() - t0

        if not sims:
            return {"framework": "InsightFace (ArcFace w600k_r50)", "error": "all pairs skipped — detection failed"}

        acc, thr = find_best_threshold(sims, labels)
        auc  = round(roc_auc_score(labels, sims), 4)
        eer, _ = compute_eer(labels, sims)
        tar1   = tar_at_far(labels, sims, 0.01)
        tar01  = tar_at_far(labels, sims, 0.001)

        result = {
            "framework": "InsightFace (ArcFace w600k_r50)",
            "pairs_tested": len(sims), "pairs_skipped": skipped,
            "accuracy": acc, "best_threshold": thr,
            "auc_roc": auc, "eer": eer,
            "tar_at_far_1pct": tar1, "tar_at_far_01pct": tar01,
            "time_sec": round(elapsed, 2),
            "pairs_per_sec": round(len(sims) / elapsed, 2),
            "embedding_dim": 512,
        }
        print(f"  acc={acc:.4f}, AUC={auc:.4f}, EER={eer:.4f}, TAR@1%={tar1:.4f}")
        return result
    except Exception as e:
        print(f"  ✗ {e}")
        return {"framework": "InsightFace (ArcFace w600k_r50)", "error": str(e)}


# ─── DeepFace ────────────────────────────────────────────────────────────────

def eval_deepface(pairs, model_name):
    print(f"\n[DeepFace {model_name}]...")
    try:
        from deepface import DeepFace
        sims, labels, skipped = [], [], 0
        t0 = time.time()
        for img1, img2, label in pairs:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1, \
                 tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
                cv2.imwrite(f1.name, img1)
                cv2.imwrite(f2.name, img2)
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

        if not sims:
            return {"framework": f"DeepFace ({model_name})", "error": "all failed"}

        acc, thr = find_best_threshold(sims, labels)
        auc  = round(roc_auc_score(labels, sims), 4)
        eer, _ = compute_eer(labels, sims)
        tar1   = tar_at_far(labels, sims, 0.01)
        tar01  = tar_at_far(labels, sims, 0.001)

        emb_dims = {"ArcFace": 512, "Facenet512": 512, "VGG-Face": 2622, "OpenFace": 128}
        result = {
            "framework": f"DeepFace ({model_name})",
            "pairs_tested": len(sims), "pairs_skipped": skipped,
            "accuracy": acc, "best_threshold": thr,
            "auc_roc": auc, "eer": eer,
            "tar_at_far_1pct": tar1, "tar_at_far_01pct": tar01,
            "time_sec": round(elapsed, 2),
            "pairs_per_sec": round(len(sims) / elapsed, 2),
            "embedding_dim": emb_dims.get(model_name, "?"),
        }
        print(f"  acc={acc:.4f}, AUC={auc:.4f}, EER={eer:.4f}, TAR@1%={tar1:.4f}")
        return result
    except Exception as e:
        print(f"  ✗ {e}")
        return {"framework": f"DeepFace ({model_name})", "error": str(e)}


# ─── Dlib ────────────────────────────────────────────────────────────────────

def eval_dlib(pairs):
    print("\n[Dlib ResNet-128]...")
    try:
        import dlib
        models_dir = Path("../day2/models")
        shape_model = models_dir / "shape_predictor_68_face_landmarks.dat"
        face_model  = models_dir / "dlib_face_recognition_resnet_model_v1.dat"

        if not shape_model.exists():
            return {"framework": "Dlib (ResNet-128)", "error": "model files not found — run day2 first"}

        detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor(str(shape_model))
        facerec = dlib.face_recognition_model_v1(str(face_model))

        sims, labels, skipped = [], [], 0
        t0 = time.time()

        for img1_bgr, img2_bgr, label in pairs:
            try:
                embs = []
                for img_bgr in [img1_bgr, img2_bgr]:
                    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    dets = detector(rgb, 1)
                    if not dets:
                        embs.append(None); continue
                    shape = sp(rgb, dets[0])
                    embs.append(np.array(facerec.compute_face_descriptor(rgb, shape)))
                if embs[0] is None or embs[1] is None:
                    skipped += 1; continue
                sims.append(cosine_sim(embs[0], embs[1]))
                labels.append(label)
            except Exception:
                skipped += 1

        elapsed = time.time() - t0
        if not sims:
            return {"framework": "Dlib (ResNet-128)", "error": "all pairs skipped"}

        acc, thr = find_best_threshold(sims, labels)
        auc  = round(roc_auc_score(labels, sims), 4)
        eer, _ = compute_eer(labels, sims)
        tar1   = tar_at_far(labels, sims, 0.01)

        result = {
            "framework": "Dlib (ResNet-128)",
            "pairs_tested": len(sims), "pairs_skipped": skipped,
            "accuracy": acc, "best_threshold": thr,
            "auc_roc": auc, "eer": eer,
            "tar_at_far_1pct": tar1,
            "time_sec": round(elapsed, 2),
            "pairs_per_sec": round(len(sims) / elapsed, 2),
            "embedding_dim": 128,
        }
        print(f"  acc={acc:.4f}, AUC={auc:.4f}, EER={eer:.4f}")
        return result
    except ImportError:
        return {"framework": "Dlib (ResNet-128)", "error": "not installed"}
    except Exception as e:
        return {"framework": "Dlib", "error": str(e)}


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DAY 4 — Task 3: Final Head-to-Head Benchmark")
    print("=" * 60)

    pairs = load_balanced_pairs()
    results = []
    results.append(eval_insightface(pairs))
    results.append(eval_deepface(pairs, "ArcFace"))
    results.append(eval_deepface(pairs, "Facenet512"))
    results.append(eval_dlib(pairs))

    with open(RESULTS_DIR / "final_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'Framework':<35} {'Acc':>7} {'AUC':>7} {'EER':>7} {'TAR@1%':>8} {'p/s':>6}")
    print("-" * 72)
    for r in results:
        if "error" in r:
            print(f"{r['framework']:<35} ERROR: {r['error']}")
        else:
            print(f"{r['framework']:<35} {r['accuracy']:>7.4f} {r['auc_roc']:>7.4f} "
                  f"{r['eer']:>7.4f} {r['tar_at_far_1pct']:>8.4f} {r['pairs_per_sec']:>6.2f}")
    print(f"\n✓ Saved → results/final_benchmark_results.json")


if __name__ == "__main__":
    main()