# day2/eval_proprietary_apis.py
"""
Day 2 Task 5: Proprietary API Evaluation — AWS Rekognition & Azure Face API
Set your credentials in environment variables before running:

  AWS:
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_DEFAULT_REGION=us-east-1

  Azure:
    export AZURE_FACE_KEY=...
    export AZURE_FACE_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com

If credentials are missing, the script logs placeholders so the report
can still be generated with manual fill-in.

Output: results/proprietary_api_results.json
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
MAX_PAIRS     = 50   # keep low — API calls cost money


def find_best_threshold(sims, labels):
    best_acc, best_t = 0, 0.5
    for t in np.arange(0.1, 1.0, 0.01):
        preds = (np.array(sims) >= t).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return round(best_acc, 4), round(float(best_t), 2)


def load_pairs():
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
    return pairs


# ─── AWS Rekognition ─────────────────────────────────────────────────────────

def eval_aws_rekognition(pairs):
    print("\n[AWS Rekognition] Testing...")
    key    = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    if not key or not secret:
        print("  ⚠ AWS credentials not set — returning placeholder")
        return {
            "service": "AWS Rekognition",
            "status": "credentials_missing",
            "note": "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env vars",
            "pricing": "$1.00 per 1,000 images (CompareFaces)",
            "free_tier": "5,000 images/month for 12 months",
        }

    try:
        import boto3
        client = boto3.client("rekognition", region_name=region)

        sims, labels, skipped = [], [], 0
        total_cost_usd = 0
        t0 = time.time()

        for img1, img2, label in pairs:
            _, buf1 = cv2.imencode(".jpg", img1)
            _, buf2 = cv2.imencode(".jpg", img2)
            b1 = buf1.tobytes()
            b2 = buf2.tobytes()

            try:
                r = client.compare_faces(
                    SourceImage={"Bytes": b1},
                    TargetImage={"Bytes": b2},
                    SimilarityThreshold=0,
                )
                matches = r.get("FaceMatches", [])
                sim = matches[0]["Similarity"] / 100.0 if matches else 0.0
                sims.append(sim)
                labels.append(label)
                total_cost_usd += 0.001   # $1/1000 calls
            except Exception as e:
                skipped += 1

        elapsed = time.time() - t0
        acc, thr = find_best_threshold(sims, labels)
        auc = round(roc_auc_score(labels, sims), 4) if len(set(labels)) > 1 else None

        return {
            "service": "AWS Rekognition",
            "status": "tested",
            "pairs_tested": len(sims),
            "pairs_skipped": skipped,
            "accuracy": acc,
            "best_threshold": thr,
            "auc_roc": auc,
            "time_sec": round(elapsed, 2),
            "estimated_cost_usd": round(total_cost_usd, 4),
            "pricing": "$1.00 per 1,000 images",
            "free_tier": "5,000 images/month for 12 months",
        }

    except ImportError:
        return {"service": "AWS Rekognition", "status": "boto3_not_installed",
                "note": "pip install boto3"}
    except Exception as e:
        return {"service": "AWS Rekognition", "status": "error", "error": str(e)}


# ─── Azure Face API ───────────────────────────────────────────────────────────

def eval_azure_face(pairs):
    print("\n[Azure Face API] Testing...")
    key      = os.environ.get("AZURE_FACE_KEY")
    endpoint = os.environ.get("AZURE_FACE_ENDPOINT")

    if not key or not endpoint:
        print("  ⚠ Azure credentials not set — returning placeholder")
        return {
            "service": "Azure Face API",
            "status": "credentials_missing",
            "note": "Set AZURE_FACE_KEY and AZURE_FACE_ENDPOINT env vars",
            "pricing": "$0.001 per transaction (after free tier)",
            "free_tier": "30,000 transactions/month",
        }

    try:
        import requests as req

        detect_url = endpoint.rstrip("/") + "/face/v1.0/detect"
        verify_url = endpoint.rstrip("/") + "/face/v1.0/verify"
        headers = {
            "Ocp-Apim-Subscription-Key": key,
            "Content-Type": "application/octet-stream",
        }

        sims, labels, skipped = [], [], 0
        t0 = time.time()

        for img1, img2, label in pairs:
            try:
                fids = []
                for img_bgr in [img1, img2]:
                    _, buf = cv2.imencode(".jpg", img_bgr)
                    resp = req.post(
                        detect_url,
                        params={"returnFaceId": "true", "detectionModel": "detection_01"},
                        headers=headers,
                        data=buf.tobytes(),
                        timeout=10,
                    )
                    faces = resp.json()
                    fids.append(faces[0]["faceId"] if faces else None)

                if not fids[0] or not fids[1]:
                    skipped += 1
                    continue

                v_resp = req.post(
                    verify_url,
                    headers={**headers, "Content-Type": "application/json"},
                    json={"faceId1": fids[0], "faceId2": fids[1]},
                    timeout=10,
                ).json()

                sim = v_resp.get("confidence", 0)
                sims.append(sim)
                labels.append(label)

            except Exception:
                skipped += 1

        elapsed = time.time() - t0
        acc, thr = find_best_threshold(sims, labels)
        auc = round(roc_auc_score(labels, sims), 4) if len(set(labels)) > 1 else None

        return {
            "service": "Azure Face API",
            "status": "tested",
            "pairs_tested": len(sims),
            "pairs_skipped": skipped,
            "accuracy": acc,
            "best_threshold": thr,
            "auc_roc": auc,
            "time_sec": round(elapsed, 2),
            "pricing": "$0.001 per transaction",
            "free_tier": "30,000 transactions/month",
        }

    except Exception as e:
        return {"service": "Azure Face API", "status": "error", "error": str(e)}


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DAY 2 — Task 5: Proprietary API Evaluation")
    print("=" * 60)
    pairs = load_pairs()

    results = [
        eval_aws_rekognition(pairs),
        eval_azure_face(pairs),
    ]

    out = RESULTS_DIR / "proprietary_api_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved → {out}")

    for r in results:
        print(f"\n{r['service']}:")
        for k, v in r.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()