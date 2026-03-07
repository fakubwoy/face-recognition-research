# day4/generate_recommendation.py
"""
Day 4 Task 5: Final Recommendation Report Generator
Reads all Day 2–4 results and writes a structured JSON recommendation
that Claude will use to generate the final research document.

Output: results/recommendation.json
"""

import json
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load(name):
    p = RESULTS_DIR / name
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def pick_best(benchmark_results, key="accuracy"):
    """Return the framework with highest value for key."""
    valid = [r for r in benchmark_results if key in r and "error" not in r]
    if not valid:
        return "unknown"
    return max(valid, key=lambda r: r[key])["framework"]


def main():
    print("=" * 60)
    print("DAY 4 — Task 5: Generating Final Recommendation")
    print("=" * 60)

    final_bench  = load("final_benchmark_results.json") or []
    retrieval    = load("retrieval_results.json") or {}
    clustering   = load("clustering_results.json") or []
    cost         = load("cost_results.json") or []
    cost_5k      = next((c for c in cost if c["n_images"] == 5_000), {})
    cost_50k     = next((c for c in cost if c["n_images"] == 50_000), {})
    cost_500k    = next((c for c in cost if c["n_images"] == 500_000), {})

    best_acc_framework = pick_best(final_bench, "accuracy")
    best_auc_framework = pick_best(final_bench, "auc_roc")

    # Best clustering
    best_cluster = None
    if clustering and isinstance(clustering, list):
        valid_cl = [c for c in clustering if "nmi" in c]
        if valid_cl:
            best_cluster = max(valid_cl, key=lambda x: x["nmi"])

    recommendation = {
        "summary": {
            "recommended_detection": "InsightFace (RetinaFace)",
            "recommended_embedding": "InsightFace (ArcFace w600k_r50)",
            "recommended_vector_db": "FAISS IndexFlatIP (up to 100k), HNSW beyond",
            "recommended_image_storage": "AWS S3",
            "recommended_metadata_db": "PostgreSQL",
            "recommended_api": "FastAPI + uvicorn (ONNX CPU runtime)",
            "recommended_queue": "Celery + Redis",
        },
        "key_findings": {
            "best_accuracy_framework": best_acc_framework,
            "best_auc_framework": best_auc_framework,
            "retrieval_rank1": retrieval.get("rank1_accuracy"),
            "retrieval_rank5": retrieval.get("rank5_accuracy"),
            "retrieval_map": retrieval.get("mAP"),
            "best_clustering_method": best_cluster.get("method") if best_cluster else None,
            "best_clustering_nmi": best_cluster.get("nmi") if best_cluster else None,
        },
        "cost_estimates": {
            "5k_images": {
                "monthly_self_hosted_usd": cost_5k.get("total_monthly_self_hosted_usd"),
                "monthly_managed_usd": cost_5k.get("total_monthly_managed_usd"),
                "one_time_gpu_ingestion_usd": cost_5k.get("embedding_generation_onetime", {}).get("gpu_cost_usd"),
            },
            "50k_images": {
                "monthly_self_hosted_usd": cost_50k.get("total_monthly_self_hosted_usd"),
                "monthly_managed_usd": cost_50k.get("total_monthly_managed_usd"),
                "one_time_gpu_ingestion_usd": cost_50k.get("embedding_generation_onetime", {}).get("gpu_cost_usd"),
            },
            "500k_images": {
                "monthly_self_hosted_usd": cost_500k.get("total_monthly_self_hosted_usd"),
                "monthly_managed_usd": cost_500k.get("total_monthly_managed_usd"),
                "one_time_gpu_ingestion_usd": cost_500k.get("embedding_generation_onetime", {}).get("gpu_cost_usd"),
            },
        },
        "comparison_table": [
            {
                "solution": r["framework"],
                "accuracy": r.get("accuracy"),
                "auc": r.get("auc_roc"),
                "eer": r.get("eer"),
                "tar_at_far_1pct": r.get("tar_at_far_1pct"),
                "speed_pairs_per_sec": r.get("pairs_per_sec"),
                "cost": "Free" if "InsightFace" in r["framework"] or "Dlib" in r["framework"] or "DeepFace" in r["framework"] else "Paid",
                "embedding_dim": r.get("embedding_dim"),
            }
            for r in final_bench if "error" not in r
        ],
        "future_improvements": [
            "GPU inference (10–20× speedup on embedding generation)",
            "Real-time clustering for auto-tagging unknown faces",
            "Super-resolution preprocessing for CCTV/low-res input",
            "HNSW index migration once dataset exceeds 100k vectors",
            "Celery async queue for bulk ingestion pipeline",
            "Anti-spoofing (liveness detection) layer",
            "Demographic parity evaluation (bias audit)",
        ],
        "architecture_notes": {
            "faiss_latency_p50_ms": 2.36,
            "faiss_latency_p99_ms": 3.79,
            "api_prototype": "FastAPI endpoint live at /search",
            "index_size_16k_vectors_mb": 32,
        }
    }

    out = RESULTS_DIR / "recommendation.json"
    with open(out, "w") as f:
        json.dump(recommendation, f, indent=2)

    print("\n📋 Recommendation Summary:")
    print(f"  Detection  : {recommendation['summary']['recommended_detection']}")
    print(f"  Embedding  : {recommendation['summary']['recommended_embedding']}")
    print(f"  Vector DB  : {recommendation['summary']['recommended_vector_db']}")
    print(f"  Storage    : {recommendation['summary']['recommended_image_storage']}")
    print(f"  Best acc   : {recommendation['key_findings']['best_accuracy_framework']}")
    print(f"  Rank-1     : {recommendation['key_findings']['retrieval_rank1']}")
    print(f"  mAP        : {recommendation['key_findings']['retrieval_map']}")
    print(f"\n✓ Saved → results/recommendation.json")


if __name__ == "__main__":
    main()