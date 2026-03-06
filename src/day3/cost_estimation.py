# day3/cost_estimation.py
"""
Day 3 Task 6: Cost Estimation
Calculates storage + compute costs for the face recognition pipeline
at different scales: 5k, 50k, 500k images.

Output: results/cost_estimation.json
"""

import json
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ─── constants ────────────────────────────────────────────────────────────────

EMBEDDING_DIM        = 512         # ArcFace
BYTES_PER_FLOAT      = 4
FACES_PER_IMAGE      = 1.2        # average including multi-face images
AVG_IMAGE_SIZE_MB    = 0.5        # compressed JPEG average
METADATA_BYTES_FACE  = 200        # per face: path, label, bbox (JSON approx)


# ─── pricing (as of 2026) ─────────────────────────────────────────────────────

PRICING = {
    "aws_s3": {
        "storage_per_gb_month": 0.023,
        "get_per_1000": 0.0004,
        "put_per_1000": 0.005,
        "transfer_out_per_gb": 0.09,
    },
    "gcs_standard": {
        "storage_per_gb_month": 0.020,
        "get_per_1000": 0.0004,
        "put_per_1000": 0.005,
        "transfer_out_per_gb": 0.12,
    },
    "pinecone_serverless": {
        "storage_per_gb_month": 0.33,
        "per_1m_queries": 0.08,
    },
    "aws_rekognition": {
        "per_1000_images": 1.00,
        "free_tier_monthly": 5000,
    },
    "azure_face": {
        "per_1000_transactions": 1.00,
        "free_tier_monthly": 30000,
    },
}


# ─── compute cost at a given scale ───────────────────────────────────────────

def estimate_at_scale(n_images):
    n_faces      = int(n_images * FACES_PER_IMAGE)
    image_gb     = (n_images * AVG_IMAGE_SIZE_MB) / 1024
    embedding_gb = (n_faces * EMBEDDING_DIM * BYTES_PER_FLOAT) / 1e9
    metadata_gb  = (n_faces * METADATA_BYTES_FACE) / 1e9
    total_storage_gb = image_gb + embedding_gb + metadata_gb

    # Self-hosted stack costs
    self_hosted = {
        "images_gb": round(image_gb, 2),
        "embeddings_gb": round(embedding_gb * 1000, 2),   # in MB for readability
        "embeddings_gb_unit": "MB",
        "metadata_gb": round(metadata_gb * 1000, 2),
        "metadata_gb_unit": "MB",
        "aws_s3_image_storage_per_month": round(image_gb * PRICING["aws_s3"]["storage_per_gb_month"], 2),
        "faiss_index_ram_mb": round(n_faces * EMBEDDING_DIM * BYTES_PER_FLOAT / 1e6, 1),
        "faiss_index_disk_mb": round(n_faces * EMBEDDING_DIM * BYTES_PER_FLOAT / 1e6, 1),
        "monthly_cost_estimate": {
            "s3_storage": round(image_gb * PRICING["aws_s3"]["storage_per_gb_month"], 2),
            "compute_cpu_server": 0 if n_images <= 10000 else round(n_images / 100000 * 50, 2),
            "total_self_hosted": round(
                image_gb * PRICING["aws_s3"]["storage_per_gb_month"]
                + (0 if n_images <= 10000 else n_images / 100000 * 50),
                2,
            ),
        },
    }

    # Pinecone
    pinecone_storage_gb = embedding_gb
    pinecone_monthly = {
        "storage": round(pinecone_storage_gb * PRICING["pinecone_serverless"]["storage_per_gb_month"], 4),
        "queries_10k_month": round(10000 / 1e6 * PRICING["pinecone_serverless"]["per_1m_queries"], 4),
        "queries_1m_month": round(1e6 / 1e6 * PRICING["pinecone_serverless"]["per_1m_queries"], 2),
    }

    # AWS Rekognition (if used instead of self-hosted)
    aws_rek_ingest = max(
        0,
        ((n_images - PRICING["aws_rekognition"]["free_tier_monthly"])
         / 1000 * PRICING["aws_rekognition"]["per_1000_images"]),
    )

    # Azure Face
    azure_face_ingest = max(
        0,
        ((n_images - PRICING["azure_face"]["free_tier_monthly"])
         / 1000 * PRICING["azure_face"]["per_1000_transactions"]),
    )

    return {
        "n_images": n_images,
        "n_face_embeddings": n_faces,
        "storage_breakdown": {
            "images_gb": round(image_gb, 3),
            "embeddings_mb": round(embedding_gb * 1024, 2),
            "metadata_kb": round(metadata_gb * 1e6, 1),
        },
        "self_hosted": self_hosted,
        "pinecone_monthly_estimate": pinecone_monthly,
        "proprietary_api_ingest_cost": {
            "aws_rekognition_one_time": round(aws_rek_ingest, 2),
            "azure_face_one_time": round(azure_face_ingest, 2),
            "note": "One-time ingestion cost only. Neither API stores vectors for FAISS-style search.",
        },
    }


def build_gpu_vs_cpu():
    """Embedding throughput and cost comparison: CPU vs cloud GPU."""
    return {
        "cpu_insightface": {
            "throughput_embeddings_per_sec": 3.4,
            "time_to_embed_5k_images_min": round(5000 / 3.4 / 60, 1),
            "time_to_embed_50k_images_hr": round(50000 / 3.4 / 3600, 1),
            "hardware": "Intel Xeon 12-core (no GPU)",
            "monthly_cost": 0,
        },
        "aws_g4dn_xlarge_gpu": {
            "throughput_estimate_embeddings_per_sec": 120,
            "time_to_embed_5k_images_min": round(5000 / 120 / 60, 2),
            "time_to_embed_50k_images_min": round(50000 / 120 / 60, 1),
            "hardware": "NVIDIA T4 GPU",
            "spot_price_per_hr": 0.16,
            "cost_to_embed_5k": round(5000 / 120 / 3600 * 0.16, 4),
            "cost_to_embed_50k": round(50000 / 120 / 3600 * 0.16, 3),
        },
        "recommendation": (
            "For the initial 5k–50k image dataset, CPU is sufficient. "
            "A one-time 50k embedding run takes ~4 hours on CPU (free) vs ~7 minutes on "
            "a T4 spot instance (~$0.002). GPU becomes cost-effective only for continuous "
            "bulk ingestion at 100k+ images/month."
        ),
    }


def build_monthly_recurring(n_images):
    """Monthly recurring costs at steady state."""
    image_gb = (n_images * AVG_IMAGE_SIZE_MB) / 1024
    s3_cost  = round(image_gb * PRICING["aws_s3"]["storage_per_gb_month"], 2)

    return {
        "aws_s3_storage": s3_cost,
        "faiss_server_ram_gb": round(n_images * FACES_PER_IMAGE * EMBEDDING_DIM * 4 / 1e9, 2),
        "estimated_vps_cost": (
            "$0 (dev laptop / WSL2)"
            if n_images <= 20000
            else "$10–40/month (4GB RAM VPS)"
            if n_images <= 200000
            else "$80–200/month (16GB RAM dedicated)"
        ),
        "total_estimate": s3_cost,
    }


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DAY 3 — Task 6: Cost Estimation")
    print("=" * 60)

    scales = [5_000, 50_000, 500_000]
    scale_results = {str(s): estimate_at_scale(s) for s in scales}

    output = {
        "generated": datetime.now().isoformat(),
        "assumptions": {
            "avg_image_size_mb": AVG_IMAGE_SIZE_MB,
            "avg_faces_per_image": FACES_PER_IMAGE,
            "embedding_dim": EMBEDDING_DIM,
            "embedding_dtype": "float32 (4 bytes/value)",
        },
        "pricing_references": {
            "aws_s3": "$0.023/GB/month",
            "pinecone_serverless": "$0.33/GB/month + $0.08/1M queries",
            "aws_rekognition": "$1.00/1000 images",
            "azure_face": "$1.00/1000 transactions",
        },
        "scales": scale_results,
        "gpu_vs_cpu": build_gpu_vs_cpu(),
        "monthly_recurring": {
            str(s): build_monthly_recurring(s) for s in scales
        },
        "recommendation": {
            "storage": "AWS S3 for images (cheapest, most reliable). FAISS index on local disk or EBS volume.",
            "vector_db": "FAISS IndexFlatIP up to ~500k vectors; switch to IVFFlat at 1M+.",
            "embedding_compute": "CPU sufficient for initial build; add GPU if doing real-time ingestion > 1k images/day.",
            "total_5k_monthly": f"~${round((5000 * AVG_IMAGE_SIZE_MB / 1024) * PRICING['aws_s3']['storage_per_gb_month'], 2)}/month (S3 only)",
        },
    }

    out = RESULTS_DIR / "cost_estimation.json"
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Saved → {out}")

    print("\nCost summary:")
    for scale, data in scale_results.items():
        n = int(scale)
        monthly = data["self_hosted"]["monthly_cost_estimate"]["total_self_hosted"]
        faiss_mb = data["self_hosted"]["faiss_index_ram_mb"]
        print(f"  {n:>8,} images — S3 storage: ${data['self_hosted']['monthly_cost_estimate']['s3_storage']:.2f}/mo, "
              f"FAISS index: {faiss_mb:.1f} MB, total est: ${monthly:.2f}/mo")


if __name__ == "__main__":
    main()