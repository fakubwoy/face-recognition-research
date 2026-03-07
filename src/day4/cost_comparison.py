# day4/cost_comparison.py
"""
Day 4 Task 4: Cost Estimation at Scale
Computes storage, compute, and API costs at 5k / 50k / 500k images.

Output: results/cost_results.json
"""

import json
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

SCALES = [5_000, 50_000, 500_000]
AVG_IMAGE_SIZE_MB = 2.5          # average JPEG
AVG_FACES_PER_IMAGE = 1.2
EMBEDDING_DIM = 512
EMBEDDING_BYTES = EMBEDDING_DIM * 4   # float32

# Pricing (USD, as of 2025)
S3_STORAGE_PER_GB_MONTH   = 0.023
S3_PUT_PER_1000            = 0.005
S3_GET_PER_1000            = 0.0004
EC2_CPU_HOURLY             = 0.085   # t3.medium
EC2_GPU_HOURLY             = 0.526   # g4dn.xlarge (T4)
LAMBDA_PER_1M_REQUESTS     = 0.20
AWS_REKOGNITION_PER_1000   = 1.00
AZURE_FACE_PER_1000        = 1.00
PINECONE_SERVERLESS_PER_1M = 0.10   # writes + reads blended

def estimate_for_scale(n_images):
    n_faces = int(n_images * AVG_FACES_PER_IMAGE)
    image_storage_gb = (n_images * AVG_IMAGE_SIZE_MB) / 1024
    embedding_storage_mb = (n_faces * EMBEDDING_BYTES) / (1024 * 1024)
    embedding_storage_gb = embedding_storage_mb / 1024

    # Storage (monthly)
    s3_image_cost    = image_storage_gb * S3_STORAGE_PER_GB_MONTH
    s3_put_cost      = (n_images / 1000) * S3_PUT_PER_1000
    s3_total_monthly = s3_image_cost + s3_put_cost

    # Embedding generation (one-time)
    # InsightFace CPU: ~3.4 emb/sec → hours needed
    cpu_emb_per_sec = 3.4
    gpu_emb_per_sec = 80.0   # estimated with T4

    cpu_hours  = (n_faces / cpu_emb_per_sec) / 3600
    gpu_hours  = (n_faces / gpu_emb_per_sec) / 3600
    cpu_cost   = cpu_hours * EC2_CPU_HOURLY
    gpu_cost   = gpu_hours * EC2_GPU_HOURLY

    # Vector DB monthly
    faiss_cost      = 0.0   # self-hosted, part of EC2
    pinecone_cost   = (n_faces / 1_000_000) * PINECONE_SERVERLESS_PER_1M * 30  # daily queries

    # Proprietary API (one-time ingestion)
    aws_api_cost   = (n_faces / 1000) * AWS_REKOGNITION_PER_1000
    azure_api_cost = (n_faces / 1000) * AZURE_FACE_PER_1000

    return {
        "n_images": n_images,
        "n_faces_estimated": n_faces,
        "storage": {
            "image_storage_gb": round(image_storage_gb, 2),
            "embedding_storage_mb": round(embedding_storage_mb, 2),
            "s3_monthly_usd": round(s3_total_monthly, 2),
        },
        "embedding_generation_onetime": {
            "cpu_hours": round(cpu_hours, 1),
            "gpu_hours": round(gpu_hours, 2),
            "cpu_cost_usd": round(cpu_cost, 2),
            "gpu_cost_usd": round(gpu_cost, 2),
        },
        "vector_db_monthly": {
            "faiss_self_hosted_usd": faiss_cost,
            "pinecone_serverless_usd": round(pinecone_cost, 2),
        },
        "proprietary_api_ingestion_onetime": {
            "aws_rekognition_usd": round(aws_api_cost, 2),
            "azure_face_usd": round(azure_api_cost, 2),
        },
        "total_monthly_self_hosted_usd": round(s3_total_monthly + faiss_cost, 2),
        "total_monthly_managed_usd": round(s3_total_monthly + pinecone_cost, 2),
    }


def main():
    print("=" * 60)
    print("DAY 4 — Task 4: Cost Estimation at Scale")
    print("=" * 60)

    results = [estimate_for_scale(n) for n in SCALES]

    with open(RESULTS_DIR / "cost_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'Scale':>10} {'Images_GB':>10} {'Emb_MB':>8} "
          f"{'S3/mo':>8} {'CPU_hrs':>9} {'CPU_$':>8} {'GPU_$':>8}")
    print("-" * 68)
    for r in results:
        print(f"{r['n_images']:>10,} "
              f"{r['storage']['image_storage_gb']:>10.1f} "
              f"{r['storage']['embedding_storage_mb']:>8.1f} "
              f"${r['storage']['s3_monthly_usd']:>7.2f} "
              f"{r['embedding_generation_onetime']['cpu_hours']:>9.1f} "
              f"${r['embedding_generation_onetime']['cpu_cost_usd']:>7.2f} "
              f"${r['embedding_generation_onetime']['gpu_cost_usd']:>7.2f}")

    print(f"\n✓ Saved → results/cost_results.json")


if __name__ == "__main__":
    main()