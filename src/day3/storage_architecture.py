# day3/storage_architecture.py
"""
Day 3 Task 7: Storage Architecture Research & Design
Produces a structured JSON document comparing storage options
and proposing the final architecture for the face recognition pipeline.

Output: results/storage_architecture.json
"""

import json
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def build_image_storage_comparison():
    return {
        "purpose": "Store original and processed face images",
        "options": {
            "aws_s3": {
                "type": "Object storage (managed)",
                "cost_per_gb_month": 0.023,
                "cost_25gb": 0.58,
                "max_object_size_gb": 5,
                "latency_ms": "10–50",
                "availability_sla": "99.99%",
                "durability": "99.999999999%",
                "cdn_integration": "CloudFront",
                "pros": ["battle-tested", "cheap at scale", "lifecycle policies", "event triggers for Lambda"],
                "cons": ["egress costs at scale", "US-centric pricing"],
                "verdict": "RECOMMENDED",
            },
            "gcs": {
                "type": "Object storage (managed)",
                "cost_per_gb_month": 0.020,
                "cost_25gb": 0.50,
                "latency_ms": "10–50",
                "cdn_integration": "Cloud CDN",
                "pros": ["slightly cheaper", "good Python SDK"],
                "cons": ["less ecosystem tooling than S3"],
                "verdict": "ALTERNATIVE",
            },
            "local_nas": {
                "type": "Network-attached storage",
                "cost_per_gb_month": "~$0.01 (amortised hardware)",
                "latency_ms": "<5 (LAN)",
                "pros": ["fastest local access", "no egress costs", "full control"],
                "cons": ["no redundancy without RAID", "single location", "scaling requires hardware"],
                "verdict": "DEV/PROTOTYPE ONLY",
            },
            "minio": {
                "type": "Self-hosted S3-compatible object store",
                "cost_per_gb_month": "hardware only",
                "latency_ms": "1–10 (LAN)",
                "pros": ["S3 API compatible", "self-hosted", "no cloud dependency"],
                "cons": ["you own ops", "replication is manual"],
                "verdict": "ON-PREMISE OPTION",
            },
        },
    }


def build_vector_storage_comparison():
    return {
        "purpose": "Store and search 512-dim ArcFace embeddings",
        "options": {
            "faiss_flat": {
                "type": "In-memory exact search",
                "open_source": True,
                "self_hosted": True,
                "cost": "free",
                "latency_ms": "<5 (16k vectors)",
                "recall": 1.0,
                "max_practical_vectors": "1–5M (RAM limited)",
                "ram_16k": "32 MB",
                "ram_1m": "2 GB",
                "supports_metadata": False,
                "faiss_compatible": True,
                "pros": ["exact results", "no setup", "already built in Day 1"],
                "cons": ["RAM-only", "no metadata filtering", "manual persistence"],
                "verdict": "RECOMMENDED for <500k vectors",
            },
            "faiss_ivfflat": {
                "type": "In-memory approximate search",
                "open_source": True,
                "self_hosted": True,
                "cost": "free",
                "latency_ms": "<2 (16k vectors)",
                "recall": "0.95–0.99 (tunable)",
                "max_practical_vectors": "50M+",
                "pros": ["much faster at scale", "same API as Flat", "proven in production"],
                "cons": ["approximate — slight recall drop", "requires training step"],
                "verdict": "RECOMMENDED for 500k+ vectors",
            },
            "pgvector": {
                "type": "PostgreSQL extension",
                "open_source": True,
                "self_hosted": True,
                "cost": "free",
                "latency_ms": "1–10 (local)",
                "recall": "0.95–0.99",
                "max_practical_vectors": "10M (with HNSW index)",
                "ram_16k": "uses PostgreSQL shared_buffers",
                "pros": [
                    "SQL + vector search in one DB",
                    "easy metadata filtering (WHERE clauses)",
                    "familiar tooling",
                    "ACID transactions",
                ],
                "cons": ["higher latency than pure FAISS", "PostgreSQL knowledge required"],
                "verdict": "RECOMMENDED if metadata filtering is critical",
            },
            "weaviate": {
                "type": "Dedicated vector database",
                "open_source": True,
                "self_hosted": True,
                "cost": "free (self-hosted) / $25/mo (cloud)",
                "latency_ms": "1–15",
                "recall": "0.95–0.99",
                "pros": [
                    "GraphQL + REST API",
                    "built-in metadata filtering",
                    "Docker deployment",
                    "schema management",
                ],
                "cons": ["Docker dependency", "higher memory overhead", "more complex than FAISS"],
                "verdict": "CONSIDER if building REST-first microservices",
            },
            "pinecone": {
                "type": "Managed vector database",
                "open_source": False,
                "self_hosted": False,
                "cost": "free tier → $70+/month at scale",
                "latency_ms": "5–50 (network)",
                "recall": "0.95–0.99",
                "pros": ["fully managed", "scales automatically", "REST API"],
                "cons": [
                    "no embedding extraction",
                    "data leaves your infra",
                    "expensive at scale vs self-hosted",
                    "not FAISS-compatible",
                ],
                "verdict": "NOT RECOMMENDED — cost + privacy concerns",
            },
        },
    }


def build_metadata_storage():
    return {
        "purpose": "Store per-face metadata: person label, image path, bounding box, timestamp",
        "recommended": "SQLite (dev) → PostgreSQL (production)",
        "schema": {
            "face_embeddings": {
                "id": "SERIAL PRIMARY KEY",
                "image_id": "INTEGER REFERENCES images(id)",
                "label": "TEXT",
                "embedding_index": "INTEGER (FAISS row index)",
                "bbox_x1": "INTEGER",
                "bbox_y1": "INTEGER",
                "bbox_x2": "INTEGER",
                "bbox_y2": "INTEGER",
                "confidence": "FLOAT",
                "created_at": "TIMESTAMP",
            },
            "images": {
                "id": "SERIAL PRIMARY KEY",
                "storage_key": "TEXT (S3 key or local path)",
                "source": "TEXT (upload, cctv, manual)",
                "ingested_at": "TIMESTAMP",
                "width": "INTEGER",
                "height": "INTEGER",
                "faces_detected": "INTEGER",
            },
        },
    }


def build_pipeline_architecture():
    return {
        "ingestion_pipeline": {
            "description": "Bulk image → embedding → index pipeline",
            "steps": [
                "1. Upload images to S3 (or local disk)",
                "2. S3 event triggers SQS message (or Celery task)",
                "3. Worker picks up task: download image → detect faces → extract embeddings",
                "4. Save embeddings to FAISS index (or pgvector)",
                "5. Write metadata (label, bbox, path) to PostgreSQL",
                "6. Acknowledge task completion",
            ],
            "queue": "Celery + Redis (self-hosted) or AWS SQS",
            "workers": "Python processes with InsightFace loaded once at startup",
            "throughput": "3.4 embeddings/sec per CPU worker (from Day 1 benchmark)",
        },
        "search_pipeline": {
            "description": "Query photo → top-N similar faces",
            "steps": [
                "1. Client uploads query image to FastAPI /search endpoint",
                "2. API extracts face embedding (InsightFace ArcFace)",
                "3. FAISS index.search(embedding, top_k)",
                "4. Fetch metadata for top-K result indices from PostgreSQL",
                "5. Return ranked results with scores + image paths",
            ],
            "latency_target": "<100ms end-to-end (CPU, 16k vectors)",
            "search_latency": "<5ms (FAISS query alone, from Day 1)",
            "embedding_latency": "~50ms (InsightFace, CPU)",
        },
        "recommended_stack": {
            "image_store": "AWS S3 (or MinIO for on-premise)",
            "vector_store": "FAISS IndexFlatIP → IVFFlat at 500k+",
            "metadata_db": "PostgreSQL (with pgvector optional for hybrid queries)",
            "queue": "Celery + Redis",
            "api": "FastAPI (prototype built in Task 5)",
            "embedding_model": "InsightFace ArcFace w600k_r50",
            "detection_model": "InsightFace RetinaFace",
        },
    }


def build_scaling_roadmap():
    return {
        "phase_1_prototype": {
            "scale": "5,000 images / 16k embeddings",
            "vector_store": "FAISS IndexFlatIP (32MB RAM)",
            "image_store": "Local disk or S3",
            "api": "FastAPI single process",
            "queue": "None (sync processing)",
            "monthly_cost": "~$0.60 (S3 only)",
        },
        "phase_2_production": {
            "scale": "50,000 images / 60k embeddings",
            "vector_store": "FAISS IVFFlat (nlist=256, 240MB RAM)",
            "image_store": "AWS S3",
            "api": "FastAPI + uvicorn workers",
            "queue": "Celery + Redis",
            "monthly_cost": "~$5–20 (S3 + small VPS)",
        },
        "phase_3_scale": {
            "scale": "500,000+ images / 600k+ embeddings",
            "vector_store": "FAISS IVFFlat or pgvector with HNSW (2–4GB RAM)",
            "image_store": "AWS S3 + CloudFront CDN",
            "api": "FastAPI behind load balancer",
            "queue": "Celery + Redis cluster",
            "monthly_cost": "$80–200+ (depends on query volume)",
        },
    }


def main():
    print("=" * 60)
    print("DAY 3 — Task 7: Storage Architecture Design")
    print("=" * 60)

    output = {
        "generated": datetime.now().isoformat(),
        "image_storage": build_image_storage_comparison(),
        "vector_storage": build_vector_storage_comparison(),
        "metadata_storage": build_metadata_storage(),
        "pipeline_architecture": build_pipeline_architecture(),
        "scaling_roadmap": build_scaling_roadmap(),
    }

    out = RESULTS_DIR / "storage_architecture.json"
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Saved → {out}")

    print("\nRecommended stack:")
    stack = output["pipeline_architecture"]["recommended_stack"]
    for k, v in stack.items():
        print(f"  {k:<20}: {v}")


if __name__ == "__main__":
    main()