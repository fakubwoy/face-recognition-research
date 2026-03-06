# day3/eval_vector_dbs.py
"""
Day 3 Task 3: Vector Database Benchmark
Compares FAISS (local), pgvector (PostgreSQL), Weaviate (local Docker),
and simulates Pinecone (documented specs, no live API required).

Tests: index build time, query latency (P50/P95/P99), recall@K, memory use.

Output: results/vector_db_benchmark.json
"""

import json
import time
import os
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR    = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Synthetic benchmark config
N_VECTORS  = int(os.environ.get("DAY3_N_VECTORS", 16000))   # match Day 1 FAISS index
DIM        = 512                                             # ArcFace embedding dim
N_QUERIES  = 200
TOP_K      = 10
SEED       = 42


# ─── helpers ─────────────────────────────────────────────────────────────────

def generate_data(n, dim, seed=42):
    rng = np.random.default_rng(seed)
    vecs = rng.random((n, dim)).astype("float32")
    # Normalize (cosine similarity space)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / (norms + 1e-8)


def percentiles(latencies):
    arr = np.array(latencies) * 1000  # → ms
    return {
        "p50_ms":  round(float(np.percentile(arr, 50)), 3),
        "p95_ms":  round(float(np.percentile(arr, 95)), 3),
        "p99_ms":  round(float(np.percentile(arr, 99)), 3),
        "mean_ms": round(float(np.mean(arr)), 3),
    }


def ground_truth_topk(corpus, queries, k):
    """Brute-force exact top-k via dot product (since vecs are normalized)."""
    scores = queries @ corpus.T        # (Q, N)
    return np.argsort(-scores, axis=1)[:, :k]


def recall_at_k(predicted, ground_truth):
    """Mean recall@K across all queries."""
    recalls = []
    for pred_row, gt_row in zip(predicted, ground_truth):
        gt_set = set(gt_row.tolist())
        hit = len(set(pred_row.tolist()) & gt_set)
        recalls.append(hit / len(gt_set))
    return round(float(np.mean(recalls)), 4)


# ─── FAISS benchmarks ────────────────────────────────────────────────────────

def bench_faiss_flat(corpus, queries, gt):
    print("\n[FAISS IndexFlatIP] ...")
    try:
        import faiss

        # Build
        t0 = time.time()
        idx = faiss.IndexFlatIP(DIM)
        idx.add(corpus)
        build_sec = time.time() - t0

        # Query latencies
        latencies = []
        results_all = []
        for q in queries:
            q_vec = q.reshape(1, -1)
            t0 = time.time()
            D, I = idx.search(q_vec, TOP_K)
            latencies.append(time.time() - t0)
            results_all.append(I[0])

        recall = recall_at_k(results_all, gt)
        mem_mb = round((corpus.nbytes + idx.sa_code_size() * N_VECTORS
                        if hasattr(idx, "sa_code_size") else corpus.nbytes) / 1e6, 2)

        result = {
            "system": "FAISS IndexFlatIP",
            "type": "exact",
            "n_vectors": N_VECTORS,
            "dim": DIM,
            "build_sec": round(build_sec, 3),
            "index_size_mb": round(corpus.nbytes / 1e6, 2),
            "recall_at_k": recall,
            **percentiles(latencies),
        }
        print(f"  ✓ build={build_sec:.3f}s, p50={result['p50_ms']}ms, recall={recall}")
        return result

    except ImportError:
        return {"system": "FAISS IndexFlatIP", "error": "faiss-cpu not installed"}
    except Exception as e:
        return {"system": "FAISS IndexFlatIP", "error": str(e)}


def bench_faiss_ivf(corpus, queries, gt, nlist=128):
    print(f"\n[FAISS IVFFlat nlist={nlist}] ...")
    try:
        import faiss

        t0 = time.time()
        quantizer = faiss.IndexFlatIP(DIM)
        idx = faiss.IndexIVFFlat(quantizer, DIM, nlist, faiss.METRIC_INNER_PRODUCT)
        idx.train(corpus)
        idx.add(corpus)
        build_sec = time.time() - t0

        idx.nprobe = 16  # search 16 clusters

        latencies = []
        results_all = []
        for q in queries:
            q_vec = q.reshape(1, -1)
            t0 = time.time()
            D, I = idx.search(q_vec, TOP_K)
            latencies.append(time.time() - t0)
            results_all.append(I[0])

        recall = recall_at_k(results_all, gt)

        result = {
            "system": f"FAISS IVFFlat (nlist={nlist}, nprobe=16)",
            "type": "approximate",
            "n_vectors": N_VECTORS,
            "dim": DIM,
            "build_sec": round(build_sec, 3),
            "index_size_mb": round(corpus.nbytes / 1e6, 2),
            "recall_at_k": recall,
            **percentiles(latencies),
        }
        print(f"  ✓ build={build_sec:.3f}s, p50={result['p50_ms']}ms, recall={recall}")
        return result

    except ImportError:
        return {"system": "FAISS IVFFlat", "error": "faiss-cpu not installed"}
    except Exception as e:
        return {"system": "FAISS IVFFlat", "error": str(e)}


def bench_faiss_hnsw(corpus, queries, gt, M=32, ef=200):
    print(f"\n[FAISS HNSW M={M}] ...")
    try:
        import faiss

        t0 = time.time()
        idx = faiss.IndexHNSWFlat(DIM, M, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = ef
        idx.add(corpus)
        build_sec = time.time() - t0

        idx.hnsw.efSearch = 64

        latencies = []
        results_all = []
        for q in queries:
            q_vec = q.reshape(1, -1)
            t0 = time.time()
            D, I = idx.search(q_vec, TOP_K)
            latencies.append(time.time() - t0)
            results_all.append(I[0])

        recall = recall_at_k(results_all, gt)

        result = {
            "system": f"FAISS HNSW (M={M}, efSearch=64)",
            "type": "approximate",
            "n_vectors": N_VECTORS,
            "dim": DIM,
            "build_sec": round(build_sec, 3),
            "index_size_mb": round((N_VECTORS * DIM * 4 + N_VECTORS * M * 2 * 8) / 1e6, 2),
            "recall_at_k": recall,
            **percentiles(latencies),
        }
        print(f"  ✓ build={build_sec:.3f}s, p50={result['p50_ms']}ms, recall={recall}")
        return result

    except ImportError:
        return {"system": "FAISS HNSW", "error": "faiss-cpu not installed"}
    except Exception as e:
        return {"system": "FAISS HNSW", "error": str(e)}


# ─── pgvector ─────────────────────────────────────────────────────────────────

def bench_pgvector(corpus, queries, gt):
    """
    Requires: PostgreSQL running locally with pgvector extension.
    Connection: postgresql://postgres:postgres@localhost:5432/face_db
    """
    print("\n[pgvector] ...")
    pg_url = os.environ.get(
        "PGVECTOR_URL", "postgresql://postgres:postgres@localhost:5432/face_db"
    )
    try:
        import psycopg2
        from pgvector.psycopg2 import register_vector

        conn = psycopg2.connect(pg_url)
        register_vector(conn)
        cur = conn.cursor()

        # Setup
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("DROP TABLE IF EXISTS face_embeddings")
        cur.execute(f"CREATE TABLE face_embeddings (id SERIAL, embedding vector({DIM}))")
        conn.commit()

        # Build
        t0 = time.time()
        for i, vec in enumerate(corpus):
            cur.execute(
                "INSERT INTO face_embeddings (embedding) VALUES (%s)",
                (vec.tolist(),),
            )
        conn.commit()

        # IVFFlat index
        cur.execute(
            f"CREATE INDEX ON face_embeddings USING ivfflat (embedding vector_cosine_ops) "
            f"WITH (lists = 128)"
        )
        conn.commit()
        build_sec = time.time() - t0

        # Query
        latencies = []
        results_all = []
        cur.execute("SET ivfflat.probes = 16")
        for q in queries:
            t0 = time.time()
            cur.execute(
                f"SELECT id-1 FROM face_embeddings ORDER BY embedding <=> %s LIMIT {TOP_K}",
                (q.tolist(),),
            )
            ids = [row[0] for row in cur.fetchall()]
            latencies.append(time.time() - t0)
            results_all.append(ids)

        recall = recall_at_k(results_all, gt)
        cur.execute("DROP TABLE IF EXISTS face_embeddings")
        conn.commit()
        conn.close()

        result = {
            "system": "pgvector (IVFFlat, lists=128, probes=16)",
            "type": "approximate",
            "n_vectors": N_VECTORS,
            "dim": DIM,
            "build_sec": round(build_sec, 2),
            "index_size_mb": round((N_VECTORS * DIM * 4) / 1e6, 2),
            "recall_at_k": recall,
            **percentiles(latencies),
        }
        print(f"  ✓ build={build_sec:.2f}s, p50={result['p50_ms']}ms, recall={recall}")
        return result

    except ImportError:
        msg = "psycopg2 or pgvector not installed (pip install psycopg2-binary pgvector)"
        print(f"  ⚠ {msg}")
        return {"system": "pgvector", "status": "not_installed", "note": msg}
    except Exception as e:
        msg = str(e)
        print(f"  ⚠ pgvector unavailable: {msg[:80]}")
        return {
            "system": "pgvector",
            "status": "unavailable",
            "note": f"Requires PostgreSQL + pgvector extension. Error: {msg[:120]}",
            "documented_specs": {
                "type": "approximate (IVFFlat / HNSW)",
                "latency_estimate_ms": "1–5 (local), 5–20 (networked)",
                "recall_at_k": "0.95–0.99",
                "storage": "shares PostgreSQL instance",
                "self_hosted": True,
                "cost": "free (open source)",
            },
        }


# ─── Weaviate ─────────────────────────────────────────────────────────────────

def bench_weaviate(corpus, queries, gt):
    """
    Requires: Weaviate running in Docker: docker run -p 8080:8080 semitechnologies/weaviate
    """
    print("\n[Weaviate] ...")
    weaviate_url = os.environ.get("WEAVIATE_URL", "http://localhost:8080")

    try:
        import weaviate
        from weaviate.classes.config import Configure, Property, DataType

        client = weaviate.connect_to_local(
            host="localhost", port=8080, grpc_port=50051
        )

        if not client.is_ready():
            raise ConnectionError("Weaviate not ready")

        coll_name = "FaceEmbeddings"
        if client.collections.exists(coll_name):
            client.collections.delete(coll_name)

        collection = client.collections.create(
            name=coll_name,
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=weaviate.classes.config.VectorDistances.COSINE,
                ef_construction=200,
                max_connections=32,
            ),
        )

        # Build
        t0 = time.time()
        with collection.batch.dynamic() as batch:
            for i, vec in enumerate(corpus):
                batch.add_object(properties={"face_id": i}, vector=vec.tolist())
        build_sec = time.time() - t0

        # Query
        latencies = []
        results_all = []
        for q in queries:
            t0 = time.time()
            response = collection.query.near_vector(
                near_vector=q.tolist(),
                limit=TOP_K,
                return_properties=["face_id"],
            )
            ids = [int(o.properties["face_id"]) for o in response.objects]
            latencies.append(time.time() - t0)
            results_all.append(ids)

        recall = recall_at_k(results_all, gt)
        client.collections.delete(coll_name)
        client.close()

        result = {
            "system": "Weaviate (HNSW, ef_construction=200, M=32)",
            "type": "approximate",
            "n_vectors": N_VECTORS,
            "dim": DIM,
            "build_sec": round(build_sec, 2),
            "recall_at_k": recall,
            **percentiles(latencies),
        }
        print(f"  ✓ build={build_sec:.2f}s, p50={result['p50_ms']}ms, recall={recall}")
        return result

    except ImportError:
        msg = "weaviate-client not installed (pip install weaviate-client)"
        print(f"  ⚠ {msg}")
        return {"system": "Weaviate", "status": "not_installed", "note": msg}
    except Exception as e:
        msg = str(e)
        print(f"  ⚠ Weaviate unavailable: {msg[:80]}")
        return {
            "system": "Weaviate",
            "status": "unavailable",
            "note": f"Requires Docker: docker run -p 8080:8080 cr.weaviate.io/semitechnologies/weaviate:latest. Error: {msg[:100]}",
            "documented_specs": {
                "type": "approximate (HNSW)",
                "latency_estimate_ms": "1–10 (local Docker)",
                "recall_at_k": "0.95–0.99",
                "storage": "self-hosted Docker or Weaviate Cloud",
                "self_hosted": True,
                "cost": "free (open source) / $25/mo cloud",
                "graphql_api": True,
                "rest_api": True,
            },
        }


# ─── Pinecone (documented specs) ─────────────────────────────────────────────

def bench_pinecone_documented():
    """
    Pinecone requires a paid API key; we document published specs instead.
    Set PINECONE_API_KEY env var to run live.
    """
    print("\n[Pinecone] ...")
    api_key = os.environ.get("PINECONE_API_KEY")

    if not api_key:
        print("  ℹ  PINECONE_API_KEY not set — returning documented specs")
        return {
            "system": "Pinecone (managed vector DB)",
            "status": "documented_specs",
            "note": "Set PINECONE_API_KEY env var to run live benchmark",
            "documented_specs": {
                "type": "approximate (proprietary ANNS)",
                "latency_p50_ms": "5–20 (serverless, us-east-1)",
                "latency_p99_ms": "50–100",
                "recall_at_k": "0.95–0.99 (tunable)",
                "index_size_limit": "unlimited (serverless)",
                "self_hosted": False,
                "free_tier": "2GB storage, shared",
                "pricing_per_1m_queries": "$0.08 (serverless)",
                "pricing_storage_per_gb_month": "$0.33",
                "cost_16k_vectors_512dim_mb": round(16000 * 512 * 4 / 1e6, 1),
                "estimated_monthly_cost_low_volume": "$0 (free tier)",
                "estimated_monthly_cost_100k_queries": "$8–12/month",
            },
        }

    try:
        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=api_key)
        idx_name = "face-bench-tmp"

        if idx_name in [i.name for i in pc.list_indexes()]:
            pc.delete_index(idx_name)

        pc.create_index(
            name=idx_name,
            dimension=DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        index = pc.Index(idx_name)

        # Build
        t0 = time.time()
        batch_size = 100
        for i in range(0, len(corpus), batch_size):
            batch = corpus[i : i + batch_size]
            vectors = [(str(i + j), v.tolist()) for j, v in enumerate(batch)]
            index.upsert(vectors=vectors)
        build_sec = time.time() - t0

        # Query
        latencies = []
        results_all = []
        for q in queries:
            t0 = time.time()
            resp = index.query(vector=q.tolist(), top_k=TOP_K)
            ids = [int(m["id"]) for m in resp["matches"]]
            latencies.append(time.time() - t0)
            results_all.append(ids)

        recall = recall_at_k(results_all, gt)
        pc.delete_index(idx_name)

        result = {
            "system": "Pinecone (serverless)",
            "type": "approximate",
            "n_vectors": N_VECTORS,
            "dim": DIM,
            "build_sec": round(build_sec, 2),
            "recall_at_k": recall,
            **percentiles(latencies),
        }
        print(f"  ✓ build={build_sec:.2f}s, p50={result['p50_ms']}ms, recall={recall}")
        return result

    except ImportError:
        return {"system": "Pinecone", "status": "not_installed",
                "note": "pip install pinecone"}
    except Exception as e:
        return {"system": "Pinecone", "status": "error", "error": str(e)}


# ─── Summary comparison ───────────────────────────────────────────────────────

def build_comparison_summary(results):
    """Builds a structured comparison for the Day 3 report."""
    summary = {
        "criteria": [
            "latency", "recall", "self_hosted", "scaling",
            "setup_complexity", "cost", "faiss_compatible"
        ],
        "systems": {}
    }
    for r in results:
        name = r["system"].split(" ")[0]
        entry = {
            "p50_ms":    r.get("p50_ms"),
            "p99_ms":    r.get("p99_ms"),
            "recall_k":  r.get("recall_at_k"),
            "build_sec": r.get("build_sec"),
            "status":    r.get("status", "tested"),
        }
        if "documented_specs" in r:
            entry.update(r["documented_specs"])
        summary["systems"][name] = entry
    return summary


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DAY 3 — Task 3: Vector Database Benchmark")
    print(f"         {N_VECTORS} vectors × {DIM} dims, {N_QUERIES} queries, top-{TOP_K}")
    print("=" * 60)

    print(f"\nGenerating {N_VECTORS} synthetic ArcFace-like embeddings...")
    corpus  = generate_data(N_VECTORS, DIM, SEED)
    queries = generate_data(N_QUERIES, DIM, SEED + 1)

    print("Computing ground truth (brute force)...")
    gt = ground_truth_topk(corpus, queries, TOP_K)

    results = []
    results.append(bench_faiss_flat(corpus, queries, gt))
    results.append(bench_faiss_ivf(corpus, queries, gt))
    results.append(bench_faiss_hnsw(corpus, queries, gt))
    results.append(bench_pgvector(corpus, queries, gt))
    results.append(bench_weaviate(corpus, queries, gt))
    results.append(bench_pinecone_documented())

    output = {
        "benchmark_config": {
            "n_vectors": N_VECTORS,
            "dim": DIM,
            "n_queries": N_QUERIES,
            "top_k": TOP_K,
        },
        "results": results,
        "comparison_summary": build_comparison_summary(results),
    }

    out = RESULTS_DIR / "vector_db_benchmark.json"
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Saved → {out}")

    print(f"\n{'System':<45} {'P50ms':>7} {'P99ms':>7} {'Recall':>8} {'Build(s)':>10}")
    print("-" * 80)
    for r in results:
        sys_name = r["system"][:44]
        p50      = f"{r['p50_ms']:.2f}" if r.get("p50_ms") else "—"
        p99      = f"{r['p99_ms']:.2f}" if r.get("p99_ms") else "—"
        recall   = f"{r['recall_at_k']:.4f}" if r.get("recall_at_k") else "—"
        build    = f"{r['build_sec']:.2f}" if r.get("build_sec") else "—"
        print(f"{sys_name:<45} {p50:>7} {p99:>7} {recall:>8} {build:>10}")


if __name__ == "__main__":
    main()