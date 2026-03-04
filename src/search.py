# src/search.py
import cv2
import numpy as np
import faiss
import pickle
from pathlib import Path
import matplotlib
matplotlib.use("Agg")   # headless (WSL has no display)
import matplotlib.pyplot as plt


def search_face(query_image_path: str,
                index_path: str = "embeddings/face_index.faiss",
                meta_path: str = "embeddings/metadata.pkl",
                top_k: int = 5,
                output_path: str = "outputs/search_result.png"):
    """
    Given a query image, find the top-k most similar faces in the index.
    """
    from insightface.app import FaceAnalysis

    # ── Load index ────────────────────────────────────────────────────────
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    print(f"Loaded FAISS index: {index.ntotal} embeddings")

    # ── Detect + embed query face ─────────────────────────────────────────
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    img = cv2.imread(query_image_path)
    if img is None:
        print(f"Could not read {query_image_path}")
        return

    faces = app.get(img)
    if not faces:
        print("No face detected in query image.")
        return

    print(f"Detected {len(faces)} face(s) in query image. Using the first.")
    query_emb = faces[0].normed_embedding.reshape(1, -1).astype("float32")
    faiss.normalize_L2(query_emb)

    # ── Search ─────────────────────────────────────────────────────────────
    distances, indices = index.search(query_emb, top_k)

    print(f"\nTop {top_k} matches:")
    print(f"{'Rank':<6} {'Score':>7}  {'Label':<25} {'Image'}")
    print("-" * 75)

    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        m = metadata[idx]
        score = float(dist)
        print(f"{rank:<6} {score:>7.4f}  {m['label']:<25} {Path(m['image_path']).name}")
        results.append((m, score))

    # ── Visualise ──────────────────────────────────────────────────────────
    Path("outputs").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, top_k + 1, figsize=(4 * (top_k + 1), 4))
    fig.suptitle("Face Search Results", fontsize=14, fontweight="bold")

    # Query
    q_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(q_rgb)
    axes[0].set_title("Query", fontweight="bold", color="blue")
    axes[0].axis("off")

    # Matches
    for i, (m, score) in enumerate(results):
        match_img = cv2.imread(m["image_path"])
        if match_img is not None:
            match_rgb = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
            axes[i + 1].imshow(match_rgb)
        axes[i + 1].set_title(f"#{i+1} {m['label']}\n{score:.3f}", fontsize=8)
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Result saved → {output_path}")
    return results


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "datasets/lfw/lfw/George_W_Bush/George_W_Bush_0001.jpg"
    search_face(query)