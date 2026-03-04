# src/embed.py
import cv2
import numpy as np
import faiss
import pickle
import time
from pathlib import Path
from tqdm import tqdm

def generate_embeddings_insightface(image_folder: str, output_dir: str = "embeddings"):
    """
    Detect faces + generate ArcFace embeddings using InsightFace.
    Saves a FAISS index and metadata to output_dir.
    """
    from insightface.app import FaceAnalysis

    Path(output_dir).mkdir(exist_ok=True)
    image_paths = list(Path(image_folder).rglob("*.jpg"))
    
    if not image_paths:
        print(f"No images found in {image_folder}")
        return

    print(f"Processing {len(image_paths)} images with InsightFace (ArcFace)...")
    
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    embeddings = []
    metadata = []   # {image_path, face_index, bbox}
    t0 = time.time()

    for img_path in tqdm(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        faces = app.get(img)
        for i, face in enumerate(faces):
            emb = face.normed_embedding  # 512-dim ArcFace embedding
            embeddings.append(emb)
            metadata.append({
                "image_path": str(img_path),
                "face_index": i,
                "bbox": face.bbox.tolist(),
                "label": img_path.parent.name,  # folder name = person name in LFW
            })

    elapsed = time.time() - t0
    print(f"\n✓ Extracted {len(embeddings)} embeddings in {elapsed:.1f}s")
    print(f"  ({len(embeddings)/elapsed:.1f} embeddings/sec)")

    if not embeddings:
        print("No faces detected.")
        return

    # ── Build FAISS Index ──────────────────────────────────────────────────
    dim = len(embeddings[0])
    emb_matrix = np.array(embeddings).astype("float32")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(emb_matrix)
    
    index = faiss.IndexFlatIP(dim)   # Inner Product = cosine after normalization
    index.add(emb_matrix)

    faiss.write_index(index, f"{output_dir}/face_index.faiss")
    with open(f"{output_dir}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    np.save(f"{output_dir}/embeddings.npy", emb_matrix)

    print(f"\n✓ FAISS index saved → {output_dir}/face_index.faiss")
    print(f"✓ Metadata saved   → {output_dir}/metadata.pkl")
    print(f"✓ Index size: {index.ntotal} vectors, dim={dim}")
    return index, metadata


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "datasets/lfw"
    generate_embeddings_insightface(folder)