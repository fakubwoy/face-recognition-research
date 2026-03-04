# demo.py — Full face recognition pipeline demo
import cv2, faiss, pickle, numpy as np
from pathlib import Path
from tqdm import tqdm

def build_and_query(image_folder, query_image, top_k=3):
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    # BUILD INDEX
    embeddings, meta = [], []
    for p in tqdm(list(Path(image_folder).rglob("*.jpg"))[:200], desc="Indexing"):
        img = cv2.imread(str(p))
        if img is None: continue
        for i, face in enumerate(app.get(img)):
            embeddings.append(face.normed_embedding)
            meta.append({"path": str(p), "label": p.parent.name})

    X = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    print(f"\n✓ Index built: {index.ntotal} face embeddings")

    # QUERY
    q_img = cv2.imread(query_image)
    q_faces = app.get(q_img)
    if not q_faces:
        print("No face in query image"); return
    q_emb = q_faces[0].normed_embedding.reshape(1, -1).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)

    print(f"\nTop {top_k} matches for: {Path(query_image).name}")
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), 1):
        m = meta[idx]
        print(f"  #{rank} [{score:.4f}] {m['label']} — {Path(m['path']).name}")

if __name__ == "__main__":
    import sys
    folder = sys.argv[1]
    query  = sys.argv[2]
    build_and_query(folder, query)