# day3/api_server.py
"""
Day 3 Task 5: FastAPI Face Search Endpoint Prototype
Uses the FAISS index built in Day 1 (embeddings/face_index.faiss).

Endpoints:
  GET  /health           — service health check
  POST /search           — search by uploaded image
  GET  /stats            — index statistics
  POST /embed            — extract embedding only (no search)

Run:
  uvicorn day3.api_server:app --reload --port 8000

Test:
  curl -X POST http://localhost:8000/search \
    -F "file=@datasets/lfw_home/lfw_funneled/George_W_Bush/George_W_Bush_0001.jpg" \
    -F "top_k=5"
"""

import time
import io
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, Form
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    raise ImportError(
        "FastAPI dependencies not installed.\n"
        "Run: pip install fastapi uvicorn[standard] python-multipart"
    )

# ─── paths (relative to project root when launched with uvicorn from root) ───
FAISS_INDEX_PATH = Path("embeddings/face_index.faiss")
METADATA_PATH    = Path("embeddings/metadata.pkl")
EMBEDDINGS_PATH  = Path("embeddings/embeddings.npy")

# ─── lazy-loaded singletons ───────────────────────────────────────────────────
_index    = None
_metadata = None
_app_insight = None


def get_index():
    global _index, _metadata
    if _index is None:
        import faiss, pickle
        if not FAISS_INDEX_PATH.exists():
            raise RuntimeError(
                f"FAISS index not found at {FAISS_INDEX_PATH}. "
                "Run: python3 src/embed.py datasets/lfw_home/lfw_funneled"
            )
        _index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(METADATA_PATH, "rb") as f:
            _metadata = pickle.load(f)
    return _index, _metadata


def get_face_model():
    global _app_insight
    if _app_insight is None:
        from insightface.app import FaceAnalysis
        _app_insight = FaceAnalysis(providers=["CPUExecutionProvider"])
        _app_insight.prepare(ctx_id=-1, det_size=(640, 640))
    return _app_insight


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Face Search API",
    description="Reverse image search for faces using InsightFace ArcFace + FAISS",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic models ──────────────────────────────────────────────────────────

class FaceMatch(BaseModel):
    rank: int
    score: float
    label: str
    image_path: str
    bbox: Optional[List[int]] = None


class SearchResponse(BaseModel):
    query_face_detected: bool
    num_faces_in_query: int
    matches: List[FaceMatch]
    latency_ms: float
    index_size: int


class EmbedResponse(BaseModel):
    face_detected: bool
    num_faces: int
    embedding: Optional[List[float]]
    embedding_dim: Optional[int]
    latency_ms: float


class StatsResponse(BaseModel):
    index_size: int
    embedding_dim: int
    index_path: str
    metadata_entries: int
    faiss_index_type: str


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    model_loaded: bool
    timestamp: float


# ─── helpers ─────────────────────────────────────────────────────────────────

def decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes → BGR numpy array."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Ensure the file is a valid JPEG/PNG.")
    return img


def extract_embedding(img_bgr: np.ndarray):
    """
    Run InsightFace detection + embedding on BGR image.
    Returns (faces, embedding) where embedding is the first detected face's
    normed_embedding, or None if no face found.
    """
    model  = get_face_model()
    faces  = model.get(img_bgr)
    if not faces:
        return faces, None
    embedding = faces[0].normed_embedding.astype("float32")
    return faces, embedding


def search_index(embedding: np.ndarray, top_k: int = 10):
    """Search FAISS index and return (scores, metadata_entries)."""
    index, metadata = get_index()
    q_vec = embedding.reshape(1, -1)
    scores, indices = index.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        meta = metadata[idx] if idx < len(metadata) else {}
        results.append({
            "score": float(score),
            "label": meta.get("label", "unknown"),
            "image_path": meta.get("path", ""),
            "bbox": meta.get("bbox", None),
        })
    return results


# ─── routes ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    index_loaded = FAISS_INDEX_PATH.exists()
    model_loaded = _app_insight is not None
    return HealthResponse(
        status="ok",
        index_loaded=index_loaded,
        model_loaded=model_loaded,
        timestamp=time.time(),
    )


@app.get("/stats", response_model=StatsResponse)
def stats():
    index, metadata = get_index()
    return StatsResponse(
        index_size=index.ntotal,
        embedding_dim=index.d,
        index_path=str(FAISS_INDEX_PATH.resolve()),
        metadata_entries=len(metadata),
        faiss_index_type=type(index).__name__,
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(file: UploadFile = File(...)):
    """Extract face embedding from an uploaded image."""
    t0 = time.time()
    contents = await file.read()
    try:
        img = decode_image(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    faces, embedding = extract_embedding(img)
    latency_ms = round((time.time() - t0) * 1000, 2)

    if embedding is None:
        return EmbedResponse(
            face_detected=False,
            num_faces=0,
            embedding=None,
            embedding_dim=None,
            latency_ms=latency_ms,
        )

    return EmbedResponse(
        face_detected=True,
        num_faces=len(faces),
        embedding=embedding.tolist(),
        embedding_dim=len(embedding),
        latency_ms=latency_ms,
    )


@app.post("/search", response_model=SearchResponse)
async def search(
    file: UploadFile = File(...),
    top_k: int = Form(default=10, ge=1, le=100),
):
    """Upload an image and return the top-N most similar faces in the index."""
    t0 = time.time()
    contents = await file.read()
    try:
        img = decode_image(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    faces, embedding = extract_embedding(img)

    if embedding is None:
        latency_ms = round((time.time() - t0) * 1000, 2)
        index, _ = get_index()
        return SearchResponse(
            query_face_detected=False,
            num_faces_in_query=0,
            matches=[],
            latency_ms=latency_ms,
            index_size=index.ntotal,
        )

    raw_results = search_index(embedding, top_k)
    index, _ = get_index()
    latency_ms = round((time.time() - t0) * 1000, 2)

    matches = [
        FaceMatch(
            rank=i + 1,
            score=r["score"],
            label=r["label"],
            image_path=r["image_path"],
            bbox=r["bbox"],
        )
        for i, r in enumerate(raw_results)
    ]

    return SearchResponse(
        query_face_detected=True,
        num_faces_in_query=len(faces),
        matches=matches,
        latency_ms=latency_ms,
        index_size=index.ntotal,
    )


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("day3.api_server:app", host="0.0.0.0", port=8000, reload=True)