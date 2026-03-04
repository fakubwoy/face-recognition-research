# Day 1 Research — Face Recognition Pipeline

**Date:** 2026-03-04  
**Status:** ✅ Complete  
**Researcher:** fakubwoy

---

## 1. Problem Overview

The goal is to build a reverse face search system: given a query photo, find all matching
or similar faces across a large image dataset. This has applications in event photography,
law enforcement tooling, and social media tagging.

### System Pipeline

```
User uploads query photo
         │
         ▼
  ┌─────────────┐
  │ Face        │  Detect all face regions in the image
  │ Detection   │  → bounding boxes + landmarks
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ Embedding   │  Run each face crop through a CNN
  │ Generation  │  → 512-dimensional float vector
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ Vector      │  Cosine similarity search across stored embeddings
  │ Search      │  → top-N closest matches
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ Result      │  Return matched image paths, scores, metadata
  │ Retrieval   │
  └─────────────┘
```

### Expected Scale

| Parameter | Value |
|-----------|-------|
| Initial dataset | ~5,000 images |
| Faces per image | 1–5 (average ~1.2 in LFW) |
| Total embeddings | 10,000–20,000 (LFW: 16,058) |
| Embedding size | 512 floats = 2KB per face |
| Total index size | ~32MB for 16k embeddings |

### Key Challenges Identified

- **Low-quality images** — CCTV footage, compressed mobile photos, motion blur
- **Lighting variation** — outdoor/indoor, flash, shadows
- **Pose variation** — profile views, tilted heads, looking away
- **Partial occlusion** — masks, glasses, hair, hands
- **Search speed** — sub-second response at 100k+ embeddings
- **False positive rate** — critical in identification contexts

---

## 2. Face Detection Algorithm Research

### Algorithms Evaluated

#### MTCNN (Multi-task Cascaded Convolutional Networks)
- Three-stage cascade: P-Net → R-Net → O-Net
- Detects face bounding box + 5 facial landmarks
- **Accuracy:** Good on frontal, degrades on profile/occluded
- **Speed:** ~10 FPS CPU, slower than modern alternatives
- **Low-res:** Struggles below 30×30 pixels
- **Verdict:** Solid baseline, not state-of-the-art

#### RetinaFace (InsightFace)
- Single-stage detector with feature pyramid network
- Detects box + 5 landmarks + 3D face mesh (buffalo_l model)
- **Accuracy:** Best-in-class, handles occlusion and pose well
- **Speed:** ~15–20 FPS CPU with ONNX runtime
- **Low-res:** Handles 20×20 pixel faces with `det_size=(640,640)`
- **Verdict:** ✅ Recommended for production

#### YOLOv8-Face
- Adapted from YOLOv8 object detector
- Extremely fast inference (GPU: 100+ FPS)
- **Accuracy:** Near RetinaFace level
- **Speed:** Fastest available on GPU
- **Low-res:** Good with augmentation
- **Verdict:** Best for real-time/CCTV use cases

#### Dlib CNN Detector
- HOG-based (fast) and CNN-based (accurate)
- Slower than InsightFace, limited to frontal faces in HOG mode
- **Verdict:** Legacy option, not recommended for new projects

### Detection Comparison Table

| Detector | Accuracy | CPU Speed | Low-Res | Landmarks | Recommended |
|----------|----------|-----------|---------|-----------|-------------|
| RetinaFace | ★★★★★ | 15–20 FPS | ★★★★★ | 5-point + 3D | ✅ Yes |
| MTCNN | ★★★★ | 8–12 FPS | ★★★ | 5-point | ⚠️ Legacy |
| YOLOv8-Face | ★★★★★ | 30+ FPS | ★★★★ | 5-point | ✅ GPU use |
| Dlib HOG | ★★★ | 20 FPS | ★★ | None | ❌ No |
| OpenCV Haar | ★★ | 50+ FPS | ★ | None | ❌ Baseline only |

---

## 3. Face Recognition (Embedding) Model Research

### Models Evaluated

#### FaceNet (Google, 2015)
- Triplet loss training on 200M+ images
- Embedding: 128 or 512 dimensions
- **LFW Accuracy:** 99.63%
- **Notes:** Still competitive, widely deployed, good ecosystem

#### ArcFace (InsightFace, 2019)
- Additive Angular Margin Loss — improves class separation
- Embedding: 512 dimensions
- **LFW Accuracy:** 99.83% (w600k_r50 backbone)
- **Notes:** Current state-of-the-art for face verification; used in this project

#### Dlib ResNet
- ResNet-based, 128-dim embedding
- **LFW Accuracy:** 99.38%
- **Notes:** Lightweight, easy to use, less accurate than ArcFace

#### DeepFace (Wrapper Library)
- Wraps FaceNet, ArcFace, VGG-Face, OpenFace, DeepID, Dlib
- Easy API but overhead per call
- Good for research/comparison, not optimal for production

### Embedding Model Comparison

| Model | LFW Accuracy | Embedding Dim | Speed (CPU) | Low-Res | Size |
|-------|-------------|---------------|-------------|---------|------|
| ArcFace (w600k_r50) | **99.83%** | 512 | Fast | ★★★★★ | 170MB |
| Facenet512 | 99.65% | 512 | Medium | ★★★★ | 90MB |
| FaceNet128 | 99.63% | 128 | Fast | ★★★★ | 90MB |
| VGG-Face | 98.78% | 4096 | Slow | ★★★ | 550MB |
| Dlib ResNet | 99.38% | 128 | Medium | ★★★ | 22MB |

---

## 4. Framework Comparison

### InsightFace
- **Best overall framework** for production use
- Integrates RetinaFace detection + ArcFace recognition in one pipeline
- ONNX runtime enables fast CPU inference without PyTorch
- Models auto-download on first run (~300MB total)
- Python API: `FaceAnalysis().get(img)` returns faces with `.normed_embedding`
- **Tested:** ✅ Working — 16,058 embeddings extracted from LFW at 3.4 emb/sec CPU

### DeepFace
- Excellent for research and quick comparisons
- Supports 9 models (ArcFace, FaceNet, VGG-Face, etc.) and 5 detectors
- Simple API: `DeepFace.verify(img1, img2)` and `DeepFace.find(img, db_path)`
- Slower per call due to wrapper overhead
- Good for validating InsightFace results during research

### Dlib + face_recognition
- Adam Geitgey's `face_recognition` library wraps Dlib
- Extremely simple: 3 lines to detect + embed
- Not scalable — no batch processing, no GPU
- Lower accuracy than ArcFace

### AWS Rekognition / Azure Face API
- Managed, no setup, scales automatically
- AWS: $1/1000 images (~$5 for 5000 photos)
- Azure: Free tier 30k calls/month, then ~$0.001/call
- Privacy concern: images leave your infrastructure
- No embedding access — can't build custom FAISS index

---

## 5. Vector Search Strategy

### Why Vector Search?

Face embeddings are 512-dimensional float vectors. Matching a query means finding
the stored embedding with highest cosine similarity. At 16,000+ vectors, brute-force
is still fast, but at 100k+ we need indexing.

### FAISS (Facebook AI Similarity Search)
- **Chosen for this project**
- `IndexFlatIP` — exact cosine search, no approximation
- Search 16,058 vectors in <1ms on CPU
- `IndexIVFFlat` or `IndexHNSWFlat` — approximate, needed at 100k+

### Other Options Researched

| System | Type | Strength | Weakness |
|--------|------|----------|---------|
| FAISS | Library | Fastest, free, no infra | No persistence/API out of box |
| Pinecone | Managed DB | Hosted, easy API | Paid, data leaves server |
| Weaviate | Vector DB | REST API, filtering | Heavy setup |
| pgvector | PostgreSQL ext | SQL + vectors together | Slower than FAISS |
| Chroma | Embedded DB | Easy local dev | Less battle-tested |

**Recommendation:** FAISS for self-hosted; Pinecone if managed service is acceptable.

---

## 6. Prototype Results

### What Was Built

A working end-to-end pipeline on the LFW dataset:

1. **Detect** faces with InsightFace RetinaFace
2. **Embed** each face with ArcFace (512-dim)
3. **Index** all embeddings in a FAISS IndexFlatIP
4. **Query** any photo and retrieve top-N matches by cosine similarity

### Verified Results

Query: `George_W_Bush_0001.jpg`

| Rank | Score | Match |
|------|-------|-------|
| 1 | 1.0000 | George_W_Bush_0001.jpg (exact) |
| 2 | 0.8455 | George_W_Bush_0146.jpg |
| 3 | 0.8435 | George_W_Bush_0133.jpg |
| 4 | 0.8379 | George_W_Bush_0125.jpg |
| 5 | 0.8301 | George_W_Bush_0034.jpg |

All top-5 results are correct matches — same person, different photos.  
Score of 1.0 = exact image match. Score >0.7 generally indicates same person with ArcFace.

### Performance (CPU only, no GPU)

| Metric | Value |
|--------|-------|
| Embedding throughput | 3.4 embeddings/sec |
| Total embeddings built | 16,058 |
| Index build time | ~79 minutes |
| Search latency | <5ms per query |
| Index size on disk | ~32MB |

---

## 7. Evaluation Metrics Defined

For the full evaluation phase (Day 2), these metrics will be measured:

- **LFW Verification Accuracy** — standard pairs test (6,000 pairs)
- **TAR @ FAR=0.01%** — True Accept Rate at 0.01% False Accept Rate (IJB-C metric)
- **Rank-1 Accuracy** — does the correct person appear in top result?
- **mAP** — mean Average Precision across all query identities
- **Search latency** — P50 / P95 / P99 query time
- **Throughput** — embeddings/sec during indexing

---

## 8. Day 2 Plan

- [ ] Formal LFW pairs evaluation on InsightFace vs DeepFace vs Dlib
- [ ] Test on low-resolution images (downscale LFW to 64×64, 32×32)
- [ ] Test on partially occluded faces
- [ ] Compare DeepFace backends side-by-side (ArcFace, FaceNet512, VGG-Face)
- [ ] Evaluate proprietary APIs (AWS Rekognition, Azure Face)
- [ ] Notebook with visual confusion analysis

---

## 9. Recommended Stack (Preliminary)

Based on Day 1 research:

| Layer | Choice | Reason |
|-------|--------|--------|
| Detection | RetinaFace (InsightFace) | Best accuracy + handles low-res |
| Recognition | ArcFace w600k_r50 | 99.83% LFW, 512-dim, fast |
| Vector search | FAISS IndexFlatIP → IVFFlat at scale | Free, fast, battle-tested |
| Image storage | AWS S3 (or local NAS) | Cheap, scalable |
| Backend | Python + FastAPI | Async, easy to containerise |
| Queue | Celery + Redis | Background embedding jobs |

This will be finalised after Day 2–3 evaluation and cost analysis.