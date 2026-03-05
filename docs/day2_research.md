# Day 2 Research — Framework Evaluation

**Date:** 2026-03-05
**Environment:** WSL2 / Ubuntu — Intel Xeon E-2186M @ 2.90GHz, 12 cores, 12.5GB RAM, CPU-only

---

## 1. Objectives

Day 2 moved from research to measurement. The goal was to run each framework against real data and produce accuracy, speed, and robustness numbers that can drive the final production stack decision.

### Tasks Completed

| # | Task | Pairs | Status |
|---|------|-------|--------|
| 1 | LFW pairs formal benchmark — InsightFace, DeepFace, Dlib | 300 |  Done |
| 2 | Low-resolution stress test — original, 128px, 64px, 32px | 150 |  Done |
| 3 | Occlusion robustness — eyes, lower face, random patch, hat | 100 |  Done |
| 4 | DeepFace backend deep comparison — ArcFace, Facenet512, VGG-Face, OpenFace | 80 |  Done |
| 5 | Proprietary API assessment — AWS Rekognition, Azure Face | N/A |  Reviewed |

---

## 2. LFW Pairs Formal Evaluation

Each framework was given the same 300 same-person LFW image pairs. Accuracy is reported at the optimal decision threshold found by sweeping 0.10–0.99.

### Results

| Framework | Pairs Tested | Accuracy | Best Threshold | Pairs/s | Time (s) |
|-----------|-------------|----------|---------------|---------|----------|
| InsightFace (ArcFace) | 0 * | N/A | — | — | — |
| DeepFace (ArcFace) | 300 | 94.00% | 0.10 | 1.67 | 179.4 |
| DeepFace (Facenet512) | 300 | **98.67%** | 0.10 | 1.28 | 234.7 |
| DeepFace (VGG-Face) | 300 | 98.33% | 0.10 | 0.78 | 387.0 |
| Dlib (ResNet-128) | 256 / 300 † | 100% † | 0.10 | 5.01 | 51.2 |

> `*` **InsightFace** skipped all 300 pairs. Face detection returns empty results when images are passed as in-memory numpy arrays from sklearn's pixel data. This is a data-pipeline issue specific to this test harness — **not a model failure**. Day 1 confirmed InsightFace works correctly from JPEG files on disk (16,058 embeddings extracted successfully).

> † **Dlib** skipped 44 pairs (14.7%) where it failed to detect a face. The 100% accuracy figure is inflated because the test set contained only same-person pairs. Real-world LFW accuracy for Dlib is ~99.38% per published benchmarks. AUC-ROC could not be computed (single class).

### Speed Comparison

| Framework | Pairs/s | Note |
|-----------|---------|------|
| Dlib (ResNet-128) | 5.01 | Fastest — lightweight 22MB model, no TF overhead |
| DeepFace (ArcFace) | 1.67 | Best speed among DeepFace backends |
| DeepFace (Facenet512) | 1.28 | Best accuracy/speed tradeoff overall |
| DeepFace (VGG-Face) | 0.78 | Slowest — 4,096-dim embeddings, 580MB model |

### Analysis

Facenet512 via DeepFace achieved the highest measured accuracy at **98.67%** on 300 same-person pairs, processing at 1.28 pairs/sec. VGG-Face matched near this accuracy (98.33%) at less than half the speed, with a 580MB model footprint that also triggered TensorFlow memory warnings on the 12.5GB test machine — making it impractical for production.

DeepFace (ArcFace) came in at 94%, likely influenced by the positive-only test set affecting threshold selection. Published accuracy for InsightFace ArcFace on the full 6,000-pair benchmark is 99.83%, which remains the target for Day 3 re-evaluation.

Dlib's 5.01 pairs/sec makes it the fastest tested, but its 128-dimensional embedding limits representational capacity compared to 512-dim models. It is a viable option only for low-accuracy-requirement, resource-constrained deployments.

---

## 3. Low-Resolution Stress Test

Each image was downscaled to the target resolution, then upscaled back to original dimensions before passing to the model — matching how a genuinely low-res capture looks when presented to a detector.

### DeepFace (ArcFace) — Accuracy vs Resolution

| Resolution | Pairs Tested | Accuracy | Detection Rate | Time (s) |
|------------|-------------|----------|---------------|----------|
| Original (~125px) | 150 | 90.67% | 100% | 55.9 |
| 128 × 128 px | 150 | 88.00% | 100% | 54.1 |
| 64 × 64 px | 150 | 85.33% | 100% | 54.2 |
| 32 × 32 px | 150 | 86.67% | 100% | 54.1 |

### InsightFace Low-Resolution

InsightFace returned 0% detection across all resolutions, for the same in-memory image pipeline reason described in Section 2. This does not reflect InsightFace's real capability — published benchmarks show RetinaFace handles faces as small as 20×20 pixels, which is its primary advantage for CCTV use cases.

### Analysis

DeepFace (ArcFace) maintained a **100% face detection rate** across all resolutions — OpenCV's detector is robust to upscaled low-res images. Accuracy degrades gradually: from 90.67% at original quality down to **85.33% at 64×64**, a drop of 5.3 percentage points. The slight uptick at 32×32 (86.67%) is within variance at this sample size.

For CCTV scenarios where faces are genuinely 32×32px or smaller at source, an image super-resolution pre-processing step (e.g. ESRGAN or Real-ESRGAN) should be evaluated before embedding. This will be explored in Day 3.

---

## 4. Occlusion Robustness Test

Both images in each pair had the same occlusion applied. The five occlusion types were:

| Occlusion | Description |
|-----------|-------------|
| None (baseline) | Unmodified image pair |
| Lower Face | Bottom 45% blacked out — simulates face mask / scarf |
| Eyes | Eye band (25–50% of height) blacked out — simulates sunglasses |
| Random Patch | Random 40×40% block anywhere on the face |
| Hat / Forehead | Top 30% blacked out — simulates hat or low camera angle |

### DeepFace (ArcFace) — Occlusion Results

| Occlusion | Pairs | Accuracy | Detection Rate | vs Baseline |
|-----------|-------|----------|---------------|-------------|
| None (baseline) | 100 | 93.00% | 100% | — |
| Lower Face | 100 | 100.00% | 100% | +7.0% |
| Eyes | 100 | 100.00% | 100% | +7.0% |
| Random Patch | 100 | 97.00% | 100% | +4.0% |
| Hat / Forehead | 100 | 98.00% | 100% | +5.0% |

### InsightFace Occlusion

Same 0% detection issue as above — not usable data from this run.

### Analysis

Accuracy improved under occlusion versus the unoccluded baseline, which is counterintuitive but explainable: the test set contained **only same-person pairs**. When occlusion removes features like eyes or mouth, the model relies on remaining consistent facial structure, reducing the noise that causes false negatives near the decision boundary. With a balanced positive/negative test set, occlusion would show the expected accuracy degradation.

The **100% detection rate** across all occlusion types is a meaningful result — DeepFace's OpenCV detector continues to find faces even when substantial regions are blacked out, relying on overall facial structure and skin regions.

The critical production risk here is **false positives under occlusion**, not false negatives. An occluded face may be incorrectly matched to a different person more easily. This requires testing with negative pairs, which will be addressed in the Day 3 full balanced run.

---

## 5. DeepFace Backend Deep Comparison

All four DeepFace-supported models were tested on the same 80 same-person pairs, providing a direct side-by-side breakdown of embedding quality and speed.

### Accuracy & Speed

| Model | Pairs | Accuracy | s/pair | Model Size |
|-------|-------|----------|--------|------------|
| ArcFace | 80 | 93.75% | 0.384 | 137MB |
| Facenet512 | 80 | 100% * | 0.529 | 95MB |
| VGG-Face | 80 | 97.50% | 0.411 | 580MB |
| OpenFace | 80 | 100% * | 0.288 | 15MB |

> \* Inflated — same-person only test set. Published LFW accuracy: Facenet512 ~99.65%, OpenFace ~92%.

### Similarity Score Distributions (Same-Person Pairs)

A higher mean with lower standard deviation is preferable — it indicates the model is confident and consistent when matching genuine pairs.

| Model | Mean Sim | Std Dev | Min | Max |
|-------|----------|---------|-----|-----|
| OpenFace | 0.650 | 0.145 | 0.299 | 0.953 |
| Facenet512 | 0.638 | 0.184 | 0.128 | 0.913 |
| ArcFace | 0.508 | 0.189 | 0.022 | 0.821 |
| VGG-Face | 0.477 | 0.174 | 0.084 | 0.843 |

### Analysis

**Facenet512** stands out as the best overall option based on measured data: 98.67% accuracy on Task 1's 300-pair run, mean similarity of 0.638, and a reasonable 95MB model size. Its separability score will be measurable once negative pairs are included.

**OpenFace** has the tightest same-person clustering (mean 0.650, std 0.145) and is the fastest at 0.288s/pair with only a 15MB footprint. However its published accuracy of ~92% is too low for production use in an identification context. It's a candidate only for latency-critical demos where false positives are acceptable.

**VGG-Face** delivers good accuracy but at a 580MB model size, 4,096-dimensional embeddings, and the slowest speed. The memory warnings on a 12.5GB machine are a practical deployment concern. It is not recommended.

**ArcFace via DeepFace** showed lower measured accuracy in this run (93.75%), but this is partly a threshold calibration artifact of the positive-only test set. Published InsightFace ArcFace accuracy (99.83%) significantly outperforms what was measurable here.

---

## 6. Proprietary API Assessment

AWS Rekognition and Azure Face API were not benchmarked with live credentials. The following comparison is based on published specifications and pricing documentation.

| | AWS Rekognition | Azure Face API |
|---|---|---|
| Published LFW Accuracy | ~99.5% | ~99.5% |
| Free Tier | 5,000 images/month (12 months) | 30,000 transactions/month (ongoing) |
| Paid Pricing | $1.00 per 1,000 images | $0.001 per transaction |
| Cost @ 5k images | ~$5.00 (after free tier) | Free (within free tier) |
| Cost @ 100k/month | ~$100/month | ~$70/month |
| Embedding Access |  No |  No |
| FAISS Compatible |  No |  No |
| Data Privacy | Images sent to AWS | Images sent to Azure |
| Setup | boto3 + IAM | REST API + subscription key |

### Key Limitations

- **No embedding access** — you cannot extract face vectors, so you cannot build a custom FAISS index or run similarity search at scale.
- **Data privacy** — every query image is transmitted to a third-party server. Significant constraint for sensitive individuals or regulated data.
- **Vendor lock-in** — switching providers requires re-engineering the entire search layer.
- **Cost at scale** — at 100k+ images/month, both APIs cost $70–100/month versus $0 for self-hosted InsightFace + FAISS.

### When They Make Sense

Managed APIs suit low-volume, rapid-prototype use cases where setup time matters more than cost, or regulated environments requiring vendor-provided compliance certification. For a custom reverse image search over a self-hosted dataset, they are not suitable.

---

## 7. Issues & Root Causes

### 7.1 InsightFace Detection Failure

InsightFace returned zero face detections across all tasks. The root cause is that sklearn's `fetch_lfw_pairs()` returns float32 numpy arrays normalised to `[0, 1]` in RGB channel order. InsightFace expects BGR uint8 arrays (OpenCV format). The `cv2.cvtColor + uint8` cast was applied in the scripts, but an upstream issue with the image data shape or pixel range prevented the detector from triggering.

**This does not reflect InsightFace's real-world performance.** Day 1 confirmed it successfully processed 13,233 JPEG files and produced 16,058 embeddings. For Day 3, all InsightFace testing should use disk-loaded JPEG images directly rather than sklearn in-memory arrays.

### 7.2 Positive-Only Test Set (AUC-ROC = NaN)

`fetch_lfw_pairs()` in quick mode returned 300 same-person pairs and 0 different-person pairs. AUC-ROC requires both positive and negative samples, so it could not be computed for any framework. Accuracy measurements reflect only the **true-positive rate**, not full verification performance. Day 3 should run without `--quick` using the standard 6,000-pair test split.

### 7.3 VGG-Face Memory Warnings

VGG-Face's weight allocations (4× ~411MB blocks) triggered TensorFlow's memory warning threshold on the 12.5GB test machine. On machines with less RAM this could cause OOM errors. Further reason to exclude VGG-Face from the production shortlist.

---

## 8. Updated Metrics Status

Metrics defined in Day 1 and their current status:

| Metric | Status | Notes |
|--------|--------|-------|
| LFW Verification Accuracy |  Partial | Positive-only pairs; full run needed |
| TAR @ FAR=0.01% |  Not yet | Requires balanced pairs + ROC curve |
| Rank-1 Accuracy |  Not yet | Requires retrieval evaluation (Day 3) |
| mAP |  Not yet | Requires multiple query identities |
| Search latency P50/P95/P99 |  Day 1 | < 5ms per query on 16k index |
| Embedding throughput |  Day 1 | 3.4 embeddings/sec (InsightFace, CPU) |

---

## 9. Day 3 Plan

- [ ] Re-run InsightFace evaluation using disk-loaded JPEG images (fix the pipeline issue)
- [ ] Run full 6,000-pair LFW test without `--quick` to get balanced accuracy, AUC-ROC, and TAR@FAR
- [ ] Research and compare vector database options: FAISS vs Pinecone vs pgvector vs Weaviate
- [ ] Design full storage architecture: image store, embedding store, metadata store
- [ ] Prototype the FastAPI search endpoint using the existing FAISS index from Day 1
- [ ] Evaluate image super-resolution preprocessing for low-res face improvement

---

## 10. Recommended Stack (Updated)

Based on Day 1 + Day 2 combined findings:

| Layer | Choice | Reasoning |
|-------|--------|-----------|
| Detection | RetinaFace (InsightFace) | Day 1 confirmed. Retest from disk in Day 3. |
| Embedding | ArcFace (InsightFace) | 99.83% published LFW. Retest from disk in Day 3. |
| Embedding (fallback) | DeepFace Facenet512 | 98.67% measured. Fallback if InsightFace pipeline issues persist. |
| Vector search | FAISS IndexFlatIP | Sub-5ms search confirmed Day 1. Scale to IVFFlat at 100k+. |
| Image storage | AWS S3 or local NAS | ~$0.60/month for 25GB dataset on S3. |
| Backend | Python + FastAPI | Async, containerisable, works with ONNX + FAISS. |
| Queue | Celery + Redis | Background embedding jobs for bulk ingestion. |

The primary outstanding question is InsightFace's real performance on a disk-based evaluation run, which will be resolved in Day 3.