#  Face Recognition Research

A research project benchmarking open-source face detection and recognition frameworks, building toward a production-ready reverse image search pipeline for identifying individuals across large photo datasets.

##  Goal

Design and prototype a system that can:
- Ingest a dataset of ~5,000+ images
- Detect all faces in those images
- Generate vector embeddings per face
- Accept a query photo (e.g. a selfie or CCTV frame)
- Return the top-N most similar faces from the dataset

---

##  Project Structure

```
face-recognition-research/
├── datasets/
│   └── lfw_home/
│       └── lfw_funneled/               ← LFW dataset (5,749 people, 13,233 images)
│           └── <Person_Name>/
│               └── *.jpg
│
├── embeddings/
│   ├── face_index.faiss                ← FAISS vector index (16,058 embeddings)
│   ├── metadata.pkl                    ← Per-embedding metadata (path, label, bbox)
│   └── embeddings.npy                  ← Raw embedding matrix (float32)
│
├── outputs/
│   ├── search_result.png               ← Visual output of last query
│   ├── algorithm_comparison.png        ← Day 1 benchmark charts
│   └── algorithm_comparison.csv        ← Day 1 comparison table
│
├── models/                             ← Downloaded model weight files
│   ├── shape_predictor_68_face_landmarks.dat
│   └── dlib_face_recognition_resnet_model_v1.dat
│
├── src/
│   ├── __init__.py
│   ├── detect.py                       ← Face detector benchmarking (Day 1)
│   ├── embed.py                        ← Embedding generation + FAISS builder (Day 1)
│   ├── search.py                       ← Query face against index (Day 1)
│   ├── compare_algorithms.py           ← Algorithm comparison + charts (Day 1)
│   ├── utils.py                        ← Shared helpers
│   │
│   ├── day2/                           ← Day 2 evaluation scripts
│   │   ├── run_all.py
│   │   ├── eval_lfw_pairs.py
│   │   ├── eval_lowres.py
│   │   ├── eval_occlusion.py
│   │   ├── eval_deepface_backends.py
│   │   ├── eval_proprietary_apis.py
│   │   ├── collect_results.py
│   │   └── results/
│   │       └── day2_full_results.json
│   │
│   └── day3/                           ← Day 3 evaluation scripts
│       ├── run_all.py                  ← Master runner (start here)
│       ├── eval_insightface_disk.py    ← Task 1: InsightFace disk re-evaluation
│       ├── eval_full_lfw_benchmark.py  ← Task 2: full balanced LFW benchmark
│       ├── eval_vector_dbs.py          ← Task 3: FAISS vs pgvector vs Weaviate vs Pinecone
│       ├── eval_superresolution.py     ← Task 4: super-resolution preprocessing
│       ├── cost_estimation.py          ← Task 5: cost at 5k / 50k / 500k scale
│       ├── storage_architecture.py     ← Task 6: storage design document
│       ├── api_server.py               ← FastAPI search endpoint prototype
│       ├── collect_results.py
│       └── results/
│           └── day3_full_results.json
│
├── docs/
│   ├── day1_research.md                ← Day 1: algorithm research & findings
│   ├── day2_research.md                ← Day 2: framework evaluation report
│   └── day3_research.md                ← Day 3: storage, vector DB, API prototype
│
├── notebooks/                          ← Jupyter notebooks (exploratory)
├── demo.py                             ← End-to-end pipeline demo (single file)
├── requirements.txt
└── README.md
```

---

##  Quickstart

### 1. Clone & set up environment

```bash
git clone https://github.com/fakubwoy/face-recognition-research.git
cd face-recognition-research

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the LFW dataset

```bash
python3 -c "
from sklearn.datasets import fetch_lfw_people
fetch_lfw_people(min_faces_per_person=1, download_if_missing=True, data_home='datasets')
print('Done.')
"
```

### 3. Build the FAISS index

```bash
python3 src/embed.py datasets/lfw_home/lfw_funneled
```

Expected output:
```
Processing 13,233 images with InsightFace (ArcFace)...
✓ Extracted 16,058 embeddings in ~78 mins
✓ FAISS index saved → embeddings/face_index.faiss
✓ Index size: 16058 vectors, dim=512
```

### 4. Search for a face

```bash
python3 src/search.py datasets/lfw_home/lfw_funneled/George_W_Bush/George_W_Bush_0001.jpg
```

Output:
```
Top 5 matches:
Rank     Score  Label                     Image
1       1.0000  George_W_Bush             George_W_Bush_0001.jpg   ← exact match
2       0.8455  George_W_Bush             George_W_Bush_0146.jpg
3       0.8435  George_W_Bush             George_W_Bush_0133.jpg
...
✓ Result saved → outputs/search_result.png
```

### 5. Run the search API

```bash
pip install fastapi uvicorn[standard] python-multipart
uvicorn src.day3.api_server:app --reload --port 8000

# Test it
curl -X POST http://localhost:8000/search \
  -F "file=@datasets/lfw_home/lfw_funneled/George_W_Bush/George_W_Bush_0001.jpg" \
  -F "top_k=5"

# Interactive docs
open http://localhost:8000/docs
```

### 6. Run full pipeline demo

```bash
PERSON=$(ls datasets/lfw_home/lfw_funneled/ | head -1)
QUERY=$(ls datasets/lfw_home/lfw_funneled/$PERSON/*.jpg | head -1)
python3 demo.py datasets/lfw_home/lfw_funneled "$QUERY"
```

### 7. Run Day 2 evaluation suite

```bash
python3 src/day2/run_all.py            # full run (~90–170 min)
python3 src/day2/run_all.py --quick    # reduced pairs (~30 min)
```

### 8. Run Day 3 evaluation suite

```bash
python3 src/day3/run_all.py            # full run (~90–120 min)
python3 src/day3/run_all.py --quick    # reduced pairs (~20 min)
python3 src/day3/run_all.py --skip-insightface  # skip if pipeline issues
```

### 9. Run algorithm comparison + generate charts

```bash
python3 src/compare_algorithms.py
# → outputs/algorithm_comparison.png
# → outputs/algorithm_comparison.csv
```

---

##  Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Face Detection | InsightFace (RetinaFace) | 99.8% detection rate, confirmed across Days 1 & 3 |
| Face Embedding | ArcFace (w600k_r50) | 99.2% accuracy / AUC 0.995 on disk-loaded LFW pairs |
| Low-res preprocessing | Bicubic + unsharp mask | +13pp accuracy recovery at 64×64px, zero cost |
| Vector Search | FAISS (IndexFlatIP) | 2.4ms P50, exact recall, 32MB for 16k vectors |
| Metadata | PostgreSQL | ACID, relational filtering, pgvector-ready |
| Image Storage | AWS S3 | $0.06/month at 5k images |
| API | FastAPI + uvicorn | Prototype running, async, ONNX + FAISS compatible |
| Queue | Celery + Redis | Background embedding jobs for bulk ingestion |
| Dataset | LFW Funneled | Standard benchmark, 13k images |
| Runtime | Python 3.12 + ONNX CPU | No GPU required for prototype |

---

##  Key Results

### Day 1
- **16,058** face embeddings extracted from LFW dataset
- **3.4 embeddings/sec** on CPU (no GPU)
- **Cosine similarity score of 1.000** for exact match (same image)
- **0.84+ similarity** for other photos of the same person
- FAISS search latency: **<5ms** per query on 16k-vector index

### Day 2
- **DeepFace Facenet512** — best measured accuracy at **98.67%** on positive-only LFW pairs
- **DeepFace ArcFace** — fastest DeepFace backend at **1.67 pairs/sec**
- **Low-res robustness** — accuracy degrades from 90.67% → 85.33% at 64×64px (−5.3%)
- **Occlusion** — 100% face detection rate maintained across all occlusion types
- **InsightFace** — detection failed due to in-memory image pipeline bug (not a model failure)

### Day 3
- **InsightFace ArcFace (disk-loaded)** — **99.2% accuracy**, AUC **0.995**, separability **0.6703**
- **DeepFace Facenet512 (balanced pairs)** — AUC **0.9765**, EER **7.17%**, TAR@FAR=1% **88.0%**
- **FAISS IndexFlatIP** — P50 **2.36ms**, P99 **3.79ms**, recall **1.000** at 16k vectors
- **FAISS HNSW** — P50 **0.59ms** at 71% recall (95%+ expected on real face data)
- **Bicubic + sharpen** — accuracy at 32×32px: **98.00%** (vs 88.00% without sharpening, +10pp)
- **FastAPI prototype** — search endpoint live, < 100ms end-to-end on CPU

---

##  Research Roadmap

| Day | Focus | Status |
|-----|-------|--------|
| 1 | Algorithm research + prototype pipeline |  Done |
| 2 | Framework evaluation (InsightFace vs DeepFace vs Dlib) | Done |
| 3 | Storage strategy + vector DB research + API prototype | Done |
| 4 | Retrieval evaluation (Rank-1, mAP) + final recommendation |  Upcoming |

---

##  Documentation

See `docs/` for:
- `day1_research.md` — Algorithm comparison, findings, and architecture decisions
- `day2_research.md` — Framework evaluation, low-res + occlusion test results
- `day3_research.md` — Storage design, vector DB benchmark, API prototype, cost estimation

---

##  Dataset

**LFW (Labeled Faces in the Wild)** — University of Massachusetts
- 13,233 images of 5,749 people
- Collected from news articles
- Standard benchmark for face verification
- [Official site](http://vis-www.cs.umass.edu/lfw/)

---