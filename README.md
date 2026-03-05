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
│   └── day2/                           ← Day 2 evaluation scripts
│       ├── run_all.py                  ← Master runner (start here)
│       ├── eval_lfw_pairs.py           ← Task 1: LFW pairs benchmark
│       ├── eval_lowres.py              ← Task 2: low-res degradation test
│       ├── eval_occlusion.py           ← Task 3: occlusion robustness
│       ├── eval_deepface_backends.py   ← Task 4: DeepFace model comparison
│       ├── eval_proprietary_apis.py    ← Task 5: AWS + Azure eval
│       ├── collect_results.py          ← Merges all JSONs → day2_full_results.json
│       ├── README.md                   ← Day 2 run instructions
│       └── results/                    ← All Day 2 JSON outputs land here
│           ├── lfw_pairs_results.json
│           ├── lowres_results.json
│           ├── occlusion_results.json
│           ├── deepface_backends.json
│           ├── proprietary_api_results.json
│           └── day2_full_results.json  ← Merged output (paste to Claude for report)
│
├── docs/
│   ├── day1_research.md                ← Day 1: algorithm research & findings
│   └── day2_research.md                ← Day 2: framework evaluation report
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

### 5. Run full pipeline demo

```bash
PERSON=$(ls datasets/lfw_home/lfw_funneled/ | head -1)
QUERY=$(ls datasets/lfw_home/lfw_funneled/$PERSON/*.jpg | head -1)
python3 demo.py datasets/lfw_home/lfw_funneled "$QUERY"
```

### 6. Run Day 2 evaluation suite

```bash
# From project root:
python3 src/day2/run_all.py            # full run (~90–170 min)
python3 src/day2/run_all.py --quick    # reduced pairs (~30 min)

# Results land in src/day2/results/day2_full_results.json
```

### 7. Run algorithm comparison + generate charts

```bash
python3 src/compare_algorithms.py
# → outputs/algorithm_comparison.png
# → outputs/algorithm_comparison.csv
```

---

##  Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Face Detection | InsightFace (RetinaFace) | Best accuracy + speed on CPU |
| Face Embedding | ArcFace (w600k_r50) | 99.83% LFW accuracy, 512-dim |
| Vector Search | FAISS (IndexFlatIP) | Fastest CPU vector search |
| Dataset | LFW Funneled | Standard benchmark, 13k images |
| Runtime | Python 3.12 + ONNX CPU | No GPU required for prototype |

---

##  Key Results

### Day 1
- **16,058** face embeddings extracted from LFW dataset
- **3.4 embeddings/sec** on CPU (no GPU)
- **Cosine similarity score of 1.000** for exact match (same image)
- **0.84+ similarity** for other photos of the same person
- Index size: ~32MB on disk (512 × 16058 × 4 bytes)

### Day 2
- **DeepFace Facenet512** — best measured accuracy at **98.67%** on LFW pairs
- **DeepFace ArcFace** — fastest DeepFace backend at **1.67 pairs/sec**
- **Low-res robustness** — accuracy degrades from 90.67% → 85.33% at 64×64px (−5.3%)
- **Occlusion** — 100% detection rate maintained across eyes, lower face, and random patch blocks
- **InsightFace** — detection blocked by in-memory image pipeline issue (see docs/day2_research.md §8.1)

---

##  Research Roadmap

| Day | Focus | Status |
|-----|-------|--------|
| 1 | Algorithm research + prototype pipeline |  Done |
| 2 | Framework evaluation (InsightFace vs DeepFace vs Dlib) |  Done |
| 3 | Storage strategy + vector DB research |  Upcoming |
| 4 | Cost estimation + final recommendations |  Upcoming |

---

##  Documentation

See `docs/` for:
- `day1_research.md` — Algorithm comparison, findings, and architecture decisions
- `day2_research.md` — Framework evaluation, low-res + occlusion test results

---

##  Dataset

**LFW (Labeled Faces in the Wild)** — University of Massachusetts
- 13,233 images of 5,749 people
- Collected from news articles
- Standard benchmark for face verification
- [Official site](http://vis-www.cs.umass.edu/lfw/)

---