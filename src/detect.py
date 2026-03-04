# src/detect.py
import cv2
import time
import numpy as np
from pathlib import Path
from PIL import Image

def benchmark_detectors(image_folder: str, max_images: int = 50):
    """
    Compare MTCNN vs InsightFace (RetinaFace) vs OpenCV Haar
    on a folder of images.
    """
    results = {}
    image_paths = list(Path(image_folder).rglob("*.jpg"))[:max_images]
    
    if not image_paths:
        print(f"No images found in {image_folder}")
        return

    print(f"Found {len(image_paths)} images to test\n")

    # ── 1. OpenCV Haar Cascade (baseline, fast but weak) ──────────────────
    print("Testing OpenCV Haar Cascade...")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    total_faces, t0 = 0, time.time()
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        total_faces += len(faces)
    haar_time = time.time() - t0
    results["OpenCV Haar"] = {
        "faces_detected": total_faces,
        "time_sec": round(haar_time, 2),
        "fps": round(len(image_paths) / haar_time, 1),
    }
    print(f"  ✓ {total_faces} faces in {haar_time:.2f}s\n")

    # ── 2. MTCNN ──────────────────────────────────────────────────────────
    try:
        from mtcnn import MTCNN
        print("Testing MTCNN...")
        detector = MTCNN()
        total_faces, t0 = 0, time.time()
        for p in image_paths:
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(img)
            total_faces += len(faces)
        mt_time = time.time() - t0
        results["MTCNN"] = {
            "faces_detected": total_faces,
            "time_sec": round(mt_time, 2),
            "fps": round(len(image_paths) / mt_time, 1),
        }
        print(f"  ✓ {total_faces} faces in {mt_time:.2f}s\n")
    except ImportError:
        print("  ⚠ MTCNN not installed (pip install mtcnn)\n")

    # ── 3. InsightFace RetinaFace ─────────────────────────────────────────
    try:
        import insightface
        from insightface.app import FaceAnalysis
        print("Testing InsightFace (RetinaFace)...")
        app = FaceAnalysis(allowed_modules=["detection"], providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        total_faces, t0 = 0, time.time()
        for p in image_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            faces = app.get(img)
            total_faces += len(faces)
        rf_time = time.time() - t0
        results["InsightFace (RetinaFace)"] = {
            "faces_detected": total_faces,
            "time_sec": round(rf_time, 2),
            "fps": round(len(image_paths) / rf_time, 1),
        }
        print(f"  ✓ {total_faces} faces in {rf_time:.2f}s\n")
    except Exception as e:
        print(f"  ⚠ InsightFace error: {e}\n")

    # ── Print Summary ─────────────────────────────────────────────────────
    print("=" * 55)
    print(f"{'Detector':<30} {'Faces':>8} {'Time(s)':>8} {'FPS':>6}")
    print("=" * 55)
    for name, r in results.items():
        print(f"{name:<30} {r['faces_detected']:>8} {r['time_sec']:>8} {r['fps']:>6}")
    print("=" * 55)
    return results


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "datasets/lfw"
    benchmark_detectors(folder)