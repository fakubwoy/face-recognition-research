"""
Drop this in src/day4/ and run it.
Diagnoses why InsightFace skips all pairs in eval_final_benchmark.py
"""
import tempfile, os, cv2, numpy as np

# ── reproduce exactly what load_balanced_pairs() + _write_reload() does ──────
from sklearn.datasets import fetch_lfw_pairs
import tempfile, os

LFW_DATA_HOME = "../../datasets"
data = fetch_lfw_pairs(subset="test", data_home=LFW_DATA_HOME,
                       download_if_missing=False, resize=1.0, color=True)

# grab one same-person pair
img_array = data.pairs[0, 0]   # shape (H, W, 3), dtype float64, range [0,1]
print(f"Raw sklearn array: shape={img_array.shape}, dtype={img_array.dtype}, "
      f"min={img_array.min():.3f}, max={img_array.max():.3f}")

# step 1: convert to uint8 BGR  (what load_balanced_pairs does)
img_uint8_rgb = (img_array * 255).astype(np.uint8)
img_bgr       = cv2.cvtColor(img_uint8_rgb, cv2.COLOR_RGB2BGR)
print(f"After RGB->BGR uint8: shape={img_bgr.shape}, dtype={img_bgr.dtype}, "
      f"min={img_bgr.min()}, max={img_bgr.max()}")

# step 2: _write_reload  (mkstemp version)
fd, path = tempfile.mkstemp(suffix=".jpg")
os.close(fd)
ok = cv2.imwrite(path, img_bgr)
print(f"cv2.imwrite ok={ok}, file_size={os.path.getsize(path)} bytes")
reloaded = cv2.imread(path)
os.unlink(path)
print(f"cv2.imread result: {reloaded.shape if reloaded is not None else 'None'}")

# step 3: run InsightFace on the reloaded image
from insightface.app import FaceAnalysis
app = FaceAnalysis(providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

faces = app.get(reloaded)
print(f"\nInsightFace detected {len(faces)} face(s) in reloaded image")

# step 4: also try different det_sizes
for sz in [(320, 320), (480, 480), (640, 640)]:
    app2 = FaceAnalysis(providers=["CPUExecutionProvider"])
    app2.prepare(ctx_id=-1, det_size=sz)
    f = app2.get(reloaded)
    print(f"  det_size={sz}: {len(f)} faces")

# step 5: check actual image dimensions
print(f"\nImage dimensions: {reloaded.shape}  (H x W x C)")
print("Note: LFW funneled images are 250x250. InsightFace det_size=(640,640)")
print("should work fine for 250x250 images. Checking if image is too small...")

# step 6: try without resizing (raw image)
faces_raw = app.get(img_bgr)
print(f"\nInsightFace on in-memory BGR (no disk round-trip): {len(faces_raw)} faces")

# step 7: check what resize=1.0 actually gives us
print(f"\nSklearn image shape at resize=1.0: {img_array.shape}")
# fetch again with resize=0.5 to see default
data2 = fetch_lfw_pairs(subset="test", data_home=LFW_DATA_HOME,
                        download_if_missing=False, resize=0.5, color=True)
img2 = data2.pairs[0, 0]
print(f"Sklearn image shape at resize=0.5: {img2.shape}")