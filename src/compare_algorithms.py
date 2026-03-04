# src/compare_algorithms.py
"""
Day 1 Research Output: Benchmark + compare all face recognition approaches.
Generates a summary table and charts.
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import cv2
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import roc_auc_score

def run_deepface_comparison(image_folder: str, max_pairs: int = 30):
    """
    Compare DeepFace backends: VGG-Face, ArcFace, Facenet on same pairs.
    """
    from deepface import DeepFace
    
    image_paths = list(Path(image_folder).rglob("*.jpg"))[:60]
    if len(image_paths) < 2:
        print("Not enough images for DeepFace comparison")
        return {}

    models = ["VGG-Face", "ArcFace", "Facenet"]
    results = {}

    for model in models:
        print(f"  Testing DeepFace model: {model}...")
        correct, total, t0 = 0, 0, time.time()
        errors = 0

        for i in range(min(max_pairs, len(image_paths) - 1)):
            p1 = str(image_paths[i])
            p2 = str(image_paths[i + 1])
            same_person = image_paths[i].parent == image_paths[i + 1].parent
            try:
                r = DeepFace.verify(p1, p2, model_name=model,
                                    detector_backend="opencv",
                                    enforce_detection=False, silent=True)
                pred_same = not r["verified"]  # flip: verify=True means same
                # Actually verify=True means same person
                pred_same = r["verified"]
                if pred_same == same_person:
                    correct += 1
                total += 1
            except Exception:
                errors += 1

        elapsed = time.time() - t0
        acc = correct / total if total > 0 else 0
        results[model] = {
            "accuracy": round(acc, 3),
            "time_sec": round(elapsed, 2),
            "pairs_tested": total,
            "errors": errors,
        }
        print(f"    ✓ acc={acc:.3f}, time={elapsed:.1f}s")

    return results


def generate_comparison_table():
    """
    Static research-based comparison table (Day 1 output).
    Based on published benchmarks (LFW, IJB-C etc.)
    """
    data = {
        "Solution": [
            "InsightFace (ArcFace)", "DeepFace (ArcFace)",
            "DeepFace (Facenet512)", "DeepFace (VGG-Face)",
            "Dlib ResNet", "face_recognition lib",
            "AWS Rekognition", "Azure Face API"
        ],
        "LFW Accuracy": ["99.83%", "99.64%", "99.65%", "98.78%",
                          "99.38%", "99.38%", "~99.5%", "~99.5%"],
        "Speed (CPU)": ["Fast", "Slow", "Slow", "Slow",
                         "Medium", "Medium", "API", "API"],
        "Embedding Dim": [512, 512, 512, 4096, 128, 128, "N/A", "N/A"],
        "Cost": ["Free", "Free", "Free", "Free",
                  "Free", "Free", "Paid", "Paid"],
        "Ease of Use": ["Medium", "Easy", "Easy", "Easy",
                         "Medium", "Very Easy", "Easy", "Easy"],
        "Low-Res Perf": ["★★★★★", "★★★★", "★★★★", "★★★",
                          "★★★", "★★★", "★★★★", "★★★★"],
        "Recommended": ["✅ YES", "⚡ OK", "⚡ OK", "❌ No",
                         "⚠️ OK", "❌ No", "💰 Costly", "💰 Costly"],
    }
    return pd.DataFrame(data)


def generate_charts(df: pd.DataFrame, output_dir: str = "outputs"):
    """
    Create visual comparison charts for the research document.
    """
    Path(output_dir).mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Face Recognition Algorithm Comparison — Day 1 Research",
                 fontsize=14, fontweight="bold", y=1.02)

    open_source = df[~df["Solution"].str.contains("AWS|Azure")]
    names = open_source["Solution"].str.replace("InsightFace ", "")
    acc_str = open_source["LFW Accuracy"].str.replace("%", "").str.replace("~", "")
    acc_vals = pd.to_numeric(acc_str, errors="coerce")

    colors = ["#2ecc71" if "ArcFace" in n or "InsightFace" in n else "#3498db"
              for n in open_source["Solution"]]

    # Chart 1: Accuracy
    ax = axes[0]
    bars = ax.bar(range(len(names)), acc_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("LFW Accuracy (%)")
    ax.set_title("Accuracy on LFW Benchmark")
    ax.set_ylim(97, 100.2)
    ax.axhline(y=99.5, color="red", linestyle="--", alpha=0.5, label="99.5% threshold")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, acc_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.2f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")

    # Chart 2: Speed (qualitative)
    speed_map = {"Fast": 3, "Medium": 2, "Slow": 1, "API": 2.5}
    speed_vals = open_source["Speed (CPU)"].map(speed_map)
    ax2 = axes[1]
    bars2 = ax2.barh(range(len(names)), speed_vals, color=colors, edgecolor="white")
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel("Relative Speed (CPU)")
    ax2.set_title("Processing Speed (CPU)")
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(["Slow", "Medium", "Fast"])

    # Chart 3: Spider/Radar - InsightFace vs DeepFace
    categories = ["Accuracy", "Speed", "Low-Res", "Ease of Use", "Scalability"]
    insight_scores = [5, 4, 5, 3, 5]
    deepface_scores = [4, 2, 4, 5, 3]
    dlib_scores = [3, 3, 3, 4, 3]

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax3 = axes[2]
    ax3 = plt.subplot(1, 3, 3, polar=True)
    
    for scores, label, color in [
        (insight_scores, "InsightFace", "#2ecc71"),
        (deepface_scores, "DeepFace", "#3498db"),
        (dlib_scores, "Dlib", "#e74c3c"),
    ]:
        vals = scores + scores[:1]
        ax3.plot(angles, vals, "o-", linewidth=2, label=label, color=color)
        ax3.fill(angles, vals, alpha=0.1, color=color)

    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, size=8)
    ax3.set_ylim(0, 5)
    ax3.set_title("Framework Comparison\n(Radar Chart)", pad=15)
    ax3.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    plt.tight_layout()
    chart_path = f"{output_dir}/algorithm_comparison.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"✓ Chart saved → {chart_path}")
    return chart_path


def main():
    print("=" * 60)
    print("DAY 1: Face Recognition Algorithm Research & Comparison")
    print("=" * 60)

    df = generate_comparison_table()
    print("\n📊 Algorithm Comparison Table:")
    print(df.to_string(index=False))

    chart = generate_charts(df)

    df.to_csv("outputs/algorithm_comparison.csv", index=False)
    print("\n✓ Table saved → outputs/algorithm_comparison.csv")
    print("\n✅ Day 1 research output complete!")


if __name__ == "__main__":
    main()