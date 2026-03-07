# day4/collect_results.py
"""
Day 4 Final Step: Collect all result JSONs into one file.
Run AFTER all eval scripts have completed.

Output: results/day4_full_results.json
"""

import json
import platform
import subprocess
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("results")


def load_json(name):
    p = RESULTS_DIR / name
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"error": f"{name} not found"}


def get_system_info():
    info = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if "Model name" in line:
                info["cpu_model"] = line.split(":")[1].strip()
    except Exception:
        pass
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        info["ram_gb"] = "unknown"
    return info


def main():
    print("Collecting all Day 4 results...")

    combined = {
        "day": 4,
        "generated": datetime.now().isoformat(),
        "system": get_system_info(),
        "retrieval_evaluation":   load_json("retrieval_results.json"),
        "clustering_evaluation":  load_json("clustering_results.json"),
        "final_benchmark":        load_json("final_benchmark_results.json"),
        "cost_estimation":        load_json("cost_results.json"),
        "recommendation":         load_json("recommendation.json"),
    }

    out = RESULTS_DIR / "day4_full_results.json"
    with open(out, "w") as f:
        json.dump(combined, f, indent=2)

    size_kb = round(out.stat().st_size / 1024, 1)
    print(f"\n✓ Combined results saved → {out}  ({size_kb} KB)")
    print("\nSections present:")
    for k, v in combined.items():
        if isinstance(v, dict) and "error" in v:
            print(f"  ✗ {k}: MISSING")
        elif isinstance(v, list):
            print(f"  ✓ {k}: {len(v)} entries")
        elif isinstance(v, dict):
            print(f"  ✓ {k}")
    print(f"\n→ Paste the contents of {out} back to Claude to generate the Day 4 report.")


if __name__ == "__main__":
    main()