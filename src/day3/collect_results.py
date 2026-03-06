# day3/collect_results.py
"""
Day 3 Final Step: Collect all Day 3 result JSONs into one file.
Run this AFTER all eval scripts have completed.

Output: results/day3_full_results.json
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
    return {"error": f"{name} not found — did you run that eval script?"}


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
        pass
    return info


def main():
    print("Collecting all Day 3 results...")

    combined = {
        "day": 3,
        "generated": datetime.now().isoformat(),
        "system": get_system_info(),
        "insightface_disk_evaluation": load_json("insightface_disk_results.json"),
        "full_lfw_benchmark": load_json("full_lfw_benchmark.json"),
        "vector_db_benchmark": load_json("vector_db_benchmark.json"),
        "superresolution_test": load_json("superresolution_results.json"),
        "cost_estimation": load_json("cost_estimation.json"),
        "storage_architecture": load_json("storage_architecture.json"),
    }

    out = RESULTS_DIR / "day3_full_results.json"
    with open(out, "w") as f:
        json.dump(combined, f, indent=2)

    size_kb = round(out.stat().st_size / 1024, 1)
    print(f"\n✓ Combined results saved → {out}  ({size_kb} KB)")
    print("\nSections present:")
    for k, v in combined.items():
        if isinstance(v, (list, dict)):
            has_error = (isinstance(v, dict) and "error" in v) or \
                        (isinstance(v, list) and any("error" in x for x in v if isinstance(x, dict)))
            status = "✗ MISSING" if isinstance(v, dict) and "error" in v else "✓"
            count = f"{len(v)} entries" if isinstance(v, list) else ""
            print(f"  {status} {k} {count}")

    print(f"\n→ Paste the contents of {out} back to Claude to generate the Day 3 report.")


if __name__ == "__main__":
    main()