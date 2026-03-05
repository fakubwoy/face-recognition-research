# day2/collect_results.py
"""
Day 2 Final Step: Collect all result JSONs + system info into one file.
Run this AFTER all eval scripts have completed.

Output: results/day2_full_results.json  ← paste this back to Claude
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
        "cpu": platform.processor() or "unknown",
    }
    # Try to get CPU details
    try:
        out = subprocess.check_output(
            ["lscpu"], text=True, stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            if "Model name" in line:
                info["cpu_model"] = line.split(":")[1].strip()
            if "CPU(s):" in line and "NUMA" not in line and "On-line" not in line:
                info["cpu_cores"] = line.split(":")[1].strip()
    except Exception:
        pass
    # RAM
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_gb"] = round(mem.total / 1e9, 1)
    except Exception:
        info["ram_gb"] = "unknown (pip install psutil)"
    return info


def main():
    print("Collecting all Day 2 results...")

    combined = {
        "day": 2,
        "generated": datetime.now().isoformat(),
        "system": get_system_info(),
        "lfw_pairs_evaluation": load_json("lfw_pairs_results.json"),
        "low_resolution_test": load_json("lowres_results.json"),
        "occlusion_test": load_json("occlusion_results.json"),
        "deepface_backends": load_json("deepface_backends.json"),
        "proprietary_apis": load_json("proprietary_api_results.json"),
    }

    out = RESULTS_DIR / "day2_full_results.json"
    with open(out, "w") as f:
        json.dump(combined, f, indent=2)

    size_kb = round(out.stat().st_size / 1024, 1)
    print(f"\n✓ Combined results saved → {out}  ({size_kb} KB)")
    print("\nSections present:")
    for k, v in combined.items():
        if isinstance(v, list):
            print(f"  ✓ {k}: {len(v)} entries")
        elif isinstance(v, dict):
            if "error" in v:
                print(f"  ✗ {k}: MISSING")
            else:
                print(f"  ✓ {k}")
    print(f"\n→ Paste the contents of {out} back to Claude to generate the Day 2 report.")


if __name__ == "__main__":
    main()