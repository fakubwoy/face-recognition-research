#!/usr/bin/env python3
# day2/run_all.py
"""
Day 2 Master Runner — runs all evaluations in order.
Estimated total time on CPU: 60–120 minutes depending on hardware.

Usage:
  python run_all.py           # run everything
  python run_all.py --quick   # reduced pairs (fast, ~15 min)
"""

import sys
import subprocess
import time
from pathlib import Path

QUICK = "--quick" in sys.argv

# Patch pair counts for quick mode
if QUICK:
    print("⚡ QUICK MODE — reduced pair counts for speed\n")
    import os
    os.environ["DAY2_MAX_PAIRS"] = "30"

SCRIPTS = [
    ("eval_lfw_pairs.py",        "Task 1: LFW Pairs Evaluation"),
    ("eval_lowres.py",           "Task 2: Low-Resolution Test"),
    ("eval_occlusion.py",        "Task 3: Occlusion Robustness"),
    ("eval_deepface_backends.py","Task 4: DeepFace Backend Comparison"),
    ("eval_proprietary_apis.py", "Task 5: Proprietary API Eval"),
    ("collect_results.py",       "Combining Results"),
]

total_start = time.time()

for script, description in SCRIPTS:
    print(f"\n{'='*60}")
    print(f"▶  {description}")
    print(f"{'='*60}")
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, script],
        cwd=Path(__file__).parent,
    )

    elapsed = time.time() - t0
    status = "✓ Done" if result.returncode == 0 else "✗ Failed (check output above)"
    print(f"\n{status} — {elapsed:.0f}s")

total = time.time() - total_start
print(f"\n{'='*60}")
print(f"✅  All Day 2 tasks complete in {total/60:.1f} minutes")
print(f"📁  Results are in: results/day2_full_results.json")
print(f"→   Paste that file to Claude to generate the Day 2 report doc")
print(f"{'='*60}")