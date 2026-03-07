#!/usr/bin/env python3
# day4/run_all.py
"""
Day 4 Master Runner
Usage:
  python run_all.py           # full run
  python run_all.py --quick   # fast mode
"""

import sys, subprocess, time
from pathlib import Path

QUICK = "--quick" in sys.argv
if QUICK:
    print("⚡ QUICK MODE\n")
    import os
    os.environ["DAY4_QUICK"] = "1"

SCRIPTS = [
    ("eval_retrieval.py",           "Task 1: Retrieval Evaluation (Rank-1, mAP, CMC)"),
    ("eval_clustering.py",          "Task 2: Face Clustering"),
    ("eval_final_benchmark.py",     "Task 3: Final Head-to-Head Benchmark"),
    ("cost_comparison.py",          "Task 4: Cost Estimation at Scale"),
    ("generate_recommendation.py",  "Task 5: Final Recommendation Report"),
    ("collect_results.py",          "Combining All Results"),
]

total_start = time.time()
for script, description in SCRIPTS:
    print(f"\n{'='*60}\n▶  {description}\n{'='*60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], cwd=Path(__file__).parent)
    elapsed = time.time() - t0
    status = "✓ Done" if result.returncode == 0 else "✗ Failed"
    print(f"\n{status} — {elapsed:.0f}s")

total = time.time() - total_start
print(f"\n{'='*60}")
print(f"✅  All Day 4 tasks complete in {total/60:.1f} minutes")
print(f"📁  Results: results/day4_full_results.json")
print(f"{'='*60}")