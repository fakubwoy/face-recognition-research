#!/usr/bin/env python3
# day3/run_all.py
"""
Day 3 Master Runner — runs all evaluations in order.
Estimated total time on CPU: 60–120 minutes.

Usage:
  python run_all.py           # run everything
  python run_all.py --quick   # reduced pairs (fast, ~20 min)
  python run_all.py --skip-insightface  # skip InsightFace if still broken
"""

import sys
import subprocess
import time
import os
from pathlib import Path

QUICK              = "--quick" in sys.argv
SKIP_INSIGHTFACE   = "--skip-insightface" in sys.argv

if QUICK:
    print("⚡ QUICK MODE — reduced pair counts\n")
    os.environ["DAY3_MAX_PAIRS"]     = "60"
    os.environ["DAY3_N_VECTORS"]     = "5000"

SCRIPTS = [
    ("eval_insightface_disk.py",    "Task 1: InsightFace Disk Re-Evaluation",   not SKIP_INSIGHTFACE),
    ("eval_full_lfw_benchmark.py",  "Task 2: Full Balanced LFW Benchmark",      True),
    ("eval_vector_dbs.py",          "Task 3: Vector DB Benchmark",              True),
    ("eval_superresolution.py",     "Task 4: Super-Resolution Preprocessing",   True),
    ("cost_estimation.py",          "Task 5: Cost Estimation",                  True),
    ("storage_architecture.py",     "Task 6: Storage Architecture Design",      True),
    ("collect_results.py",          "Combining Results",                        True),
]

total_start = time.time()

for script, description, should_run in SCRIPTS:
    if not should_run:
        print(f"\n{'='*60}")
        print(f"⏭  Skipping: {description}")
        continue

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
print(f"✅  All Day 3 tasks complete in {total/60:.1f} minutes")
print(f"📁  Results are in: results/day3_full_results.json")
print(f"🚀  Start the API server: uvicorn day3.api_server:app --reload --port 8000")
print(f"→   Paste day3_full_results.json to Claude to generate the Day 3 report doc")
print(f"{'='*60}")