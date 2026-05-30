"""
DataStorm 2026 - Master Pipeline Runner
========================================
Runs the full Bronze -> Silver -> Gold pipeline end to end.

Usage:
    python run_pipeline.py                    # Full pipeline (skips POI + validation)
    python run_pipeline.py --validate         # Also runs out-of-time validation
    python run_pipeline.py --poi              # Also runs live POI scraping (slow, ~8h)
    python run_pipeline.py --poi-sample 200   # POI on first 200 outlets (testing)
    python run_pipeline.py --poi --validate   # Full pipeline + POI + validation
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path

ROOT = Path(__file__).parent

# Core pipeline steps (always run)
CORE_STEPS = [
    ("Bronze Ingestion",        "pipeline/01_bronze_ingest.py",      []),
    ("Silver Cleaning + DQ",    "pipeline/02_silver_clean.py",       []),
    ("Gold Features & Model",   "pipeline/04_gold_features_model.py",[]),
    ("EDA Dashboard",           "pipeline/05_eda.py",                []),
]

# Ensure child processes use UTF-8
_child_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def run_step(label: str, script: str, extra_args: list) -> bool:
    """Run a single pipeline step as a subprocess. Returns True on success."""
    print(f"\n{'='*60}")
    print(f">>  {label}")
    print(f"{'='*60}")
    t0     = time.time()
    cmd    = [sys.executable, str(ROOT / script)] + extra_args
    result = subprocess.run(cmd, cwd=str(ROOT), env=_child_env)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[FAIL]  Step failed: {label}  (elapsed: {elapsed:.1f}s)")
        return False
    print(f"\n[OK]  {label} completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="DataStorm 2026 Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                     # Core pipeline only
  python run_pipeline.py --validate          # Core + out-of-time validation
  python run_pipeline.py --poi               # Core + live POI scraping
  python run_pipeline.py --poi-sample 200    # Core + POI (sample 200 outlets)
  python run_pipeline.py --poi --validate    # Everything
        """
    )
    parser.add_argument("--poi", action="store_true",
                        help="Run live POI scraping (slow — ~8h for all outlets)")
    parser.add_argument("--poi-sample", type=int, default=None,
                        help="POI scrape sample size (for testing)")
    parser.add_argument("--validate", action="store_true",
                        help="Run out-of-time TimeSeriesSplit validation after gold model")
    parser.add_argument("--compare", action="store_true",
                        help="Generate input vs prediction comparison plots")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Build step list dynamically
    # -----------------------------------------------------------------------
    steps = [
        ("Bronze Ingestion",        "pipeline/01_bronze_ingest.py",      []),
        ("Silver Cleaning + DQ",    "pipeline/02_silver_clean.py",       []),
    ]

    # POI scraping (optional — injected between Silver and Gold)
    if args.poi or args.poi_sample:
        poi_args = []
        if args.poi_sample:
            poi_args = ["--sample", str(args.poi_sample)]
        steps.append(("POI Scraping (OSM/Overpass)", "pipeline/03_poi_scraper.py", poi_args))

    steps.append(("Gold Features & Model", "pipeline/04_gold_features_model.py", []))

    # Out-of-time validation (optional — runs after gold, before EDA)
    if args.validate:
        steps.append(("Out-of-Time Validation", "pipeline/06_validation.py", []))

    # Comparison analysis plots (optional — runs after gold)
    if args.compare:
        steps.append(("Comparison Analysis Plots", "pipeline/07_comparison_plots.py", []))

    steps.append(("EDA Dashboard", "pipeline/05_eda.py", []))

    # -----------------------------------------------------------------------
    # Print plan
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(">>>  DataStorm 2026 - Latent Potential Pipeline")
    print("=" * 60)
    print(f"\n  Steps to run ({len(steps)} total):")
    for i, (label, _, _) in enumerate(steps, 1):
        print(f"    {i}. {label}")
    print()

    # -----------------------------------------------------------------------
    # Execute
    # -----------------------------------------------------------------------
    pipeline_start = time.time()
    for label, script, extra_args in steps:
        ok = run_step(label, script, extra_args)
        if not ok:
            print(f"\n[ABORT]  Pipeline halted at: {label}")
            print("         Fix the error above and re-run.\n")
            sys.exit(1)

    total_elapsed = time.time() - pipeline_start

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[DONE]  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Total elapsed:  {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"\n  Outputs:")
    print(f"    Predictions    -> output/AI_ACES_predictions.csv")
    print(f"    Gold table     -> pipeline/gold/gold_features.parquet")
    print(f"    Rejected       -> pipeline/rejected/*.csv")
    print(f"    DQ manifest    -> pipeline/rejected/dq_manifest.json")
    print(f"    Cens. analysis -> output/censoring_analysis.csv")
    if args.validate:
        print(f"    Validation     -> output/validation_report.csv")
    if args.compare:
        print(f"    Comparison     -> output/comparison_analysis.png")
    print(f"    EDA dashboard  -> output/eda_dashboard.png")
    print(f"    EDA scatter    -> output/eda_censoring_scatter.png\n")


if __name__ == "__main__":
    main()
