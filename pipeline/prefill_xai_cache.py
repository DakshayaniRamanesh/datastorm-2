"""
DataStorm 2026 - Pre-populate XAI Cache (Priority 7)
===================================================
Pre-populates data/xai_cache.json for the top N outlets by Maximum_Monthly_Liters
so executive narratives load instantly during live demos.

Uses the rules-engine fallback when ANTHROPIC_API_KEY is not set (fast, no API cost).
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from app.services.xai_service import XAIService, XAI_CACHE_PATH

TOP_N = 200


def _ensure_xai_fields(row: dict) -> dict:
    """Normalize gold row keys expected by XAIService."""
    defaults = {
        "combined_catchment_score": 0.0,
        "competitor_density_gaussian": 0.0,
        "Cooler_Count": 0,
        "yoy_growth": 0.0,
    }
    for key, default in defaults.items():
        val = row.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            row[key] = default
    return row


def main() -> None:
    print("=" * 60)
    print("XAI cache pre-population (top outlets by latent potential)")
    print("=" * 60)

    gold_path = ROOT / "pipeline" / "gold" / "gold_features.parquet"
    if not gold_path.exists():
        print(f"Error: run gold pipeline first — missing {gold_path}")
        sys.exit(1)

    df = pd.read_parquet(gold_path)
    top = df.sort_values("Maximum_Monthly_Liters", ascending=False).head(TOP_N)
    print(f"Loaded {len(df):,} outlets; caching top {len(top):,} by Maximum_Monthly_Liters.")

    xai = XAIService()
    cached = 0

    for idx, (_, row) in enumerate(top.iterrows(), 1):
        outlet_data = _ensure_xai_fields(row.to_dict())
        outlet_id = outlet_data["Outlet_ID"]
        xai.get_claude_advisor_narrative(outlet_data)
        cached += 1
        if idx % 25 == 0 or idx == len(top):
            print(f"  [{idx:3d}/{len(top)}] Cached {outlet_id}")

    if XAI_CACHE_PATH.exists():
        with open(XAI_CACHE_PATH, encoding="utf-8") as f:
            n_entries = len(json.load(f))
        print(f"\nDone. {cached} narratives written; {n_entries} total entries in {XAI_CACHE_PATH}")
        print(f"Cache size: {XAI_CACHE_PATH.stat().st_size:,} bytes")
    else:
        print("\nWarning: cache file was not created.")


if __name__ == "__main__":
    main()
