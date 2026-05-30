"""
POI SCRAPER - OpenStreetMap / Overpass API
==========================================
Fetches Points of Interest (POI) near each outlet using the free
Overpass API (no key required).  Results are cached in `pipeline/poi_cache/`
so re-runs never re-hit the API.

Rate Limiting & Resilience
---------------------------
- Exponential backoff on failures: 2s -> 4s -> 8s (3 retries max).
- Token bucket: sustained max of 1 request/second to respect Overpass ToS.
- Failures beyond retries are recorded as -1 (treated as 0 in features).

H3 Geohash Indexing
--------------------
Each outlet is assigned an H3 hex cell index at resolution 8 (~460m cells).
This enables O(1) hash-map spatial joins instead of O(N*M) Haversine loops.
If the `h3` library is not installed, the column is omitted with a warning.

Catchment Drivers Targeted
---------------------------
POI Category        | Overpass Tag                   | Demand Rationale
--------------------|--------------------------------|-------------------------------------
Schools             | amenity=school/college/univ    | Youth impulse + canteen supply
Bus Stands          | highway=bus_stop / amenity=bus_station | High footfall / commuter
Hospitals/Clinics   | amenity=hospital/clinic        | High visitor footfall
Tourist Attractions | tourism=*                      | Premium/seasonal demand
Markets / Supermarkets | shop=supermarket/market     | Competing supply signal
Mosques/Temples/Churches | amenity=place_of_worship  | Festive demand spikes
Religious Sites (Buddhist) | amenity=monastery/temple | Sri Lanka specific
Petrol Stations     | amenity=fuel                   | Passing trade proxy
Restaurants/Cafes   | amenity=restaurant/cafe/bar    | Adjacent F&B demand

Usage:
    python pipeline/03_poi_scraper.py [--sample N]
    (--sample N runs on first N outlets only - useful for testing)
"""

import sys
import time
import json
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
ROOT      = Path(__file__).parent.parent
SILVER    = ROOT / "pipeline" / "silver"
POI_CACHE = ROOT / "pipeline" / "poi_cache"
POI_CACHE.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# H3 import (optional — graceful degradation, v3.x and v4.x compatible)
# ---------------------------------------------------------------------------
H3_AVAILABLE   = False
_h3_geo_to_idx = None  # version-safe wrapper, set below

try:
    import h3 as _h3_lib

    # h3-py v4.x uses latlng_to_cell; v3.x uses geo_to_h3
    if hasattr(_h3_lib, "latlng_to_cell"):
        _h3_geo_to_idx = lambda lat, lon, res: _h3_lib.latlng_to_cell(lat, lon, res)
        _h3_version    = "v4.x"
    elif hasattr(_h3_lib, "geo_to_h3"):
        _h3_geo_to_idx = lambda lat, lon, res: _h3_lib.geo_to_h3(lat, lon, res)
        _h3_version    = "v3.x"
    else:
        raise AttributeError("Unrecognised h3 API")

    H3_AVAILABLE = True
    print(f"[INFO] h3 library loaded ({_h3_version} API, v{_h3_lib.__version__})")

except ImportError:
    print("[WARN] `h3` library not installed. H3 geohash column will be omitted.")
    print("       Install with: pip install h3")
except AttributeError as _h3_err:
    print(f"[WARN] h3 library found but API unrecognised: {_h3_err}")
    print("       H3 geohash column will be omitted.")

# H3 resolution 8 = ~460m hexagon side length (optimal for 1km POI radius)
H3_RESOLUTION = 8

# ---------------------------------------------------------------------------
# Overpass query configuration
# ---------------------------------------------------------------------------
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Radius in metres around each outlet
SEARCH_RADIUS_M = 1000  # 1 km radius for POI catchment

POI_TAGS = {
    "school":        'amenity~"school|college|university|kindergarten"',
    # bus_stop uses two separate union clauses — pipe is not valid inside a single filter
    "bus_stop":      'amenity~"bus_station"',   # bus_station nodes handled separately below
    "hospital":      'amenity~"hospital|clinic|doctors|pharmacy"',
    "tourism":       'tourism~"attraction|hotel|museum|viewpoint|zoo|theme_park"',
    "market":        'shop~"supermarket|convenience|mall|department_store"',
    "place_worship": 'amenity~"place_of_worship"',
    "fuel_station":  'amenity~"fuel"',
    "restaurant":    'amenity~"restaurant|cafe|bar|fast_food|food_court"',
    "bank_atm":      'amenity~"bank|atm"',
}

# Overpass QL timeout (seconds)
OVERPASS_TIMEOUT = 25

# ---------------------------------------------------------------------------
# Token Bucket Rate Limiter (max 1 req/sec sustained)
# ---------------------------------------------------------------------------
class TokenBucket:
    """Simple token bucket: max `rate` requests per second."""
    def __init__(self, rate: float = 1.0):
        self._rate      = rate
        self._tokens    = rate
        self._last_time = time.monotonic()

    def consume(self) -> None:
        """Block until a token is available, then consume it."""
        now    = time.monotonic()
        elapsed = now - self._last_time
        self._tokens    = min(self._rate, self._tokens + elapsed * self._rate)
        self._last_time = now
        if self._tokens < 1.0:
            sleep_secs = (1.0 - self._tokens) / self._rate
            time.sleep(sleep_secs)
            self._tokens = 0.0
        else:
            self._tokens -= 1.0


_bucket = TokenBucket(rate=1.0)


# ---------------------------------------------------------------------------
# Overpass helpers
# ---------------------------------------------------------------------------

def build_overpass_query(lat: float, lon: float, radius: int, tag_filter: str) -> str:
    """Build an Overpass QL query for a circular area.

    Uses `out count` to return just the total element count, not full geometry.
    The Overpass API returns the count in elements[0].tags.total.
    """
    return f"""
    [out:json][timeout:{OVERPASS_TIMEOUT}];
    (
      node[{tag_filter}](around:{radius},{lat},{lon});
      way[{tag_filter}](around:{radius},{lat},{lon});
    );
    out count;
    """


def fetch_poi_count(lat: float, lon: float, poi_key: str, tag_filter: str) -> int:
    """
    Return count of POI type within radius, using disk cache.

    Retry policy: exponential backoff with 3 attempts (2s -> 4s -> 8s).
    Returns -1 on persistent failure (treated as 0 in downstream features).
    """
    cache_key  = f"{lat:.5f}_{lon:.5f}_{poi_key}"
    cache_file = POI_CACHE / f"{cache_key}.json"

    # Cache hit — no network call needed
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f).get("count", 0)

    query = build_overpass_query(lat, lon, SEARCH_RADIUS_M, tag_filter)

    # Exponential backoff: 3 retries (2s, 4s, 8s)
    max_retries = 3
    for attempt in range(max_retries):
        _bucket.consume()  # enforce rate limit
        try:
            resp = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=OVERPASS_TIMEOUT + 5,
            )
            resp.raise_for_status()
            data = resp.json()

            # Overpass `out count` returns a single element with tags.total
            # Structure: {"elements": [{"type": "count", "tags": {"total": "42"}}]}
            elements = data.get("elements", [])
            if elements and isinstance(elements[0], dict):
                tags  = elements[0].get("tags", {})
                count = int(tags.get("total", 0))
            else:
                count = 0

            # Cache successful result
            with open(cache_file, "w") as f:
                json.dump({"count": count}, f)
            return count

        except Exception as e:
            wait = 2 ** (attempt + 1)  # 2s, 4s, 8s
            if attempt < max_retries - 1:
                print(f"\n    [RETRY {attempt+1}/{max_retries-1}] {poi_key} @ "
                      f"({lat:.4f},{lon:.4f}) failed: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                # All retries exhausted — cache the failure as -1
                with open(cache_file, "w") as f:
                    json.dump({"count": -1, "error": str(e)}, f)
                return -1

    return -1  # safety fallback (should not be reached)


# ---------------------------------------------------------------------------
# H3 Geohash indexing
# ---------------------------------------------------------------------------

def add_h3_index(df: pd.DataFrame, lat_col: str = "poi_lat",
                 lon_col: str = "poi_lon") -> pd.DataFrame:
    """
    Add an H3 hex cell index column at resolution H3_RESOLUTION.

    Converts expensive O(N*M) Haversine distance computations into
    O(1) hash-map lookups by grouping outlets into ~460m hexagons.

    Parameters
    ----------
    df      : DataFrame with latitude/longitude columns.
    lat_col : Name of the latitude column.
    lon_col : Name of the longitude column.

    Returns
    -------
    DataFrame with added `h3_index` column (string hex cell ID).
    """
    if not H3_AVAILABLE or _h3_geo_to_idx is None:
        return df
    df = df.copy()
    df["h3_index"] = df.apply(
        lambda row: _h3_geo_to_idx(row[lat_col], row[lon_col], H3_RESOLUTION)
        if pd.notna(row[lat_col]) and pd.notna(row[lon_col]) else None,
        axis=1,
    )
    n_unique_cells = df["h3_index"].nunique()
    print(f"  H3 indexing complete: {len(df):,} outlets -> "
          f"{n_unique_cells:,} unique H3 cells (res={H3_RESOLUTION})")
    return df


# ---------------------------------------------------------------------------
# Main scraping loop
# ---------------------------------------------------------------------------

def scrape_pois_for_outlets(coords_df: pd.DataFrame, sample_n: int = None) -> pd.DataFrame:
    """
    For each outlet with valid coordinates, fetch POI counts for all categories.

    Parameters
    ----------
    coords_df : Silver outlet_coordinates DataFrame.
    sample_n  : If provided, only process first N outlets (testing mode).

    Returns
    -------
    DataFrame with Outlet_ID + one column per POI category + h3_index.
    """
    if sample_n:
        coords_df = coords_df.head(sample_n).copy()
        print(f"  [SAMPLE MODE] Processing {sample_n} outlets only.")

    results = []
    total   = len(coords_df)
    failed  = 0

    for i, (_, row) in enumerate(coords_df.iterrows()):
        oid = row["Outlet_ID"]
        lat = row["Latitude"]
        lon = row["Longitude"]
        rec = {"Outlet_ID": oid, "poi_lat": lat, "poi_lon": lon}

        for poi_key, tag_filter in POI_TAGS.items():
            count = fetch_poi_count(lat, lon, poi_key, tag_filter)
            if count == -1:
                failed += 1
            rec[f"poi_{poi_key}"] = max(count, 0)  # treat -1 as 0

        # bus_stop: combine amenity=bus_station + highway=bus_stop counts
        # fetched separately because Overpass QL doesn't support | across key names
        bus_station_count = fetch_poi_count(lat, lon, "bus_stop_hw",
                                            'highway~"bus_stop"')
        if bus_station_count > 0:
            rec["poi_bus_stop"] = rec.get("poi_bus_stop", 0) + max(bus_station_count, 0)

        results.append(rec)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  Progress: {i+1}/{total} outlets  |  API failures: {failed}", end="\r")

    print()
    df = pd.DataFrame(results)

    # Add H3 geohash index for fast spatial joins
    df = add_h3_index(df)

    print(f"\n  Scraping complete. Outlets processed: {total:,} | API failures: {failed}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Only process first N outlets (for testing)")
    args = parser.parse_args()

    print("=" * 60)
    print("POI SCRAPER - OpenStreetMap / Overpass API")
    print("=" * 60)
    print(f"  H3 library available: {H3_AVAILABLE}")

    coords_df = pd.read_parquet(SILVER / "outlet_coordinates.parquet")
    print(f"  Outlets with valid coordinates: {len(coords_df):,}")

    poi_df    = scrape_pois_for_outlets(coords_df, sample_n=args.sample)
    out_path  = ROOT / "pipeline" / "poi_cache" / "poi_features.parquet"
    poi_df.to_parquet(out_path, index=False)
    print(f"\n[OK]  POI features saved -> {out_path}  ({len(poi_df):,} rows)")
    print(f"   Columns: {list(poi_df.columns)}\n")


if __name__ == "__main__":
    main()
