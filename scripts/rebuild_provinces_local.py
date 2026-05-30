"""Rebuild lk_provinces_4.geojson by fetching one province per Overpass request."""
import json
import sys
import time
from pathlib import Path

import requests

# Reuse conversion from fetch_lk_provinces
sys.path.insert(0, str(Path(__file__).parent))
from fetch_lk_provinces import OUT, PROVINCE_ORDER, osm_to_geojson  # noqa: E402

ENDPOINTS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]


def fetch_one(name: str) -> dict:
    query = f"""
[out:json][timeout:90];
relation["boundary"="administrative"]["admin_level"="4"]["name"]="{name}"];
out geom;
"""
    for url in ENDPOINTS:
        try:
            print(f"  Fetching {name} via {url.split('/')[2]}...")
            r = requests.post(
                url,
                data=query,
                headers={"Content-Type": "text/plain; charset=utf-8"},
                timeout=120,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"    failed: {e}")
            time.sleep(2)
    raise RuntimeError(f"All endpoints failed for {name}")


def main():
    all_elements = []
    seen_ids = set()

    for name in PROVINCE_ORDER:
        data = fetch_one(name)
        for el in data.get("elements", []):
            eid = (el["type"], el["id"])
            if eid not in seen_ids:
                seen_ids.add(eid)
                all_elements.append(el)
        time.sleep(3)

    merged = {"elements": all_elements}
    gj = osm_to_geojson(merged)
    if len(gj["features"]) < 4:
        print(f"Warning: only {len(gj['features'])} provinces parsed", file=sys.stderr)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(gj), encoding="utf-8")
    for f in gj["features"]:
        g = f["geometry"]
        n = len(g["coordinates"]) if g["type"] == "Polygon" else len(g["coordinates"])
        print(f"  OK {f['properties']['name']}: {g['type']} ({n} parts), rel={f['properties'].get('osm_relation_id')}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
