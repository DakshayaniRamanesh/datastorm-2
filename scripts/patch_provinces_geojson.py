"""Add OSM relation metadata to bundled province GeoJSON (4 separate features)."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "app" / "static" / "data" / "lk_provinces_4.geojson"

# OSM admin_level=4 relation IDs (Sri Lanka)
RELATION_IDS = {
    "Western Province": 3602201,
    "Central Province": 3602200,
    "North Western Province": 3602198,
    "Southern Province": 3602199,
}

ORDER = list(RELATION_IDS.keys())


def main():
    gj = json.loads(OUT.read_text(encoding="utf-8"))
    by_name = {f["properties"]["name"]: f for f in gj["features"]}
    features = []
    for name in ORDER:
        if name not in by_name:
            print(f"Missing: {name}")
            continue
        f = by_name[name]
        f["properties"]["osm_relation_id"] = RELATION_IDS[name]
        f["properties"]["admin_level"] = "4"
        f["properties"]["source"] = "overpass_admin_boundaries"
        features.append(f)
    gj["features"] = features
    OUT.write_text(json.dumps(gj), encoding="utf-8")
    print(f"Patched {len(features)} province features -> {OUT}")


if __name__ == "__main__":
    main()
