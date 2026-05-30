"""
Fetch 4 Sri Lanka province boundaries from Overpass (out geom) and save GeoJSON.

Each province is one OSM relation → one GeoFeature with MultiPolygon geometry
(stitched outer rings from relation members).
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "app" / "static" / "data" / "lk_provinces_4.geojson"

QUERY = """
[out:json][timeout:120];
(
  relation["boundary"="administrative"]["admin_level"="4"]["name"="Western Province"];
  relation["boundary"="administrative"]["admin_level"="4"]["name"="Central Province"];
  relation["boundary"="administrative"]["admin_level"="4"]["name"="North Western Province"];
  relation["boundary"="administrative"]["admin_level"="4"]["name"="Southern Province"];
);
out geom;
"""

PROVINCE_ORDER = [
    "Western Province",
    "Central Province",
    "North Western Province",
    "Southern Province",
]


def _pt_key(lon: float, lat: float, prec: int = 6) -> Tuple[float, float]:
    return (round(lon, prec), round(lat, prec))


def _way_coords(member: dict, ways: dict, nodes: dict) -> List[List[float]]:
    if "geometry" in member:
        return [[p["lon"], p["lat"]] for p in member["geometry"]]
    ref = member.get("ref")
    if ref not in ways:
        return []
    coords = []
    for nid in ways[ref].get("nodes", []):
        if nid in nodes:
            n = nodes[nid]
            coords.append([n["lon"], n["lat"]])
    return coords


def stitch_segments(segments: List[List[List[float]]]) -> List[List[List[float]]]:
    """Join OSM way segments (outer role) into closed rings."""
    if not segments:
        return []

    used = [False] * len(segments)
    rings: List[List[List[float]]] = []

    def endpoints(seg: List[List[float]]):
        return _pt_key(seg[0][0], seg[0][1]), _pt_key(seg[-1][0], seg[-1][1])

    for i in range(len(segments)):
        if used[i]:
            continue
        ring = list(segments[i])
        used[i] = True
        start_pt, end_pt = endpoints(ring)

        changed = True
        while changed:
            changed = False
            for j in range(len(segments)):
                if used[j]:
                    continue
                seg = segments[j]
                s, e = endpoints(seg)
                if e == end_pt:
                    ring.extend(seg[1:])
                    end_pt = _pt_key(ring[-1][0], ring[-1][1])
                    used[j] = True
                    changed = True
                elif s == end_pt:
                    ring.extend(reversed(seg[:-1]))
                    end_pt = _pt_key(ring[-1][0], ring[-1][1])
                    used[j] = True
                    changed = True
                elif s == start_pt:
                    ring = list(reversed(seg[:-1])) + ring
                    start_pt = _pt_key(ring[0][0], ring[0][1])
                    used[j] = True
                    changed = True
                elif e == start_pt:
                    ring = seg[:-1] + ring
                    start_pt = _pt_key(ring[0][0], ring[0][1])
                    used[j] = True
                    changed = True

        if ring and (ring[0][0], ring[0][1]) != (ring[-1][0], ring[-1][1]):
            ring.append(ring[0])
        if len(ring) >= 4:
            rings.append(ring)
    return rings


def relation_to_geometry(rel: dict, ways: dict, nodes: dict) -> dict:
    segments = []
    for m in rel.get("members", []):
        if m.get("type") != "way":
            continue
        if m.get("role") not in ("outer", ""):
            continue
        coords = _way_coords(m, ways, nodes)
        if len(coords) >= 2:
            segments.append(coords)

    rings = stitch_segments(segments)
    if not rings:
        return {}

    if len(rings) == 1:
        return {"type": "Polygon", "coordinates": [rings[0]]}
    return {"type": "MultiPolygon", "coordinates": [[r] for r in rings]}


def _ring_point_count(geom: dict) -> int:
    if geom["type"] == "Polygon":
        return sum(len(r) for r in geom["coordinates"])
    return sum(len(r) for poly in geom["coordinates"] for r in poly)


def osm_to_geojson(data: dict) -> dict:
    nodes = {n["id"]: n for n in data["elements"] if n["type"] == "node"}
    ways = {w["id"]: w for w in data["elements"] if w["type"] == "way"}
    relations = [e for e in data["elements"] if e["type"] == "relation"]

    by_name: Dict[str, list] = {}
    for rel in relations:
        name = rel.get("tags", {}).get("name")
        if name not in PROVINCE_ORDER:
            continue
        geom = relation_to_geometry(rel, ways, nodes)
        if not geom:
            continue
        feat = {
            "type": "Feature",
            "properties": {
                "name": name,
                "osm_relation_id": rel["id"],
                "admin_level": rel.get("tags", {}).get("admin_level", "4"),
            },
            "geometry": geom,
        }
        by_name.setdefault(name, []).append(feat)

    features = []
    for name in PROVINCE_ORDER:
        candidates = by_name.get(name, [])
        if not candidates:
            continue
        features.append(max(candidates, key=lambda f: _ring_point_count(f["geometry"])))

    return {"type": "FeatureCollection", "features": features}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("Fetching from Overpass API (out geom)...")
    r = requests.post(
        "https://overpass.kumi.systems/api/interpreter",
        data=QUERY,
        headers={"Content-Type": "text/plain; charset=utf-8"},
        timeout=180,
    )
    r.raise_for_status()
    data = r.json()
    gj = osm_to_geojson(data)
    if len(gj["features"]) != 4:
        print(f"Warning: expected 4 provinces, got {len(gj['features'])}", file=sys.stderr)
    OUT.write_text(json.dumps(gj), encoding="utf-8")
    for f in gj["features"]:
        g = f["geometry"]
        n = len(g["coordinates"]) if g["type"] == "Polygon" else len(g["coordinates"])
        print(f"  {f['properties']['name']}: {g['type']} ({n} part(s)), rel {f['properties']['osm_relation_id']}")
    print(f"Wrote -> {OUT}")


if __name__ == "__main__":
    main()
