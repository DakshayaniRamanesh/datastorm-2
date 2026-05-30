"""
DataStorm 2026 - Spatial Analytics Service
==========================================
Serves spatial coordinate datasets for Leaflet maps, heatmaps,
and computes local competitor maps for individual outlet detail cards.

OSM retail competitors are loaded from pipeline/poi_cache/osm_competitor_pois.parquet
(produced by pipeline/03_poi_scraper.py). No synthetic or random map points are used.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from app.services.db_service import DBService
import importlib

poi_scraper = importlib.import_module("pipeline.03_poi_scraper")
haversine_distance = poi_scraper.haversine_distance

logger = logging.getLogger("SpatialService")

OSM_COMPETITORS_PATH = (
    Path(__file__).parent.parent.parent / "pipeline" / "poi_cache" / "osm_competitor_pois.parquet"
)


class SpatialService:
    def __init__(self, db_service: DBService):
        self.db = db_service
        self._osm_competitors: Optional[pd.DataFrame] = None

    def _load_osm_competitors(self) -> pd.DataFrame:
        """Load cached OSM competitor coordinates (real data only)."""
        if self._osm_competitors is not None:
            return self._osm_competitors

        if not OSM_COMPETITORS_PATH.exists():
            logger.warning(
                "OSM competitor locations not found at %s. "
                "Run: python pipeline/03_poi_scraper.py",
                OSM_COMPETITORS_PATH,
            )
            self._osm_competitors = pd.DataFrame(columns=["lat", "lon"])
            return self._osm_competitors

        self._osm_competitors = pd.read_parquet(OSM_COMPETITORS_PATH)
        logger.info("Loaded %s OSM competitor map points.", f"{len(self._osm_competitors):,}")
        return self._osm_competitors

    def get_all_outlet_map_points(self) -> List[Dict[str, Any]]:
        """Fetch coordinate lists of all outlets for Leaflet heatmap/marker rendering."""
        query = """
            SELECT Outlet_ID, Latitude, Longitude, Maximum_Monthly_Liters, hist_median_vol, censoring_score
            FROM outlets
            WHERE Latitude IS NOT NULL AND Longitude IS NOT NULL
        """
        return self.db.execute_query(query)

    def _nearby_osm_competitors(
        self, t_lat: float, t_lon: float, radius_m: float
    ) -> List[Dict[str, Any]]:
        """Return OSM retail POIs within radius_m using real cached coordinates."""
        osm_df = self._load_osm_competitors()
        if osm_df.empty:
            return []

        # Bounding-box prefilter (~111 km per degree latitude)
        delta_deg = (radius_m / 111_000.0) * 1.2
        subset = osm_df[
            (osm_df["lat"] >= t_lat - delta_deg)
            & (osm_df["lat"] <= t_lat + delta_deg)
            & (osm_df["lon"] >= t_lon - delta_deg)
            & (osm_df["lon"] <= t_lon + delta_deg)
        ]

        results: List[Dict[str, Any]] = []
        for idx, row in subset.iterrows():
            d = haversine_distance(t_lat, t_lon, float(row["lat"]), float(row["lon"]))
            if d <= radius_m:
                results.append({
                    "id": f"OSM_{idx}",
                    "name": "OSM retail (OpenStreetMap)",
                    "type": "OSM Retailer",
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "distance_m": round(d, 1),
                    "volume": 0.0,
                })
        return results

    def get_nearby_competitors(self, outlet_id: str, radius_m: float = 1500.0) -> List[Dict[str, Any]]:
        """Find competitor outlets and OSM retailers within radius_m of the target outlet."""
        target_query = "SELECT Latitude, Longitude, primary_dist FROM outlets WHERE Outlet_ID = ?"
        target_res = self.db.execute_query(target_query, (outlet_id,))
        if not target_res:
            return []

        t_lat = target_res[0]["Latitude"]
        t_lon = target_res[0]["Longitude"]
        t_dist = target_res[0]["primary_dist"]

        results: List[Dict[str, Any]] = []

        # Neighboring Grocery/SMMT outlets (same distributor region)
        neighbor_outlets = self.db.execute_query(
            """
            SELECT Outlet_ID, Outlet_Type, Latitude, Longitude, hist_median_vol
            FROM outlets
            WHERE primary_dist = ? AND Outlet_ID != ? AND Latitude IS NOT NULL
              AND Outlet_Type IN ('Grocery', 'SMMT')
            """,
            (t_dist, outlet_id),
        )

        for n in neighbor_outlets:
            d = haversine_distance(t_lat, t_lon, n["Latitude"], n["Longitude"])
            if d <= radius_m:
                results.append({
                    "id": n["Outlet_ID"],
                    "name": f"Outlet: {n['Outlet_ID']} ({n['Outlet_Type']})",
                    "type": "Outlet Competitor",
                    "lat": n["Latitude"],
                    "lon": n["Longitude"],
                    "distance_m": round(d, 1),
                    "volume": n["hist_median_vol"],
                })

        # Real OSM retail competitors from Overpass cache
        results.extend(self._nearby_osm_competitors(t_lat, t_lon, radius_m))

        return sorted(results, key=lambda x: x["distance_m"])
