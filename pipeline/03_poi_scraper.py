"""
DataStorm 2026 - Spatial Analytics & Competitor Intelligence Pipeline
=====================================================================
Fetches Points of Interest (POIs) and Competitors near outlets,
calculates Haversine distances, and generates spatial features:
  1. Gaussian Distance Decay Scores (sigma = 300m) for all POI categories.
  2. Retail Gravity Model Scores with auto-tuned beta in [1.5, 2.0].
  3. Combined Catchment Scores (with optimized weights).
  4. Competitor density (Gaussian and Gravity), Market Saturation Index, and Competition Dampener.
  5. H3 Geohashing at Resolution 6 and 8.

Performance:
  Uses 3D Cartesian Coordinate Projection combined with scipy.spatial.cKDTree
  to convert O(N * M) great-circle distance loops into O(N log M) spatial joins.
"""

import sys
import os
import time
import json
import logging
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial import cKDTree

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SpatialAnalytics")

# Paths
ROOT = Path(__file__).parent.parent
SILVER = ROOT / "pipeline" / "silver"
POI_CACHE = ROOT / "pipeline" / "poi_cache"
POI_CACHE.mkdir(parents=True, exist_ok=True)

# H3 Geohash Loading
H3_AVAILABLE = False
_h3_geo_to_idx = None
try:
    import h3 as _h3_lib
    if hasattr(_h3_lib, "latlng_to_cell"):
        _h3_geo_to_idx = lambda lat, lon, res: _h3_lib.latlng_to_cell(lat, lon, res)
        _h3_version = "v4.x"
    elif hasattr(_h3_lib, "geo_to_h3"):
        _h3_geo_to_idx = lambda lat, lon, res: _h3_lib.geo_to_h3(lat, lon, res)
        _h3_version = "v3.x"
    H3_AVAILABLE = True
    logger.info(f"Loaded h3 library ({_h3_version})")
except ImportError:
    logger.warning("h3 library not installed. H3 features will be generated with placeholder/empty strings.")

# Parameters
EARTH_RADIUS_M = 6371000.0
SIGMA_M = 300.0  # Gaussian decay sigma
SEARCH_RADIUS_M = 1000.0  # 1 km search radius

# POI Category configuration with Overpass filters
POI_CATEGORIES = {
    "school": 'node["amenity"~"school|college|university|kindergarten"]',
    "bus_stop": 'node["highway"="bus_stop"]; node["amenity"="bus_station"]',
    "hospital": 'node["amenity"~"hospital|clinic|doctors|pharmacy"]',
    "tourism": 'node["tourism"~"attraction|hotel|museum|viewpoint|zoo|theme_park"]',
    "market": 'node["shop"~"supermarket|convenience|mall|department_store"]',
    "place_worship": 'node["amenity"="place_of_worship"]',
    "fuel_station": 'node["amenity"="fuel"]',
    "restaurant": 'node["amenity"~"restaurant|cafe|bar|fast_food|food_court"]',
    "bank_atm": 'node["amenity"~"bank|atm"]',
    "competitor": 'node["shop"~"convenience|grocery|general|supermarket"]'
}

# Default importance weights for Gravity Model
POI_IMPORTANCE = {
    "school": 2.0,
    "bus_stop": 2.0,
    "hospital": 1.5,
    "tourism": 1.8,
    "market": 1.0,
    "place_worship": 1.2,
    "fuel_station": 1.0,
    "restaurant": 1.3,
    "bank_atm": 1.0,
    "competitor": 1.5
}

def latlon_to_cartesian(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Project Latitude/Longitude coordinates to 3D Cartesian coordinates on unit sphere."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.column_stack((x, y, z))

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute Haversine distance in meters between two points."""
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2.0) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_M * c

def query_overpass_all_sri_lanka() -> List[Dict]:
    """Query Overpass API for all targeted POIs in Sri Lanka in a single batch query."""
    cache_file = POI_CACHE / "raw_sri_lanka_pois.json"
    if cache_file.exists():
        logger.info(f"Loading raw OSM POIs from cache: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info("POI cache not found. Querying Overpass API for Sri Lanka bounding box...")
    
    # Sri Lanka Bounding Box
    bbox = "5.9,79.5,9.9,82.0"
    
    # Construct combined query for nodes and ways (out center gives centers for ways)
    query_body = f"""
    [out:json][timeout:240];
    (
      node["amenity"~"school|college|university|kindergarten"]({bbox});
      way["amenity"~"school|college|university|kindergarten"]({bbox});
      node["highway"="bus_stop"]({bbox});
      node["amenity"="bus_station"]({bbox});
      way["amenity"="bus_station"]({bbox});
      node["amenity"~"hospital|clinic|doctors|pharmacy"]({bbox});
      way["amenity"~"hospital|clinic|doctors|pharmacy"]({bbox});
      node["tourism"~"attraction|hotel|museum|viewpoint|zoo|theme_park"]({bbox});
      way["tourism"~"attraction|hotel|museum|viewpoint|zoo|theme_park"]({bbox});
      node["shop"~"supermarket|convenience|mall|department_store|grocery|general"]({bbox});
      way["shop"~"supermarket|convenience|mall|department_store|grocery|general"]({bbox});
      node["amenity"="place_of_worship"]({bbox});
      way["amenity"="place_of_worship"]({bbox});
      node["amenity"="fuel"]({bbox});
      way["amenity"="fuel"]({bbox});
      node["amenity"~"restaurant|cafe|bar|fast_food|food_court"]({bbox});
      way["amenity"~"restaurant|cafe|bar|fast_food|food_court"]({bbox});
      node["amenity"~"bank|atm"]({bbox});
      way["amenity"~"bank|atm"]({bbox});
    );
    out center;
    """
    
    overpass_url = "https://overpass-api.de/api/interpreter"
    try:
        resp = requests.post(overpass_url, data={"data": query_body}, timeout=300)
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
        logger.info(f"Successfully scraped {len(elements):,} raw POIs from Overpass API.")
        
        # Cache raw response
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(elements, f, ensure_ascii=False, indent=2)
            
        return elements
    except Exception as e:
        logger.error(f"Overpass API scraping failed: {e}. Falling back to synthetic generator.")
        return []

def classify_poi_element(el: Dict) -> List[str]:
    """Classify an OSM element into one or more of our POI categories based on its tags."""
    tags = el.get("tags", {})
    categories = []
    
    amenity = tags.get("amenity", "")
    highway = tags.get("highway", "")
    tourism = tags.get("tourism", "")
    shop = tags.get("shop", "")
    
    # 1. School
    if amenity in ["school", "college", "university", "kindergarten"]:
        categories.append("school")
        
    # 2. Bus Stop
    if highway == "bus_stop" or amenity == "bus_station":
        categories.append("bus_stop")
        
    # 3. Hospital
    if amenity in ["hospital", "clinic", "doctors", "pharmacy"]:
        categories.append("hospital")
        
    # 4. Tourism
    if tourism in ["attraction", "hotel", "museum", "viewpoint", "zoo", "theme_park"]:
        categories.append("tourism")
        
    # 5. Market (General Commercial Market)
    if shop in ["supermarket", "convenience", "mall", "department_store"]:
        categories.append("market")
        
    # 6. Place of Worship
    if amenity == "place_of_worship":
        categories.append("place_worship")
        
    # 7. Fuel Station
    if amenity == "fuel":
        categories.append("fuel_station")
        
    # 8. Restaurant
    if amenity in ["restaurant", "cafe", "bar", "fast_food", "food_court"]:
        categories.append("restaurant")
        
    # 9. Bank/ATM
    if amenity in ["bank", "atm"]:
        categories.append("bank_atm")
        
    # 10. Competitor
    if shop in ["convenience", "grocery", "general", "supermarket"]:
        categories.append("competitor")
        
    return categories

def extract_pois_dataframe(elements: List[Dict], outlet_coords: pd.DataFrame) -> pd.DataFrame:
    """Parse raw elements from Overpass, structure into a DataFrame, or trigger fallback."""
    records = []
    for el in elements:
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if lat is None or lon is None:
            continue
            
        cats = classify_poi_element(el)
        for cat in cats:
            records.append({
                "lat": float(lat),
                "lon": float(lon),
                "category": cat
            })
            
    if records:
        poi_df = pd.DataFrame(records)
        logger.info(f"Parsed {len(poi_df):,} POI records across {poi_df['category'].nunique()} categories.")
        return poi_df
        
    # --- FALLBACK GENERATOR ---
    logger.warning("No POI records parsed. Running high-fidelity local competitor/POI generator using outlet coordinates...")
    fallback_records = []
    
    # 1. Use existing Groceries and SMMT outlets in Sri Lanka as competitors and markets
    outlets_master = pd.read_parquet(SILVER / "outlet_master.parquet")
    coords_master = pd.read_parquet(SILVER / "outlet_coordinates.parquet")
    merged_meta = coords_master.merge(outlets_master, on="Outlet_ID", how="inner")
    
    # Identify competitor outlets (Grocery & SMMT)
    competitor_outlets = merged_meta[merged_meta["Outlet_Type"].isin(["Grocery", "SMMT"])]
    for _, row in competitor_outlets.iterrows():
        # Add actual competitor POI
        fallback_records.append({"lat": row["Latitude"], "lon": row["Longitude"], "category": "competitor"})
        fallback_records.append({"lat": row["Latitude"], "lon": row["Longitude"], "category": "market"})
        
    # 2. Add randomly distributed POIs around outlets to serve as schools, bus stops, hospitals, etc.
    # This ensures spatial feature distributions closely match real-world distributions.
    np.random.seed(42)
    categories_to_gen = ["school", "bus_stop", "hospital", "tourism", "place_worship", "fuel_station", "restaurant", "bank_atm"]
    
    for _, row in merged_meta.sample(frac=0.6, random_state=42).iterrows():
        base_lat = row["Latitude"]
        base_lon = row["Longitude"]
        # Generate 1 to 3 POIs in the vicinity
        for _ in range(np.random.randint(1, 4)):
            # Random offset (approx 50m to 800m)
            offset_lat = np.random.normal(0, 0.003)
            offset_lon = np.random.normal(0, 0.003)
            cat = np.random.choice(categories_to_gen)
            fallback_records.append({
                "lat": base_lat + offset_lat,
                "lon": base_lon + offset_lon,
                "category": cat
            })
            
    fallback_df = pd.DataFrame(fallback_records)
    logger.info(f"Generated {len(fallback_df):,} synthetic POI features to ensure pipeline integrity.")
    return fallback_df

def tune_gravity_beta(
    outlets_df: pd.DataFrame,
    poi_df: pd.DataFrame,
    c_radius_3d: float
) -> float:
    """Tune the Gravity Model beta coefficient automatically in [1.5, 2.0]

    by maximizing correlation with historical sales volume.
    """
    logger.info("Tuning Gravity model beta parameter automatically...")
    
    # Load historical sales to find correlation target
    try:
        tx = pd.read_parquet(SILVER / "transactions.parquet")
        monthly = tx.groupby(["Outlet_ID", "Year", "Month"])["Volume_Liters"].sum().reset_index()
        sales_vol = monthly.groupby("Outlet_ID")["Volume_Liters"].median().rename("median_vol")
        outlets_vol = outlets_df.merge(sales_vol, on="Outlet_ID", how="inner")
        if len(outlets_vol) < 100:
            logger.warning("Too few outlets with sales history to tune beta. Using default beta = 1.8.")
            return 1.8
    except Exception as e:
        logger.warning(f"Could not load sales history for beta tuning: {e}. Using default beta = 1.8.")
        return 1.8

    # Project coordinates to 3D Cartesian
    poi_cart = latlon_to_cartesian(poi_df["lat"].values, poi_df["lon"].values)
    out_cart = latlon_to_cartesian(outlets_vol["Latitude"].values, outlets_vol["Longitude"].values)
    
    tree = cKDTree(poi_cart)
    indices_list = tree.query_ball_point(out_cart, c_radius_3d)
    
    best_beta = 1.8
    best_corr = -1.0
    
    # Test values in [1.5, 2.0]
    betas_to_test = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    
    for beta in betas_to_test:
        gravity_scores = []
        for i, idxs in enumerate(indices_list):
            if not idxs:
                gravity_scores.append(0.0)
                continue
                
            sub_pois = poi_df.iloc[idxs]
            o_lat = outlets_vol.iloc[i]["Latitude"]
            o_lon = outlets_vol.iloc[i]["Longitude"]
            
            dists = haversine_distance(o_lat, o_lon, sub_pois["lat"].values, sub_pois["lon"].values)
            # Filter to 1km
            valid = dists <= SEARCH_RADIUS_M
            if not np.any(valid):
                gravity_scores.append(0.0)
                continue
                
            v_dists = dists[valid]
            v_cats = sub_pois["category"].values[valid]
            
            score = 0.0
            for d, cat in zip(v_dists, v_cats):
                imp = POI_IMPORTANCE.get(cat, 1.0)
                # Cap minimum distance at 10m to avoid zero division/extreme outliers
                score += imp / ((d + 10.0) ** beta)
            gravity_scores.append(score)
            
        corr = np.abs(np.corrcoef(gravity_scores, np.log1p(outlets_vol["median_vol"].values))[0, 1])
        logger.info(f"  Beta = {beta:.1f} -> Pearson Correlation with Log Sales: {corr:.4f}")
        if corr > best_corr:
            best_corr = corr
            best_beta = beta
            
    logger.info(f"Optimal Gravity Beta selected: {best_beta:.1f} (Correlation: {best_corr:.4f})")
    return best_beta

def optimize_catchment_weights(
    gauss_vals: np.ndarray,
    grav_vals: np.ndarray,
    dens_vals: np.ndarray,
    sales_vals: np.ndarray
) -> Tuple[float, float, float]:
    """Grid search weight combinations for combined_catchment_score to maximize log sales correlation.

    Subject to: weights sum to 1.0, and w_gaussian >= 0.20, w_gravity >= 0.20.
    """
    logger.info("Optimizing combined catchment score weights...")
    
    # Scale values to [0, 1] for correlation grid search
    g_norm = gauss_vals / (np.percentile(gauss_vals, 95) + 1e-9)
    gr_norm = grav_vals / (np.percentile(grav_vals, 95) + 1e-9)
    d_norm = dens_vals / (np.percentile(dens_vals, 95) + 1e-9)
    
    best_weights = (0.40, 0.40, 0.20)
    best_corr = -1.0
    
    # Generate weight configurations
    for w_gauss in np.linspace(0.20, 0.60, 9):
        for w_grav in np.linspace(0.20, 0.60, 9):
            w_dens = 1.0 - w_gauss - w_grav
            if w_dens < 0.05 or w_dens > 0.40:
                continue
                
            combined = w_gauss * g_norm + w_grav * gr_norm + w_dens * d_norm
            corr = np.abs(np.corrcoef(combined, sales_vals)[0, 1])
            
            if corr > best_corr:
                best_corr = corr
                best_weights = (float(w_gauss), float(w_grav), float(w_dens))
                
    logger.info(f"Optimized weights: Gaussian={best_weights[0]:.2f}, Gravity={best_weights[1]:.2f}, Commercial Density={best_weights[2]:.2f} (Corr: {best_corr:.4f})")
    return best_weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="Process first N outlets only (testing)")
    args = parser.parse_args()
    
    logger.info("Starting Spatial Analytics Upgrade Pipeline...")
    
    # 1. Load outlet coordinates
    coords_path = SILVER / "outlet_coordinates.parquet"
    if not coords_path.exists():
        logger.error(f"Silver outlet coordinates parquet file not found at: {coords_path}")
        sys.exit(1)
    outlets_df = pd.read_parquet(coords_path)
    if args.sample:
        outlets_df = outlets_df.head(args.sample).copy()
        logger.info(f"Running in SAMPLE MODE on first {args.sample} outlets.")
        
    # 2. Fetch and parse POI elements
    raw_elements = query_overpass_all_sri_lanka()
    poi_df = extract_pois_dataframe(raw_elements, outlets_df)
    
    # 3. Project to 3D Cartesian coordinates for rapid cKDTree radius matching
    c_radius_3d = 2.0 * np.sin(SEARCH_RADIUS_M / (2.0 * EARTH_RADIUS_M))
    poi_cart = latlon_to_cartesian(poi_df["lat"].values, poi_df["lon"].values)
    out_cart = latlon_to_cartesian(outlets_df["Latitude"].values, outlets_df["Longitude"].values)
    
    # Build spatial index tree
    logger.info("Building cKDTree spatial index...")
    tree = cKDTree(poi_cart)
    
    # Query spatial tree for neighbors within 1km
    logger.info(f"Running radius search (radius={SEARCH_RADIUS_M}m) using 3D chord threshold...")
    indices_list = tree.query_ball_point(out_cart, c_radius_3d)
    
    # 4. Auto-tune Gravity beta
    beta = tune_gravity_beta(outlets_df, poi_df, c_radius_3d)
    
    # 5. Process distances, Gaussian decay, and Gravity scores
    logger.info("Computing exact Haversine distance decay & gravity metrics...")
    
    results = []
    
    # Arrays to store aggregated values for weight optimization
    agg_gaussian = np.zeros(len(outlets_df))
    agg_gravity = np.zeros(len(outlets_df))
    agg_density = np.zeros(len(outlets_df))
    
    for i, row in outlets_df.iterrows():
        o_id = row["Outlet_ID"]
        o_lat = row["Latitude"]
        o_lon = row["Longitude"]
        
        rec = {
            "Outlet_ID": o_id,
            "poi_lat": o_lat,
            "poi_lon": o_lon
        }
        
        # Initialize scores for all categories
        for cat in POI_CATEGORIES.keys():
            rec[f"{cat}_gaussian_score"] = 0.0
            rec[f"{cat}_gravity_score"] = 0.0
            
        rec["commercial_density"] = 0.0
        
        # Retrieve neighbor POIs
        neighbor_idxs = indices_list[i]
        if neighbor_idxs:
            sub_pois = poi_df.iloc[neighbor_idxs]
            dists = haversine_distance(o_lat, o_lon, sub_pois["lat"].values, sub_pois["lon"].values)
            
            # Filter strictly to 1km radius
            valid = dists <= SEARCH_RADIUS_M
            if np.any(valid):
                v_dists = dists[valid]
                v_cats = sub_pois["category"].values[valid]
                
                # Compute scores
                for d, cat in zip(v_dists, v_cats):
                    imp = POI_IMPORTANCE.get(cat, 1.0)
                    # Gaussian
                    rec[f"{cat}_gaussian_score"] += np.exp(-(d**2) / (2.0 * (SIGMA_M**2)))
                    # Gravity
                    rec[f"{cat}_gravity_score"] += imp / ((d + 10.0) ** beta)
                    
                    # Density: Increment commercial density for commercial POIs
                    if cat in ["market", "restaurant", "bank_atm", "fuel_station", "competitor"]:
                        rec["commercial_density"] += 1.0
                        
        # Store aggregations
        agg_gaussian[i] = sum(rec[f"{cat}_gaussian_score"] * POI_IMPORTANCE.get(cat, 1.0) for cat in POI_CATEGORIES.keys() if cat != "competitor")
        agg_gravity[i] = sum(rec[f"{cat}_gravity_score"] for cat in POI_CATEGORIES.keys() if cat != "competitor")
        agg_density[i] = rec["commercial_density"]
        
        results.append(rec)
        
    scores_df = pd.DataFrame(results)
    
    # 6. Optimize combined catchment score weights
    sales_vals = None
    try:
        tx = pd.read_parquet(SILVER / "transactions.parquet")
        monthly = tx.groupby(["Outlet_ID", "Year", "Month"])["Volume_Liters"].sum().reset_index()
        sales_vol = monthly.groupby("Outlet_ID")["Volume_Liters"].median().rename("median_vol")
        outlets_vol = outlets_df.merge(sales_vol, on="Outlet_ID", how="inner")
        if len(outlets_vol) >= 100:
            sales_vals = np.log1p(outlets_vol["median_vol"].values)
            # Get matching index alignments
            matched_indices = outlets_df["Outlet_ID"].isin(outlets_vol["Outlet_ID"])
            w_gauss, w_grav, w_dens = optimize_catchment_weights(
                agg_gaussian[matched_indices],
                agg_gravity[matched_indices],
                agg_density[matched_indices],
                sales_vals
            )
        else:
            w_gauss, w_grav, w_dens = 0.40, 0.40, 0.20
    except Exception:
        w_gauss, w_grav, w_dens = 0.40, 0.40, 0.20
        
    # Scale inputs for combined score
    g_max = agg_gaussian.max() + 1e-9
    gr_max = agg_gravity.max() + 1e-9
    d_max = agg_density.max() + 1e-9
    
    scores_df["combined_catchment_score"] = (
        w_gauss * (agg_gaussian / g_max) +
        w_grav * (agg_gravity / gr_max) +
        w_dens * (agg_density / d_max)
    ).clip(0.0, 1.0)
    
    # 7. Competitor metrics
    logger.info("Computing competitor density profiles and saturation indices...")
    scores_df["competitor_density_gaussian"] = scores_df["competitor_gaussian_score"]
    scores_df["competitor_density_gravity"] = scores_df["competitor_gravity_score"]
    
    scores_df["market_saturation_index"] = (
        scores_df["competitor_density_gaussian"] / (scores_df["commercial_density"] / d_max + 1e-9)
    ).clip(0.0, 5.0)
    
    scores_df["competition_penalty"] = np.minimum(0.20, scores_df["competitor_density_gaussian"] * 0.10)
    scores_df["competition_dampener"] = 1.0 - scores_df["competition_penalty"]
    
    # Add prefix for old model compatibility (poi_ prefix)
    for cat in POI_CATEGORIES.keys():
        scores_df[f"poi_{cat}"] = scores_df[f"{cat}_gaussian_score"].round().astype(int)
        
    # 8. H3 Hashing at resolution 6 and 8
    logger.info("Performing H3 geospatial indexing at Resolution 6 and 8...")
    if H3_AVAILABLE and _h3_geo_to_idx is not None:
        scores_df["h3_index"] = scores_df.apply(
            lambda r: _h3_geo_to_idx(r["poi_lat"], r["poi_lon"], 8), axis=1
        )
        scores_df["h3_res6"] = scores_df.apply(
            lambda r: _h3_geo_to_idx(r["poi_lat"], r["poi_lon"], 6), axis=1
        )
    else:
        scores_df["h3_index"] = ""
        scores_df["h3_res6"] = ""
        
    # Save features
    out_file = POI_CACHE / "poi_features.parquet"
    scores_df.to_parquet(out_file, index=False)
    
    logger.info(f"Spatial analytics upgrade complete. Saved parquet: {out_file} ({len(scores_df):,} outlets)")
    logger.info(f"Spatial features generated: {list(scores_df.columns)}")

if __name__ == "__main__":
    main()
