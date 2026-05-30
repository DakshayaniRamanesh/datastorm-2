# Data Storm v7.0 — Latent Potential Pipeline

**Team:** `AI-ACES`  
**Target:** Estimate Maximum Monthly Purchase Volume (Liters) per outlet for **January 2026**

---

## Project Structure

```
datastorm/
├── run_pipeline.py                 ← Orchestrator: run this
├── pipeline/
│   ├── 01_bronze_ingest.py         ← Raw ingestion
│   ├── 02_silver_clean.py          ← Dataset-specific cleaning logic
│   ├── 03_poi_scraper.py           ← Overpass API POI enrichment
│   ├── 04_gold_features_model.py   ← Latent demand estimation model
│   ├── 05_eda.py                   ← EDA dashboard generation
│   ├── 06_validation.py            ← TimeSeriesSplit validation
│   ├── 07_comparison_plots.py      ← Comparison analysis plots
│   ├── dq_checks.py                ← Reusable DQ check library
│   ├── bronze/                     ← Auto-generated: bronze parquets
│   ├── silver/                     ← Auto-generated: silver parquets
│   ├── gold/                       ← Auto-generated: gold features
│   ├── rejected/                   ← Quarantined bad records
│   └── poi_cache/                  ← Cached POI query results
└── input/                          ← PUT CSV FILES HERE
```

---

## Setup

### 1. Install dependencies
```bash
pip install pandas numpy requests
```
> No sklearn, statsmodels, or other ML libraries are required.

### 2. Place raw data files in `data/raw/`
| File | Description |
|------|-------------|
| `transactions_history_final.csv` | Outlet-level transaction history |
| `outlet_master.csv` | Outlet metadata (type, province, distributor) |
| `outlet_coordinates.csv` | Latitude/Longitude per outlet |
| `distributor_seasonality_details.csv` | Monthly seasonality index per distributor |
| `holiday_list.csv` | Sri Lanka public holidays |
 
---

## Running the Pipeline

### Full pipeline (Core steps)
```bash
python run_pipeline.py
```

### Full pipeline with Validation and Comparison Plots
```bash
python run_pipeline.py --validate --compare
```

### Full pipeline including POI Scraping (may take time)
```bash
python run_pipeline.py --poi
```

### Outputs
| File | Description |
|------|-------------|
| `output/{team_name}_predictions.csv` | **Submission file**: `Outlet_ID`, `Maximum_Monthly_Liters` |
| `output/predictions_detailed.csv` | Full breakdown of all model factors |
| `data/silver/rejected/` | Quarantined records with documented failure reasons |
| `data/gold/poi_cache.csv` | Cached POI query results (Overpass API) |

---

## Mathematical Framework

### Problem: Left-Censored Demand

Observed volume **V_obs = min(D_true, C_supply)**  
Where **C_supply** = credit limits + stockout constraints + delivery caps  
Therefore **V_obs ≤ D_true** (historical data is censored from above)

### Uncapping Approach (3 Stages)

**Stage 1 — Self Ceiling**
```
B_self = p85(outlet's own monthly volumes)
       blended with jan_avg if January history exists
```

**Stage 2 — Peer Ceiling Blend (Regression-to-Segment)**
```
B_blended = B_self + α × (peer_p75 − B_self)
  where α = 0.40  (40% pull toward peer segment ceiling)
  peer_p75 = 75th percentile of p90_volume within same outlet_type × province
  Cap: B_blended ≤ B_self × 3.5
```

**Stage 3 — Multiplicative Adjustments**
```
Potential = B_blended
          × jan_seasonality_index        (from distributor data)
          × trend_factor                 (OLS slope extrapolation, clipped ±20%)
          × holiday_factor               (2% per Jan holiday, max +10%)
          × cv_boost                     (high-variance = supply-constrained signal)
          × poi_boost_factor             (catchment demand drivers, max +30%)
```

### OLS Trend (from scratch — no sklearn)
```
β₁ = [n·Σ(t·y) − Σt·Σy] / [n·Σt² − (Σt)²]
t = 0, 1, ..., n-1  (month index)
```

### Peer Segmentation
Outlets are grouped by `outlet_type × province` (e.g., "grocery × Western").  
Within each group, p50, p75, and p90 of each outlet's peak volume are computed.

---

## Data Quality Checks (Reusable Functions)

All checks are in `pipeline/silver/dq_checks.py` and applied via `run_checks()`.

| Check | What it does |
|-------|-------------|
| `check_duplicates` | Removes duplicate records by configurable primary key |
| `check_nulls` | Flags records missing mandatory fields |
| `check_referential_integrity` | Validates FK values exist in reference set |
| `check_value_range` | Rejects numeric values outside [min, max] |
| `check_format` | Validates string fields against a regex pattern |

Rejected records are written to `data/silver/rejected/` with a `dq_failure_reason` column — they are **never silently dropped**.

---

## POI Scraping

Uses the **OpenStreetMap Overpass API** (free, no key needed).  
For each outlet with valid coordinates, queries a 500m radius for:
- Bus stops / stations (weight: 4–5)
- Schools, universities (weight: 3)
- Hospitals, clinics (weight: 2)
- Marketplaces (weight: 5)
- Places of worship, hotels, tourism (weight: 2–3)

Results are cached in `data/gold/poi_cache.csv` so re-runs don't re-query.
