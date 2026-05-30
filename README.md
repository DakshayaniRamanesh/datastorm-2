# Data Storm v7.0 — Latent Potential Pipeline

**Team:** `AI-ACES`  
**Target:** Estimate Maximum Monthly Purchase Volume (Liters) per outlet for **January 2026**

---

## Primary model (Option A)

**Production predictions use the interpretable latent-demand heuristic** in `pipeline/latent_heuristic.py` — a multiplicative uncapping model driven by censoring score, peer ceiling, seasonality, and OSM catchment features.

**LightGBM / quantile regressors are benchmarks** fit to observed monthly sales for comparison only; they do not drive `Maximum_Monthly_Liters` in the submission file.

---

## Project structure

```
datastorm-2/
├── run_pipeline.py              ← Orchestrator
├── input/                       ← Raw CSVs (not committed; see below)
├── pipeline/
│   ├── 01_bronze_ingest.py
│   ├── 02_silver_clean.py
│   ├── 03_poi_scraper.py        ← Real OSM only (no synthetic POI)
│   ├── 04_gold_features_model.py
│   ├── 05_budget_optimizer.py
│   ├── 06_validation.py
│   ├── latent_heuristic.py      ← Primary latent model
│   ├── bronze/ silver/ gold/
│   ├── rejected/
│   └── poi_cache/               ← raw_sri_lanka_pois.json + features
├── output/                      ← Predictions, allocations, reports
└── app/                         ← Flask dashboard (python app/app.py)
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place raw CSV files in `input/`

| File | Description |
|------|-------------|
| `transactions_history_final.csv` | Transaction history |
| `outlet_master.csv` | Outlet metadata |
| `outlet_coordinates.csv` | Lat/Lon per outlet |
| `distributor_seasonality_details.csv` | Seasonality index |
| `holiday_list.csv` | Public holidays |

### 3. OSM POI cache (first run or refresh)

```bash
python pipeline/03_poi_scraper.py
```

Fetches OpenStreetMap data via Overpass (~5 min first time). Cached at `pipeline/poi_cache/raw_sri_lanka_pois.json`. **Synthetic POI generation is disabled.**

---

## Running the pipeline

```bash
# Full pipeline (includes POI from cache, validation, SQLite compile)
python run_pipeline.py

# Skip validation (faster)
python run_pipeline.py --no-validate

# Refresh OSM from network (slow)
python pipeline/03_poi_scraper.py
python run_pipeline.py --no-poi
```

### Web app

```bash
python app/app.py
# Open http://127.0.0.1:5000
```

Set `FLASK_DEBUG=1` for local dev only. Optional: `SECRET_KEY` env var for stable sessions.

### Pre-populate XAI cache (before demo)

```bash
python pipeline/prefill_xai_cache.py
```

Caches narratives for the top 200 outlets by potential (instant load in XAI tab).

---

## Outputs

| File | Description |
|------|-------------|
| `output/AI_ACES_predictions.csv` | Submission: `Outlet_ID`, `Maximum_Monthly_Liters` |
| `output/ai_aces_budget_allocations.csv` | Western Province trade spend |
| `output/validation_report.csv` | Walk-forward metrics (heuristic + ML benchmarks) |
| `output/model_benchmark_chronological.csv` | Short comparison from gold step |
| `pipeline/gold/gold_features.parquet` | Full feature table |
| `data/outlet_intelligence.db` | SQLite for web app |

---

## Latent demand heuristic (summary)

Observed volume is treated as **supply-capped** (`V_obs ≤ D_true`). Potential is estimated as:

```
Potential = jan_base × size_factor × type_factor × season_factor
          × (1 + 0.40 × censoring_score)
          × peer_efficiency_gap
          × (1 + 0.15 × combined_catchment_score)
          × competition_dampener
```

Then: floor vs history, cap at 5× `jan_base`, optional uplift for high censoring (`> 0.40`).

See `pipeline/latent_heuristic.py` for constants and implementation.

---

## Data quality

Reusable checks in `pipeline/dq_checks.py`. Rejected rows → `pipeline/rejected/` with `dq_failure_reason`.

---

## POI scraping

OpenStreetMap Overpass API (Sri Lanka bbox). Gaussian decay (σ=300m), gravity scores, H3 indexing. Map layer uses `pipeline/poi_cache/osm_competitor_pois.parquet` (real coordinates only).
