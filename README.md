# Data Storm v7.0 — Latent Potential Pipeline

**Team:** `AI-ACES`  
**Target:** Estimate Maximum Monthly Purchase Volume (Liters) per outlet for **January 2026**

---

## Primary model (Option A)

**Production predictions** combine the interpretable two-regime heuristic with a **ceiling-target LightGBM quantile** (blend weight increases with censoring score). See `pipeline/latent_heuristic.py` (`USE_CEILING_QUANTILE_BLEND = True`).

- `Heuristic_Latent_Liters` — heuristic only (audit / XAI)
- `Maximum_Monthly_Liters` — **submission column** (blended)
- `Quantile_Ceiling_Liters` — statistical ceiling estimate

**Observed-sales LightGBM** remains a benchmark only (`06_validation.py`).

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

### Generative AI (XAI narratives)

Set a Hugging Face token (recommended) or Anthropic key.

**Option A — `.env` file (recommended):**

```bash
copy .env.example .env
# Edit .env and set HF_TOKEN=hf_...
pip install -r requirements.txt
```

**Option B — PowerShell session:**

```powershell
$env:HF_TOKEN = "hf_xxxxxxxx"
$env:HF_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
# Optional fallback
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

Priority: **HF_TOKEN → ANTHROPIC_API_KEY → rules engine**. Narratives are grounded on outlet KPIs + top attribution drivers.

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
| `output/ceiling_validation_report.csv` | Ceiling-proxy metrics (rank corr, uplift, gap recovery) |
| `pipeline/gold/heuristic_calibration.json` | Walk-forward calibrated uplift constants |
| `output/model_benchmark_chronological.csv` | Short comparison from gold step |
| `pipeline/gold/gold_features.parquet` | Full feature table (+ `Quantile_Ceiling_Liters`) |
| `pipeline/gold/ceiling_quantile_model.pkl` | Trained ceiling quantile model |
| `output/ceiling_validation_summary.json` | Ceiling validation aggregates |
| `data/outlet_intelligence.db` | SQLite for web app |
| `data/genai_transparency.jsonl` | Append-only GenAI audit log |

---

## Latent demand heuristic (summary)

Observed volume is treated as **supply-capped** (`V_obs ≤ D_true`). A **two-regime blend** applies:

- **Low censoring** → conservative baseline (size × type × season × competition)
- **High censoring** → full latent uncapping (peer gap + catchment + censoring uplift)

```
w = smoothstep(censoring_score; start=0.15, full=0.45)
baseline = jan_base × size × type × season × competition_dampener
latent   = baseline × (1 + CENSORING_UPLIFT × cens) × peer_gap × (1 + CATCHMENT_UPLIFT × catch)
potential_raw = (1 - w) × baseline + w × latent
```

Constants `CENSORING_UPLIFT` and `CATCHMENT_UPLIFT` are recalibrated walk-forward against ceiling proxies (`max(hist_p90, hist_max, actual)`) via `pipeline/calibrate_heuristic.py` — not fitted to LightGBM.

Then: floor vs history, cap at 5× `jan_base`, optional uplift for high censoring (`> 0.40`).

### Ceiling validation + quantile regression (Priority 1)

**Statistical anchor:** LightGBM quantile (τ=0.90) on ceiling proxy — blended into `Maximum_Monthly_Liters` when `USE_CEILING_QUANTILE_BLEND = True` (current default).

**Full methodology:** `deliverables/METHODOLOGY.md`

```bash
python run_pipeline.py --validate   # includes ceiling validation step
# or: python pipeline/08_ceiling_validation.py
```

| Output | Purpose |
|--------|---------|
| `output/ceiling_validation_report.csv` | Per-fold ceiling metrics (heuristic vs ceiling-quantile vs observed-LGBM) |
| `output/ceiling_validation_summary.json` | Judge-friendly aggregate |
| `samples/ceiling_validation_summary.json` | Copy committed after validation run |

Primary metrics: Spearman rank vs ceiling proxy, uplift–censoring monotonicity, % predictions exceeding observed (gap recovery). **Do not judge latent demand using observed-sales MAE alone.**

See `pipeline/latent_heuristic.py` for heuristic constants.

### Budget optimization

Shared parameters: `pipeline/optimization_config.py` — **LKR 5M**, **LKR 100k/outlet cap**. Details: `docs/OPTIMIZATION.md`.

### Judge quickstart (10 min)

1. `pip install -r requirements.txt`
2. Place competition CSVs in `input/` (see table above)
3. `python run_pipeline.py --validate`
4. Review `samples/ceiling_validation_summary.json` and `deliverables/METHODOLOGY.md`
5. `copy .env.example .env` (optional, for XAI)
6. `python app/app.py`

GenAI audit log: `data/genai_transparency.jsonl` (schema: `samples/genai_transparency.example.jsonl`).

---

## Data quality

Reusable checks in `pipeline/dq_checks.py`. Rejected rows → `pipeline/rejected/` with `dq_failure_reason`.

---

## POI scraping

OpenStreetMap Overpass API (Sri Lanka bbox). Gaussian decay (σ=300m), gravity scores, H3 indexing. Map layer uses `pipeline/poi_cache/osm_competitor_pois.parquet` (real coordinates only).
