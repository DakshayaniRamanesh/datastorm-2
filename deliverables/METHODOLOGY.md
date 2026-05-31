# Methodology — Latent Demand & Ceiling Validation (AI-ACES)

## Problem framing

Observed monthly liters at an outlet are **supply-capped**:

\[
V_{\text{obs}} \leq D_{\text{true}}
\]

The finals task is **maximum latent potential** for January 2026, not next-month observed sales.

## Production model (submission)

**Primary:** two-regime interpretable heuristic (`pipeline/latent_heuristic.py`).

| Regime | When | Formula role |
|--------|------|----------------|
| Baseline | Low censoring | Size × type × season × competition |
| Latent uncapping | High censoring | + censoring uplift, peer gap, catchment |

Walk-forward calibrated constants: `CENSORING_UPLIFT`, `CATCHMENT_UPLIFT` (`pipeline/calibrate_heuristic.py`).

**Production blend (enabled):** censoring-weighted mix of heuristic + ceiling quantile:

```python
USE_CEILING_QUANTILE_BLEND = True  # in latent_heuristic.py
w = smoothstep(censoring_score)
prediction = (1 - w) * heuristic + w * max(quantile_ceiling, 0.95 * heuristic)
```

`Heuristic_Latent_Liters` = interpretable model alone; `Maximum_Monthly_Liters` = blended submission.

Compare before/after: `output/ceiling_blend_comparison.json` (from `08_ceiling_validation.py`).

## Censoring score (uncapping signal)

Composite in `pipeline/feature_builder.py` (not “hit max once”):

1. Volume plateau (3-month CV &lt; 10%)
2. Distributor delivery cap proxy
3. Inter-year stagnation
4. Low CV vs peers
5. Q4 suppression vs annual mean

## Statistical anchor: ceiling-target quantile regression

**Model:** LightGBM quantile (α = 0.90) in `pipeline/ceiling_quantile_model.py`.

**Target (training):**

\[
y = \log\!\left(1 + \max(\text{hist\_p90}, \text{hist\_max}, V_{\text{obs}})\right)
\]

This trains on a **ceiling proxy**, not raw median sales — appropriate for latent-demand validation.

**Outputs on gold table:**

- `Quantile_Ceiling_Liters` — statistical ceiling estimate per outlet
- `Heuristic_Latent_Liters` — heuristic before optional blend
- `Maximum_Monthly_Liters` — submission column (heuristic, or blend if enabled)

## How we validate (what judges should look at)

Run:

```bash
python run_pipeline.py --validate
python pipeline/08_ceiling_validation.py
```

**Primary metrics** (`output/ceiling_validation_report.csv`, summary in `samples/ceiling_validation_summary.json`):

| Metric | Meaning |
|--------|---------|
| `rank_corr_ceiling_spearman` | Ranking vs ceiling proxy (higher = better) |
| `uplift_censoring_spearman` | Higher predicted uplift when censoring is high (should be &gt; 0) |
| `pct_pred_exceeds_observed` | Gap recovery for capped outlets |
| `log_mae_to_ceiling_weighted` | Error to ceiling, weighted by censoring |

**Do not** use observed-sales MAE alone to judge latent potential.

## What this is not

- Not full Tobit MLE (heuristic + quantile anchor instead)
- LightGBM on **observed** sales is a **benchmark only** (`06_validation.py`)

## Spatial & competition

Gaussian decay (σ = 300 m) + gravity (β tuned) + competitor density — see `pipeline/03_poi_scraper.py`.
