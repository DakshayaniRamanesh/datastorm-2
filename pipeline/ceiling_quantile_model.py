"""
DataStorm 2026 - Ceiling-target quantile regression
=================================================
Trains LightGBM quantile models on a *ceiling proxy* (not raw observed sales)
for statistical validation of latent potential estimates.

Ceiling proxy (per outlet-month):
    max(hist_p90_vol, hist_max_vol, actual_month_volume)

Production submissions remain the interpretable heuristic (Option A) unless
USE_CEILING_QUANTILE_BLEND is enabled in latent_heuristic.py.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
import pandas as pd

CEILING_QUANTILE_ALPHA = 0.90
CEILING_MODEL_NAME = "Ensemble_Ceiling_Quantile"

DEFAULT_FEATURE_COLS: List[str] = [
    "hist_mean_vol",
    "hist_median_vol",
    "hist_max_vol",
    "hist_p75_vol",
    "hist_p90_vol",
    "hist_cv",
    "hist_months",
    "censoring_score",
    "yoy_growth",
    "target_season_factor",
    "target_month",
    "Cooler_Count",
    "combined_catchment_score",
    "effective_catchment_score",
    "competitor_density_gaussian",
    "competition_dampener",
    "peer_efficiency_gap",
    "size_factor",
    "type_factor",
    "gravity_catchment_score",
    "market_saturation_index",
]

LGB_CEILING_PARAMS = {
    "objective": "quantile",
    "alpha": CEILING_QUANTILE_ALPHA,
    "metric": "quantile",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 0.2,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

XGB_CEILING_PARAMS = {
    "objective": "reg:quantileerror",
    "quantile_alpha": CEILING_QUANTILE_ALPHA,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": 42,
    "n_jobs": -1,
}

HGB_CEILING_PARAMS = {
    "loss": "quantile",
    "quantile": CEILING_QUANTILE_ALPHA,
    "max_iter": 300,
    "learning_rate": 0.05,
    "random_state": 42,
}


class CeilingQuantileEnsemble:
    """Multi-model ensemble (LightGBM, XGBoost, HistGradientBoosting) for robust quantile regression."""
    def __init__(self, lgb_params: dict, xgb_params: dict, hgb_params: dict, w_lgb=0.4, w_xgb=0.4, w_hgb=0.2):
        self.lgb_params = lgb_params
        self.xgb_params = xgb_params
        self.hgb_params = hgb_params
        self.w_lgb = w_lgb
        self.w_xgb = w_xgb
        self.w_hgb = w_hgb
        
        self.model_lgb = None
        self.model_xgb = None
        self.model_hgb = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.model_lgb = lgb.LGBMRegressor(**self.lgb_params)
        self.model_lgb.fit(X, y)
        
        self.model_xgb = xgb.XGBRegressor(**self.xgb_params)
        self.model_xgb.fit(X, y)
        
        self.model_hgb = HistGradientBoostingRegressor(**self.hgb_params)
        self.model_hgb.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p_lgb = self.model_lgb.predict(X)
        p_xgb = self.model_xgb.predict(X)
        p_hgb = self.model_hgb.predict(X)
        return self.w_lgb * p_lgb + self.w_xgb * p_xgb + self.w_hgb * p_hgb



def ceiling_proxy_array(df: pd.DataFrame, y_obs: Optional[np.ndarray] = None) -> np.ndarray:
    """Soft ceiling reference: max(p90, max history, optional observed month)."""
    p90 = df["hist_p90_vol"].values if "hist_p90_vol" in df.columns else df["hist_median_vol"].values
    hmax = df["hist_max_vol"].values if "hist_max_vol" in df.columns else p90
    ceiling = np.maximum(p90, hmax)
    if y_obs is not None:
        ceiling = np.maximum(ceiling, np.asarray(y_obs, dtype=float))
    elif "actual" in df.columns:
        ceiling = np.maximum(ceiling, df["actual"].values.astype(float))
    elif "actual_vol" in df.columns:
        ceiling = np.maximum(ceiling, df["actual_vol"].values.astype(float))
    return ceiling


def ceiling_target_log(df: pd.DataFrame, y_obs: Optional[np.ndarray] = None) -> np.ndarray:
    return np.log1p(np.clip(ceiling_proxy_array(df, y_obs=y_obs), 0.0, None))


def resolve_feature_cols(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
) -> List[str]:
    cols = list(feature_cols) if feature_cols else DEFAULT_FEATURE_COLS
    return [c for c in cols if c in df.columns]


def prepare_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in feature_cols:
        if X[c].dtype.name == "category":
            X[c] = X[c].cat.codes
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    return X


def train_ceiling_quantile_model(
    train_df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    y_obs_col: str = "actual_vol",
) -> tuple[CeilingQuantileEnsemble, List[str]]:
    """Fit quantile regressor ensemble on log1p(ceiling proxy)."""
    cols = resolve_feature_cols(train_df, feature_cols)
    if not cols:
        raise ValueError("No feature columns available for ceiling quantile model.")

    y_obs = train_df[y_obs_col].values if y_obs_col in train_df.columns else None
    y = ceiling_target_log(train_df, y_obs=y_obs)
    X = prepare_features(train_df, cols)

    model = CeilingQuantileEnsemble(
        lgb_params=LGB_CEILING_PARAMS,
        xgb_params=XGB_CEILING_PARAMS,
        hgb_params=HGB_CEILING_PARAMS,
    )
    model.fit(X, y)
    return model, cols


def predict_ceiling_quantile_liters(
    model: CeilingQuantileEnsemble,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> np.ndarray:
    """Predict ceiling potential in liters (expm1 of quantile log prediction)."""
    X = prepare_features(df, feature_cols)
    return np.expm1(model.predict(X))


def blend_heuristic_with_quantile(
    heuristic_liters: np.ndarray,
    quantile_liters: np.ndarray,
    censoring: np.ndarray,
) -> np.ndarray:
    """
    Censoring-weighted blend: high censoring -> more weight on statistical ceiling quantile.
    Quantile estimate is never below 95% of heuristic (avoids collapsing potential).
    """
    from latent_heuristic import censoring_regime_weight

    w = censoring_regime_weight(np.asarray(censoring, dtype=float))
    q = np.maximum(np.asarray(quantile_liters, dtype=float), np.asarray(heuristic_liters, dtype=float) * 0.95)
    h = np.asarray(heuristic_liters, dtype=float)
    return (1.0 - w) * h + w * q
