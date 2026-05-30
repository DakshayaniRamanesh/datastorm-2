"""
DataStorm 2026 - XAI Service (SHAP + LLM Advisory)
==================================================
- Primary explanations: exact heuristic waterfall attributions (shap_explanations.pkl)
- Optional benchmark: TreeSHAP on ML model (shap_benchmark.pkl)
- LLM narratives: Hugging Face Inference API -> Anthropic Claude -> rules fallback
  Narratives are grounded on outlet KPIs + top attribution drivers.
"""

import os
import json
import pickle
import logging
import re
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger("XAIService")

XAI_CACHE_PATH = Path(__file__).parent.parent.parent / "data" / "xai_cache.json"
XAI_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

GOLD_DIR = Path(__file__).parent.parent.parent / "pipeline" / "gold"
DEFAULT_HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


class XAIService:
    def __init__(
        self,
        shap_pickle_path: Path = GOLD_DIR / "shap_explanations.pkl",
        benchmark_pickle_path: Path = GOLD_DIR / "shap_benchmark.pkl",
    ):
        self.shap_path = shap_pickle_path
        self.benchmark_path = benchmark_pickle_path
        self.shap_data = None
        self.benchmark_data = None
        self._load_shap_data()

    def _load_shap_data(self) -> None:
        if self.shap_path.exists():
            try:
                with open(self.shap_path, "rb") as f:
                    self.shap_data = pickle.load(f)
                etype = self.shap_data.get("explanation_type", "unknown")
                logger.info("Loaded primary explanations (%s) from %s", etype, self.shap_path)
            except Exception as e:
                logger.error("Error loading explanations: %s", e)

        if self.benchmark_path.exists():
            try:
                with open(self.benchmark_path, "rb") as f:
                    self.benchmark_data = pickle.load(f)
                logger.info("Loaded TreeSHAP benchmark from %s", self.benchmark_path)
            except Exception as e:
                logger.warning("Could not load SHAP benchmark: %s", e)

    def get_explanation_type(self) -> str:
        if self.shap_data is None:
            return "unavailable"
        return str(self.shap_data.get("explanation_type", "unknown"))

    def get_global_importance(self) -> List[Dict[str, Any]]:
        if self.shap_data is None:
            return []
        vals = np.abs(self.shap_data["shap_values"])
        mean_shap = np.mean(vals, axis=0)
        feature_names = self.shap_data["feature_names"]
        importance_list = [
            {"feature": feature_names[i], "mean_abs_shap": float(mean_shap[i])}
            for i in range(len(feature_names))
        ]
        return sorted(importance_list, key=lambda x: x["mean_abs_shap"], reverse=True)

    def get_local_explanation(self, outlet_id: str) -> Optional[Dict[str, Any]]:
        if self.shap_data is None:
            return None

        oids = self.shap_data["Outlet_ID"]
        idx_arr = np.where(oids == outlet_id)[0]
        if len(idx_arr) == 0:
            return None

        idx = int(idx_arr[0])
        shap_vals = self.shap_data["shap_values"][idx]
        features = self.shap_data["feature_names"]

        if "base_values" in self.shap_data:
            base_val = float(self.shap_data["base_values"][idx])
        else:
            base_val = float(self.shap_data.get("base_value", 0.0))

        if "prediction_log" in self.shap_data:
            pred_log = float(self.shap_data["prediction_log"][idx])
        else:
            pred_log = base_val + float(np.sum(shap_vals))

        feat_vals = self.shap_data.get("X_pred")
        contributions = []
        for i, name in enumerate(features):
            if feat_vals is not None and hasattr(feat_vals, "iloc"):
                val = feat_vals.iloc[idx].get(name, shap_vals[i]) if name in feat_vals.columns else shap_vals[i]
            else:
                val = shap_vals[i]
            contributions.append({
                "feature": name,
                "shap_value": float(shap_vals[i]),
                "feature_value": float(val) if isinstance(val, (int, float, np.number)) else str(val),
            })

        contributions = sorted(contributions, key=lambda x: abs(x["shap_value"]), reverse=True)
        return {
            "outlet_id": outlet_id,
            "explanation_type": self.get_explanation_type(),
            "base_value": base_val,
            "prediction_log": pred_log,
            "contributions": contributions,
        }

    def _top_drivers_text(self, outlet_id: str, n: int = 5) -> str:
        expl = self.get_local_explanation(outlet_id)
        if not expl:
            return "No attribution data available."
        lines = []
        for c in expl["contributions"][:n]:
            direction = "increases" if c["shap_value"] >= 0 else "decreases"
            lines.append(f"- {c['feature']}: {direction} potential (contribution {c['shap_value']:+.3f} log-liters)")
        return "\n".join(lines)

    def _build_grounded_prompt(self, outlet_data: Dict[str, Any]) -> str:
        outlet_id = outlet_data["Outlet_ID"]
        pred = float(outlet_data.get("Maximum_Monthly_Liters", 0))
        hist = float(outlet_data.get("hist_median_vol", 0))
        cens = float(outlet_data.get("censoring_score", 0))
        catch = float(outlet_data.get("combined_catchment_score", 0))
        comp = float(outlet_data.get("competitor_density_gaussian", 0))
        coolers = int(outlet_data.get("Cooler_Count", 0) or 0)
        yoy = float(outlet_data.get("yoy_growth", 0))
        drivers = self._top_drivers_text(outlet_id)

        return f"""You are a Senior Sales Analytics Advisor for Sri Lanka beverage distribution.
Use ONLY the facts below. Do not invent numbers.

OUTLET FACTS:
- Outlet ID: {outlet_id}
- Type: {outlet_data.get('Outlet_Type')} | Size: {outlet_data.get('Outlet_Size')}
- Distributor: {outlet_data.get('primary_dist')}
- Historical median volume: {hist:,.1f} L/month
- Predicted January 2026 latent potential: {pred:,.1f} L/month
- Censoring score: {cens:.2f} (0=unconstrained, 1=supply-capped)
- Catchment score: {catch:.2f} | Competitor density: {comp:.2f}
- Cooler count: {coolers} | YoY growth: {yoy*100:.1f}%

TOP MODEL DRIVERS (from latent heuristic attribution):
{drivers}

Respond with ONLY valid JSON (no markdown) using exactly these keys:
{{"WhyHigh": "...", "WhyLow": "...", "Opportunities": "...", "Risks": "...", "RecommendedActions": "..."}}"""

    @staticmethod
    def _parse_narrative_json(text: str) -> Dict[str, Any]:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                return json.loads(match.group())
            raise

    def _call_huggingface(self, prompt: str) -> Dict[str, Any]:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
        if not token:
            raise RuntimeError("HF_TOKEN not set")

        model_id = os.environ.get("HF_MODEL_ID", DEFAULT_HF_MODEL)
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "inputs": f"[INST] {prompt} [/INST]",
            "parameters": {
                "max_new_tokens": 700,
                "temperature": 0.2,
                "return_full_text": False,
            },
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, list) and result:
            text = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            text = result.get("generated_text") or result.get("generated_text", "")
        else:
            text = str(result)
        narrative = self._parse_narrative_json(text)
        narrative["_source"] = "huggingface"
        narrative["_model"] = model_id
        return narrative

    def _call_claude(self, prompt: str) -> Dict[str, Any]:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        data = {
            "model": os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            "max_tokens": 1000,
            "temperature": 0.2,
            "system": "You are a Senior Sales Analytics Advisor. Output JSON only.",
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        text = resp.json()["content"][0]["text"]
        narrative = self._parse_narrative_json(text)
        narrative["_source"] = "anthropic"
        return narrative

    def _rules_narrative(self, outlet_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deterministic fallback grounded on outlet thresholds."""
        pred = float(outlet_data.get("Maximum_Monthly_Liters", 0))
        hist = float(outlet_data.get("hist_median_vol", 0))
        cens = float(outlet_data.get("censoring_score", 0))
        catch = float(outlet_data.get("combined_catchment_score", 0))
        comp = float(outlet_data.get("competitor_density_gaussian", 0))
        coolers = int(outlet_data.get("Cooler_Count", 0) or 0)
        yoy = float(outlet_data.get("yoy_growth", 0))
        uplift = ((pred / (hist + 1e-9)) - 1) * 100 if hist > 0 else 0

        why_high = []
        if uplift > 15:
            why_high.append(f"Latent potential is {uplift:.0f}% above historical median ({hist:,.0f}L -> {pred:,.0f}L).")
        if catch > 0.40:
            why_high.append(f"Strong catchment score ({catch:.2f}) indicates high footfall drivers nearby.")
        if cens > 0.40:
            why_high.append(f"High censoring score ({cens:.2f}) suggests past sales were supply-capped, not demand-limited.")
        if not why_high:
            why_high.append("Stable baseline demand with moderate upside from seasonality and segment factors.")

        why_low = []
        if comp > 1.5:
            why_low.append(f"Competitive density ({comp:.2f}) may limit share gains.")
        if coolers == 0:
            why_low.append("No dedicated cooler limits chilled beverage capacity.")
        if cens > 0.30:
            why_low.append(f"Delivery/credit constraints (censoring {cens:.2f}) may still cap realized volume.")
        if not why_low:
            why_low.append("Segment size and type impose a natural ceiling on volume.")

        return {
            "WhyHigh": " ".join(why_high),
            "WhyLow": " ".join(why_low),
            "Opportunities": "Extend credit limits for high-censoring outlets; add cooler assets where Cooler_Count < 2; prioritize outlets with catchment score > 0.4.",
            "Risks": "Competitive promotions nearby; distributor stock-outs during peak months; overestimating uncapping if supply constraints persist.",
            "RecommendedActions": "1. Review distributor delivery cap. 2. Deploy trade spend on top uplift-gap outlets. 3. Monitor actual vs predicted monthly.",
            "_source": "rules_engine",
        }

    def get_advisor_narrative(self, outlet_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate grounded executive narrative. Priority: HF -> Claude -> rules."""
        outlet_id = outlet_data["Outlet_ID"]

        cache = {}
        if XAI_CACHE_PATH.exists():
            try:
                with open(XAI_CACHE_PATH, encoding="utf-8") as f:
                    cache = json.load(f)
            except Exception as e:
                logger.warning("Could not read XAI cache: %s", e)

        if outlet_id in cache:
            return cache[outlet_id]

        prompt = self._build_grounded_prompt(outlet_data)
        narrative = None

        if os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY"):
            try:
                logger.info("Calling Hugging Face Inference API for %s...", outlet_id)
                narrative = self._call_huggingface(prompt)
            except Exception as e:
                logger.error("Hugging Face request failed: %s", e)

        if narrative is None and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                logger.info("Calling Anthropic Claude for %s...", outlet_id)
                narrative = self._call_claude(prompt)
            except Exception as e:
                logger.error("Claude request failed: %s", e)

        if narrative is None:
            logger.info("Using rules-engine narrative for %s", outlet_id)
            narrative = self._rules_narrative(outlet_data)

        cache[outlet_id] = narrative
        try:
            with open(XAI_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Could not write XAI cache: %s", e)

        return narrative

    # Backward-compatible alias
    def get_claude_advisor_narrative(self, outlet_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.get_advisor_narrative(outlet_data)
