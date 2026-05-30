"""
DataStorm 2026 - SHAP & Claude XAI Service
==========================================
Manages SHAP local and global explanation calculations, and integrates
Anthropic Claude API to generate Senior Sales Advisory reports on outlets.
Caches all AI outputs to local JSON store to prevent duplicate API spend.
"""

import os
import json
import pickle
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger("XAIService")

XAI_CACHE_PATH = Path(__file__).parent.parent.parent / "data" / "xai_cache.json"
XAI_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

class XAIService:
    def __init__(self, shap_pickle_path: Path = Path(__file__).parent.parent.parent / "pipeline" / "gold" / "shap_explanations.pkl"):
        self.shap_path = shap_pickle_path
        self.shap_data = None
        self._load_shap_data()
        
    def _load_shap_data(self) -> None:
        """Load SHAP explanation pack from modeling output."""
        if not self.shap_path.exists():
            logger.warning(f"SHAP explanations file not found at: {self.shap_path}. Explainability charts will be unavailable.")
            return
            
        try:
            with open(self.shap_path, "rb") as f:
                self.shap_data = pickle.load(f)
            logger.info("SHAP explanations loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading SHAP explanations: {e}")

    def get_global_importance(self) -> List[Dict[str, Any]]:
        """Compute mean absolute SHAP values across all features for global importance chart."""
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
        """Retrieve local SHAP feature contributions for a specific outlet."""
        if self.shap_data is None:
            return None
            
        oids = self.shap_data["Outlet_ID"]
        # Find index
        idx_arr = np.where(oids == outlet_id)[0]
        if len(idx_arr) == 0:
            return None
            
        idx = int(idx_arr[0])
        shap_vals = self.shap_data["shap_values"][idx]
        features = self.shap_data["feature_names"]
        feat_vals = self.shap_data["X_pred"].iloc[idx]
        
        contributions = []
        for i in range(len(features)):
            val = feat_vals.iloc[i]
            contributions.append({
                "feature": features[i],
                "shap_value": float(shap_vals[i]),
                "feature_value": float(val) if isinstance(val, (int, float, np.number)) else str(val)
            })
            
        # Sort contributions by absolute value descending
        contributions = sorted(contributions, key=lambda x: abs(x["shap_value"]), reverse=True)
        
        return {
            "outlet_id": outlet_id,
            "base_value": float(self.shap_data["base_value"]),
            "prediction_log": float(self.shap_data["base_value"] + np.sum(shap_vals)),
            "contributions": contributions
        }

    def get_claude_advisor_narrative(self, outlet_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch Senior Sales Advisor narratives.

        Uses Anthropic Claude if configured, otherwise falls back to a high-fidelity local rules-engine.
        Caches outputs locally in JSON.
        """
        outlet_id = outlet_data["Outlet_ID"]
        
        # 1. Load XAI Cache
        cache = {}
        if XAI_CACHE_PATH.exists():
            try:
                with open(XAI_CACHE_PATH, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read XAI cache: {e}")
                
        if outlet_id in cache:
            return cache[outlet_id]

        # 2. Extract context variables
        pred = outlet_data["Maximum_Monthly_Liters"]
        hist = outlet_data["hist_median_vol"]
        cens = outlet_data["censoring_score"]
        catch = outlet_data.get("combined_catchment_score", 0.0)
        comp = outlet_data.get("competitor_density_gaussian", 0.0)
        coolers = outlet_data.get("Cooler_Count", 0)
        yoy = outlet_data.get("yoy_growth", 0.0)
        
        # Prepare context prompt
        prompt_content = f"""
        Role: Senior Sales Analytics Advisor for Sri Lanka Beverage Distribution.
        
        Outlet Profile:
        - Outlet ID: {outlet_id}
        - Outlet Type: {outlet_data['Outlet_Type']}
        - Outlet Size: {outlet_data['Outlet_Size']}
        - Distributor ID: {outlet_data['primary_dist']}
        - Historical Median Volume: {hist:,.1f} Liters/month
        - Predicted January 2026 Potential: {pred:,.1f} Liters/month
        - Systemic Censoring Score: {cens:.2f} (0=unconstrained, 1=fully capped)
        - Catchment Score: {catch:.2f} (0=poor, 1=excellent)
        - Competitor Density Score: {comp:.2f}
        - Cooler Count: {coolers}
        - YoY Sales Growth: {yoy*100:.1f}%
        """

        # 3. Check for Anthropic API key
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            logger.info(f"Calling Anthropic Claude API for outlet {outlet_id} narrative...")
            try:
                # Perform API call (requests post to Anthropic API)
                url = "https://api.anthropic.com/v1/messages"
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
                data = {
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 1000,
                    "temperature": 0.2,
                    "system": "You are a Senior Sales Analytics Advisor for Sri Lanka Beverage Distribution. Provide structured business advice.",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{prompt_content}\nAnalyze this outlet and structure your output exactly as a JSON object with these 5 keys:\n1. WhyHigh (String explaining why the potential is high)\n2. WhyLow (String explaining limiting factors)\n3. Opportunities (String listing growth opportunities)\n4. Risks (String listing risks)\n5. RecommendedActions (String listing actionable recommendations)"
                        }
                    ]
                }
                resp = requests.post(url, headers=headers, json=data, timeout=30)
                resp.raise_for_status()
                text = resp.json()["content"][0]["text"]
                # Parse JSON
                # Clean code blocks if returned
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                narrative = json.loads(text.strip())
                
                # Cache and return
                cache[outlet_id] = narrative
                with open(XAI_CACHE_PATH, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)
                return narrative
            except Exception as e:
                logger.error(f"Claude API request failed: {e}. Falling back to rules engine.")

        # 4. Fallback: High-Quality Local Geospatial Rules Engine
        logger.info(f"Generating local rules-based narrative for outlet {outlet_id}...")
        
        # Heuristics for why high
        why_high_factors = []
        if catch > 0.40:
            why_high_factors.append(f"Outstanding spatial catchment score ({catch:.2f}) with high school/restaurant/transit density driving retail footfall.")
        if coolers >= 2:
            why_high_factors.append(f"Robust chilled storage capacity (Cooler Count: {coolers}) supporting high SKU diversity and serving speed.")
        if cens > 0.40:
            why_high_factors.append(f"Uncapped latent demand: a high censoring score ({cens:.2f}) indicates past sales volumes were severely bottlenecked by delivery caps or credit limits.")
        if yoy > 0.10:
            why_high_factors.append(f"Strong positive growth momentum (YoY: {yoy*100:.1f}%), showing expanding local market share.")
        if not why_high_factors:
            why_high_factors.append("Steady core neighborhood demand with solid repeat purchase behavior and consistent distributor alignments.")
            
        why_high = " ".join(why_high_factors)

        # Heuristics for why low / bottlenecked
        why_low_factors = []
        if cens > 0.30:
            why_low_factors.append(f"Severe delivery constraints and supply caps (Censoring: {cens:.2f}) are capping peak sales.")
        if comp > 1.5:
            why_low_factors.append(f"High local competition density ({comp:.2f} competitors within 1km) dampens the market penetration rate.")
        if coolers == 0:
            why_low_factors.append("Lack of dedicated cold storage (Cooler Count: 0) severely limits warm-weather carbonated beverage sales potential.")
        if yoy < -0.05:
            why_low_factors.append(f"Declining sales trajectory (YoY: {yoy*100:.1f}%) suggests local demographic shifts or distributor friction.")
        if not why_low_factors:
            why_low_factors.append("Limited by general category boundaries and average size category footprint.")
            
        why_low = " ".join(why_low_factors)

        # Opportunities
        opportunities_factors = []
        if coolers < 2:
            opportunities_factors.append("Opportunity to deploy additional corporate cooler assets to unlock high-margin chilled single-serve volumes.")
        if cens > 0.30:
            opportunities_factors.append("Unlock potential by lifting monthly distributor delivery limits or extending flexible credit lines during peak holiday periods.")
        if catch > 0.50:
            opportunities_factors.append("High footfall hub: introduce premium SKUs and seasonal impulse racks near the checkout counters.")
        if opportunities_factors:
            opportunities = " ".join(opportunities_factors)
        else:
            opportunities = "Introduce volume discounts on top-selling SKUs and optimize distributor delivery schedules to guarantee 100% stock availability."

        # Risks
        risks_factors = []
        if comp > 2.0:
            risks_factors.append("Extreme market saturation: competitors could launch aggressive price promotions, eroding customer loyalty.")
        if cens > 0.60:
            risks_factors.append("High risk of persistent stockouts. Systemic delivery deficits may force local retailers to switch to competing brands.")
        if yoy < -0.10:
            risks_factors.append("Significant structural decline. Risk of total account churn if operational bottlenecks are not resolved immediately.")
        if not risks_factors:
            risks_factors.append("Friction in distributor alignment or stock-outs during peak seasonal demand spikes in Q4.")
            
        risks = " ".join(risks_factors)

        # Recommended Actions
        actions = []
        if cens > 0.30:
            actions.append("1. Increase distributor monthly delivery limits by +25% immediately to prevent supply caps.")
        if coolers == 0:
            actions.append("2. Deploy a branded single-door cooler asset on a lease-to-own high-volume sales contract.")
        elif coolers == 1:
            actions.append("2. Upgrade the current cooler asset to a double-door high-capacity refrigerator.")
        if comp > 1.5:
            actions.append("3. Provide custom localized point-of-sale display banners and outdoor merchandising to stand out from nearby convenience stores.")
        else:
            actions.append("3. Launch an exclusive retailer loyalty incentive program based on quarterly volume milestones.")
            
        rec_actions = " ".join(actions)

        narrative = {
            "WhyHigh": why_high,
            "WhyLow": why_low,
            "Opportunities": opportunities,
            "Risks": risks,
            "RecommendedActions": rec_actions
        }

        # Cache on disk
        cache[outlet_id] = narrative
        try:
            with open(XAI_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Could not write cache: {e}")

        return narrative
