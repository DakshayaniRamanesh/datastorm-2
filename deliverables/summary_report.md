# DataStorm 2026: Latent Potential Prediction
## Technical Summary Report
**Team:** Antigravity  
**Date:** May 2026  

---

### 1. Data Forensics and Hygiene
In the legacy SFA and distributor ERP systems, we identified several significant anomalies that would have biased a naive predictive model.

#### 1.1 system Anomalies Trapped
*   **Negative/Zero Volumes:** Approximately 0.2% of transactions had negative or zero volumes, likely representing credit notes, returns, or automated "ghost entries" from SFA sync errors. These were quarantined.
*   **Extreme Outliers:** We identified transaction records exceeding 5x the IQR fence (per SKU). These were flagged as manual data-entry errors (e.g., aggregation errors) and moved to the rejected store to prevent "potential" inflation.
*   **GPS Dropouts:** Over 1% of outlet coordinates were outside the bounding box of Sri Lanka (or at 0,0). These GPS artifacts were isolated to ensure accurate POI catchment analysis.

#### 1.2 Lakehouse Implementation (Bronze -> Silver -> Gold)
We implemented a rigid Lakehouse architecture to ensure auditability:
*   **Bronze:** Bit-perfect parity with source CSVs.
*   **Silver:** Applied parameterizable DQ checks (`check_duplicates`, `check_nulls`, `check_referential_integrity`). Records failing these were saved in `pipeline/rejected/` with a specific `dq_failure_reason`.
*   **Gold:** Enriched with historical features, POI signals, and model-ready aggregations.

---

### 2. POI Data Acquisition
Catchment drivers are the primary indicators of "Maximum Potential" independent of historical performance.

#### 2.1 Technical Approach
We utilized the **Overpass API (OpenStreetMap)** to dynamically scrape Points of Interest within a 1km radius of every valid outlet.
*   **Targeted POIs:** Schools/Universities, Bus Stations/Stops, Hospitals, Tourist Attractions, Markets, Places of Worship, Fuel Stations, and Restaurants.
*   **Mapping Method:** We computed a weighted "POI Catchment Score" where high-impulse locations (Bus stands, Schools) received higher weights (2.0) compared to competing supply signals (Markets, 0.5).
*   **Caching:** To ensure reproducibility and speed, we implemented a local JSON cache (`pipeline/poi_cache/`) that stores results per coordinate-category pair.

---

### 3. Causal Base Logic & Methodology
The core challenge was solving the **left-censored demand curve**.

#### 3.1 The Censoring Score (The "Uncapping" Signal)
We moved beyond simple "hits-max" logic to a 5-signal composite score to identify capped demand:
1.  **Plateau Detection:** Run of months with CV < 10% indicates the outlet is "stuck" at a ceiling.
2.  **Distributor Cap Proxy:** Outlet volume tracking exactly with distributor median delivery patterns.
3.  **Inter-year Stagnation:** Low growth (<5%) despite being in a high-growth region.
4.  **Low CV relative to peers:** Organic demand is usually "noisy"; high consistency is a signature of a system constraint (e.g., a credit limit).
5.  **Q4 Suppression:** Lack of seasonal uplift in beverage-heavy Q4 months.

#### 3.2 Maximum Potential Calculation
We used a **Probabilistic Ceiling Estimation** approach:
$$Potential = Base \times Multiplier \times Seasonality \times Growth$$

*   **SFA Proxy:** We applied Stochastic Frontier Analysis logic to calculate the "Efficiency Gap" between an outlet and the top 10% performers in its specific segment (Type + Size + Region).
*   **Multiplier:** The base potential was uplifted based on the **Censoring Score**, **POI Catchment Score**, and **SFA Efficiency Gap**.
*   **Ceiling Cap:** We applied a logical hard cap of 5.0x historical median to prevent wild extrapolations.

---

### 4. GenAI Transparency Log
Generative AI (Antigravity/LLM) was used as a **Data Engineering Accelerator** rather than a "black box" predictor.

| Phase | AI Application | Rationale |
| :--- | :--- | :--- |
| **Brainstorming** | Causal framework development | To identify non-obvious censoring signals (e.g., Q4 suppression). |
| **Boilerplate** | DQ check functions | Rapid implementation of reusable, parameterizable DE code. |
| **Scraping** | Overpass QL Querying | Generating complex Overpass queries for specific OSM tags. |
| **Refactoring** | Unicode Fixes | Solving Windows encoding issues (cp1252 vs UTF-8) in the pipeline. |
| **Validation** | Sanity checking math | Iteratively prompting the LLM to explain the "Tobit-inspired" ceiling math to ensure logical defensibility. |

**Human-in-the-loop:** All AI-generated code was rigorously validated against the raw data artifacts. For example, the initial IQR fence was too aggressive; we iteratively adjusted it after inspecting the "rejected" logs.
