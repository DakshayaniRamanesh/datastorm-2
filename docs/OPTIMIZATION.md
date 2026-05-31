# Western Province Trade Spend Optimization (LKR 5M)

**Shared config:** `pipeline/optimization_config.py` (used by pipeline and Flask app).

## Objective

Maximize total incremental volume lift:

\[
\max \sum_i \text{gap}_i \cdot \alpha \cdot \ln\!\left(1 + \frac{s_i}{\beta}\right)
\]

Where:

- \(\text{gap}_i = \max(\text{Maximum\_Monthly\_Liters}_i - \text{hist\_median\_vol}_i,\, 0)\)
- \(\alpha = 0.15\), \(\beta = 10{,}000\) LKR (diminishing returns)
- \(s_i\) = trade spend at outlet \(i\)

## Constraints

1. \(\sum_i s_i \leq 5{,}000{,}000\) LKR
2. \(0 \leq s_i \leq 100{,}000\) LKR (per-outlet parity cap)
3. Outlets filtered to Western Province: `primary_dist` like `DIST_W_%`

## Solver

Dual bisection on KKT multiplier (`allocate_budget_bisection` in `pipeline/05_budget_optimizer.py`) — convex, fast, exact for the bounded formulation.

## Parameter rationale

| Parameter | Value | Role |
|-----------|-------|------|
| α | 0.15 | Scales marginal lift per liter of gap |
| β | 10,000 | Spend scale for log diminishing returns; higher β → more spread before saturation |
| Max/outlet | 100,000 | Prevents one outlet consuming entire pilot budget |

### Sensitivity (illustrative)

If **β doubles** (20,000), marginal lift per extra LKR spent at low spend levels **decreases** — allocations spread more evenly across outlets before hitting the cap.

If **α increases**, the solver pushes spend toward high-gap outlets until the budget or per-outlet cap binds.

Empirical calibration to promo response data is future work; parameters are documented and shared across pipeline + UI.

## Expected lift & ROI (reporting)

\[
\text{Expected\_Lift}_i = \text{gap}_i \cdot \alpha \cdot \ln(1 + s_i / \beta)
\]

\[
\text{ROI}_i = \text{Expected\_Lift}_i / s_i
\]

Revenue framing uses median LKR/liter from silver transactions.
