"""
DataStorm 2026 - Shared budget optimization parameters
======================================================
Single source of truth for pipeline and web app allocators.
"""

# Western Province pilot budget (LKR)
BUDGET_LKR = 5_000_000.0

# Per-outlet cap for distributor parity (LKR)
MAX_SPEND_PER_OUTLET_LKR = 100_000.0

# Diminishing-returns trade spend response: Lift_i = gap_i * ALPHA * ln(1 + Spend_i / BETA)
ALPHA = 0.15
BETA = 10_000.0

# Bisection solver tolerance (iterations)
BISection_MAX_ITER = 100
