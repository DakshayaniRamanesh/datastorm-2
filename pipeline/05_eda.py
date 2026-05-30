"""
EDA Dashboard - Exploratory Data Analysis
==========================================
Produces 8-panel visualization dashboard with business annotations,
Q4 demand peak markers, and prediction distribution analysis.
Run after the full pipeline (Silver + Gold) has completed.

Panels
------
1. Log Volume Distribution (clean transactions)
2. Total Network Volume by Month (with Q4 annotations)
3. Outlet Type Distribution
4. Median Monthly Volume by Outlet Type
5. Outlet Geospatial Distribution (Sri Lanka)
6. Censoring Score Distribution
7. Prediction Distribution (Maximum_Monthly_Liters)
8. Censoring Score vs. Predicted Potential (scatter)

Usage:
    python pipeline/05_eda.py
"""

import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

ROOT   = Path(__file__).parent.parent
SILVER = ROOT / "pipeline" / "silver"
GOLD   = ROOT / "pipeline" / "gold"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared color palette
# ---------------------------------------------------------------------------
C_BLUE    = "#1f77b4"
C_RED     = "#d62728"
C_GREEN   = "#2ca02c"
C_GOLD    = "#ff7f0e"
C_PURPLE  = "#9467bd"
C_ORANGE  = "#ff7f0e"
C_BG      = "white"
C_PANEL   = "white"
C_BORDER  = "#cccccc"
C_TEXT    = "black"


def style_axes(ax):
    """Apply consistent dark theme to an axes object."""
    ax.set_facecolor(C_PANEL)
    ax.tick_params(colors=C_TEXT, labelsize=8)
    ax.xaxis.label.set_color(C_TEXT)
    ax.yaxis.label.set_color(C_TEXT)
    ax.title.set_color(C_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_BORDER)


def load_data():
    tx     = pd.read_parquet(SILVER / "transactions.parquet")
    outlet = pd.read_parquet(SILVER / "outlet_master.parquet")
    coords = pd.read_parquet(SILVER / "outlet_coordinates.parquet")
    return tx, outlet, coords


# ---------------------------------------------------------------------------
# Panel 1 — Log Volume Distribution
# ---------------------------------------------------------------------------
def plot_volume_distribution(tx, ax):
    vols = np.log1p(tx["Volume_Liters"])
    n, bins, patches = ax.hist(vols, bins=80, color=C_BLUE, alpha=0.85, edgecolor="none")
    # Annotate the peak bin
    peak_idx = np.argmax(n)
    peak_x   = (bins[peak_idx] + bins[peak_idx + 1]) / 2
    ax.axvline(peak_x, color=C_GOLD, linestyle="--", linewidth=1, alpha=0.7)
    ax.annotate(f"Peak\n{np.expm1(peak_x):.0f} L",
                xy=(peak_x, n[peak_idx]),
                xytext=(peak_x + 0.5, n[peak_idx] * 0.85),
                color=C_GOLD, fontsize=7,
                arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=0.8))
    ax.set_title("Log Volume Distribution (Clean)", fontsize=10, fontweight="bold")
    ax.set_xlabel("log(1 + Volume_Liters)", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)


# ---------------------------------------------------------------------------
# Panel 2 — Monthly trend with Q4 annotations
# ---------------------------------------------------------------------------
def plot_monthly_trend(tx, ax):
    monthly = tx.groupby(["Year", "Month"])["Volume_Liters"].sum().reset_index()
    monthly["period"] = (monthly["Year"].astype(str) + "-"
                         + monthly["Month"].astype(str).str.zfill(2))
    monthly = monthly.sort_values(["Year", "Month"]).reset_index(drop=True)
    xs = range(len(monthly))
    ax.plot(xs, monthly["Volume_Liters"] / 1e6, color=C_RED,
            linewidth=2, marker="o", markersize=2.5, zorder=3)

    # Shade Q4 months (Oct-Dec) and annotate
    for i, row in monthly.iterrows():
        if row["Month"] in [10, 11, 12]:
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.15, color=C_GOLD, zorder=1)

    # Add Q4 label on first Q4 cluster
    q4_first = monthly[(monthly["Month"] == 10)].index
    if len(q4_first) > 0:
        ax.annotate("Q4 Demand\nSeason ▶",
                    xy=(q4_first[0], monthly.loc[q4_first[0], "Volume_Liters"] / 1e6),
                    xytext=(q4_first[0] - 2, monthly["Volume_Liters"].max() / 1e6 * 0.95),
                    color=C_GOLD, fontsize=7,
                    arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=0.8))

    tick_positions = list(range(0, len(monthly), 6))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [monthly["period"].iloc[i] for i in tick_positions], rotation=45, fontsize=7
    )
    ax.set_title("Network Volume by Month (M Liters)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Period", fontsize=8)
    ax.set_ylabel("Volume (M L)", fontsize=8)

    q4_patch = mpatches.Patch(color=C_GOLD, alpha=0.4, label="Q4 Window")
    ax.legend(handles=[q4_patch], fontsize=7, loc="lower right")


# ---------------------------------------------------------------------------
# Panel 3 — Outlet Type Distribution
# ---------------------------------------------------------------------------
def plot_outlet_type_dist(outlet, ax):
    counts = outlet["Outlet_Type"].value_counts()
    colors = plt.cm.Set2(np.linspace(0, 1, len(counts)))
    bars   = ax.barh(counts.index, counts.values, color=colors, edgecolor="none")
    for bar, val in zip(bars, counts.values):
        ax.text(val + 10, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=7, color=C_TEXT)
    ax.set_title("Outlet Type Distribution", fontsize=10, fontweight="bold")
    ax.set_xlabel("Count", fontsize=8)


# ---------------------------------------------------------------------------
# Panel 4 — Median Volume by Type
# ---------------------------------------------------------------------------
def plot_volume_by_type(tx, outlet, ax):
    merged   = tx.merge(outlet[["Outlet_ID", "Outlet_Type"]], on="Outlet_ID")
    type_vol = (merged.groupby("Outlet_Type")["Volume_Liters"]
                .median().sort_values(ascending=False))
    colors   = plt.cm.viridis(np.linspace(0.3, 0.9, len(type_vol)))
    bars     = ax.bar(type_vol.index, type_vol.values, color=colors, edgecolor="none")
    for bar, val in zip(bars, type_vol.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}", ha="center", fontsize=7, color=C_TEXT)
    ax.set_title("Median Monthly Volume by Outlet Type", fontsize=10, fontweight="bold")
    ax.set_xlabel("Outlet Type", fontsize=8)
    ax.set_ylabel("Median Vol (L)", fontsize=8)
    ax.tick_params(axis="x", rotation=35)


# ---------------------------------------------------------------------------
# Panel 5 — Geo distribution
# ---------------------------------------------------------------------------
def plot_geo_distribution(coords, ax):
    ax.scatter(coords["Longitude"], coords["Latitude"],
               s=0.5, alpha=0.35, color=C_GREEN, zorder=2)
    ax.set_title("Outlet Geospatial Distribution (Sri Lanka)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    # Sri Lanka bounding box reference lines
    ax.axhline(6.0, color=C_BORDER, lw=0.5, ls="--")
    ax.axhline(9.8, color=C_BORDER, lw=0.5, ls="--")
    ax.text(81.8, 9.85, "N Limit", fontsize=6, color=C_BORDER)
    ax.text(81.8, 6.05, "S Limit", fontsize=6, color=C_BORDER)


# ---------------------------------------------------------------------------
# Panel 6 — Censoring Score Distribution
# ---------------------------------------------------------------------------
def plot_censoring_dist(ax):
    try:
        gold = pd.read_parquet(GOLD / "gold_features.parquet")
        scores = gold["censoring_score"]
        ax.hist(scores, bins=40, color=C_GOLD, alpha=0.85, edgecolor="none")
        ax.axvline(0.30, color=C_RED, linestyle="--", linewidth=1.5,
                   label=f"Threshold 0.30\n({(scores > 0.30).sum():,} constrained)")
        n_hi   = (scores > 0.30).sum()
        pct_hi = 100 * n_hi / len(scores)
        ax.text(0.32, ax.get_ylim()[1] * 0.80 if ax.get_ylim()[1] > 0 else 1,
                f"{pct_hi:.1f}%\nconstrained", color=C_RED, fontsize=8)
        ax.set_title("Censoring Score Distribution", fontsize=10, fontweight="bold")
        ax.set_xlabel("Censoring Score (0=unconstrained, 1=fully capped)", fontsize=7)
        ax.set_ylabel("Outlets", fontsize=8)
        ax.legend(fontsize=7)
    except Exception:
        ax.text(0.5, 0.5, "Run 04_gold first", ha="center", va="center",
                color=C_TEXT, fontsize=10)
        ax.set_title("Censoring Score Distribution", fontsize=10, fontweight="bold")


# ---------------------------------------------------------------------------
# Panel 7 — Prediction Distribution
# ---------------------------------------------------------------------------
def plot_prediction_dist(ax):
    try:
        gold = pd.read_parquet(GOLD / "gold_features.parquet")
        preds = gold["Maximum_Monthly_Liters"]
        ax.hist(np.log1p(preds), bins=60, color=C_PURPLE, alpha=0.85, edgecolor="none")
        # Quartile annotations
        for q, label, clr in [(0.25, "Q1", C_BLUE), (0.50, "Median", C_GREEN),
                               (0.75, "Q3", C_ORANGE)]:
            val = preds.quantile(q)
            ax.axvline(np.log1p(val), color=clr, linestyle="--", linewidth=1.2,
                       label=f"{label}: {val:,.0f} L")
        ax.set_title("Prediction Distribution (Jan 2026)", fontsize=10, fontweight="bold")
        ax.set_xlabel("log(1 + Maximum_Monthly_Liters)", fontsize=7)
        ax.set_ylabel("Outlets", fontsize=8)
        ax.legend(fontsize=6)
    except Exception:
        ax.text(0.5, 0.5, "Run 04_gold first", ha="center", va="center",
                color=C_TEXT, fontsize=10)
        ax.set_title("Prediction Distribution", fontsize=10, fontweight="bold")


# ---------------------------------------------------------------------------
# Panel 8 — Censoring Score vs. Predicted Potential
# ---------------------------------------------------------------------------
def plot_censoring_vs_potential(ax):
    try:
        gold = pd.read_parquet(GOLD / "gold_features.parquet")
        sample = gold.sample(min(5000, len(gold)), random_state=42)
        sc = ax.scatter(
            sample["censoring_score"],
            np.log1p(sample["Maximum_Monthly_Liters"]),
            c=sample["potential_multiplier"],
            cmap="plasma", s=3, alpha=0.5, vmin=1.0, vmax=4.0
        )
        plt.colorbar(sc, ax=ax, label="Potential Multiplier", pad=0.02)
        ax.axvline(0.30, color=C_RED, linestyle="--", linewidth=1, alpha=0.7,
                   label="Censoring threshold")
        ax.set_title("Censoring Score vs. Predicted Potential", fontsize=10, fontweight="bold")
        ax.set_xlabel("Censoring Score", fontsize=8)
        ax.set_ylabel("log(1 + Predicted Liters)", fontsize=8)
        ax.legend(fontsize=7)
    except Exception:
        ax.text(0.5, 0.5, "Run 04_gold first", ha="center", va="center",
                color=C_TEXT, fontsize=10)
        ax.set_title("Censoring vs. Potential", fontsize=10, fontweight="bold")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("EDA - Generating 8-Panel Analysis Dashboard")
    print("=" * 60)

    tx, outlet, coords = load_data()
    monthly = tx.groupby(["Outlet_ID", "Year", "Month"])["Volume_Liters"].sum().reset_index()
    monthly.rename(columns={"Volume_Liters": "monthly_volume"}, inplace=True)

    fig = plt.figure(figsize=(20, 14), facecolor=C_BG)
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1:3])   # wider for trend
    ax3 = fig.add_subplot(gs[0, 3])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[1, 3])

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        style_axes(ax)

    plot_volume_distribution(tx, ax1)
    plot_monthly_trend(tx, ax2)
    plot_outlet_type_dist(outlet, ax3)
    plot_volume_by_type(tx, outlet, ax4)
    plot_geo_distribution(coords, ax5)
    plot_censoring_dist(ax6)
    plot_prediction_dist(ax7)

    # Panel 8 — Censoring vs. potential (overlaid on panel 6 position isn't possible
    # with this layout, so we save it as a separate file)
    fig.suptitle("DataStorm 2026 — EDA & Potential Estimation Dashboard",
                 fontsize=15, color=C_TEXT, fontweight="bold", y=0.995)

    main_path = OUTPUT / "eda_dashboard.png"
    fig.savefig(main_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  Saved: {main_path}")

    # Supplementary scatter plot
    fig2, ax8 = plt.subplots(figsize=(8, 5), facecolor=C_BG)
    style_axes(ax8)
    plot_censoring_vs_potential(ax8)
    fig2.suptitle("DataStorm 2026 — Censoring vs. Predicted Potential",
                  fontsize=12, color=C_TEXT, fontweight="bold")
    scatter_path = OUTPUT / "eda_censoring_scatter.png"
    fig2.savefig(scatter_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig2)
    print(f"  Saved: {scatter_path}")

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    print("\n[STATS] Key Statistics:")
    print(f"  Clean transaction rows:    {len(tx):,}")
    print(f"  Unique outlets:            {tx['Outlet_ID'].nunique():,}")
    print(f"  Date range:                2023-01 to 2025-12")
    print(f"  Volume range:              {tx['Volume_Liters'].min():.2f} - "
          f"{tx['Volume_Liters'].max():.2f} L")
    print(f"  Median monthly vol/outlet: "
          f"{monthly.groupby('Outlet_ID')['monthly_volume'].median().median():.2f} L")

    try:
        gold          = pd.read_parquet(GOLD / "gold_features.parquet")
        n_constrained = (gold["censoring_score"] > 0.30).sum()
        print(f"\n  Outlets flagged constrained (>0.30): {n_constrained:,} "
              f"({100*n_constrained/len(gold):.1f}%)")
        print(f"  Avg potential multiplier:  {gold['potential_multiplier'].mean():.2f}x")
        print(f"  Max potential multiplier:  {gold['potential_multiplier'].max():.2f}x")
        print(f"  Prediction median:         {gold['Maximum_Monthly_Liters'].median():,.1f} L")
        print(f"  Prediction 90th pctile:    {gold['Maximum_Monthly_Liters'].quantile(0.9):,.1f} L")
    except Exception:
        pass

    print("\n[OK]  EDA complete.\n")


if __name__ == "__main__":
    main()
