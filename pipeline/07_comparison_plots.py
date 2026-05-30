"""
COMPARISON PLOTS - Historical Input vs Predicted Potential
===========================================================
Generates diagnostic comparison charts between:
  - Historical observed monthly volumes (input)
  - Predicted Maximum_Monthly_Liters (output)

Purpose: Visualise model efficiency and validate that variance in predictions
is driven by meaningful signals (censoring, outlet type/size, POI catchment)
rather than random noise.

Panels Generated
----------------
1.  Scatter: Historical Median Vol vs Predicted (per outlet, coloured by censoring score)
2.  Uplift Ratio Distribution (Predicted / Historical) — where is the model adding value?
3.  Box plots: Predicted vs Historical by Outlet Type
4.  Box plots: Predicted vs Historical by Outlet Size
5.  Uplift by Censoring Score Decile (bar chart) — are constrained outlets being lifted?
6.  Residual Variance: (Predicted - Historical) vs Historical (heteroscedasticity check)
7.  Cumulative Distribution: Historical vs Predicted (overlay)
8.  Potential Multiplier Decomposition (stacked bar by component)

Usage:
    python pipeline/07_comparison_plots.py
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

ROOT    = Path(__file__).parent.parent
GOLD    = ROOT / "pipeline" / "gold"
SILVER  = ROOT / "pipeline" / "silver"
OUTPUT  = ROOT / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
C_BG     = "white"
C_PANEL  = "white"
C_BORDER = "#cccccc"
C_TEXT   = "black"
C_HIST   = "#1f77b4"   # historical — blue
C_PRED   = "#d62728"   # predicted  — red
C_GOLD   = "#ff7f0e"
C_GREEN  = "#2ca02c"
C_PURPLE = "#9467bd"
C_ORANGE = "#ff7f0e"


def style_ax(ax):
    ax.set_facecolor(C_PANEL)
    ax.tick_params(colors=C_TEXT, labelsize=8)
    ax.xaxis.label.set_color(C_TEXT)
    ax.yaxis.label.set_color(C_TEXT)
    ax.title.set_color(C_TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(C_BORDER)


def load_gold() -> pd.DataFrame:
    path = GOLD / "gold_features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "Gold features not found. Run `python run_pipeline.py` first."
        )
    df = pd.read_parquet(path)
    print(f"  Loaded gold_features.parquet: {len(df):,} outlets")
    return df


# ---------------------------------------------------------------------------
# Panel 1 — Scatter: Historical Median vs Predicted
# ---------------------------------------------------------------------------
def plot_scatter(df, ax):
    sample = df.sample(min(8000, len(df)), random_state=42)
    sc = ax.scatter(
        np.log1p(sample["hist_median_vol"]),
        np.log1p(sample["Maximum_Monthly_Liters"]),
        c=sample["censoring_score"],
        cmap="plasma", s=4, alpha=0.55,
        vmin=0, vmax=1,
    )
    plt.colorbar(sc, ax=ax, label="Censoring Score", pad=0.02, fraction=0.046)

    # Identity line (prediction = history — no uplift)
    lims = [0, max(
        np.log1p(df["hist_median_vol"].max()),
        np.log1p(df["Maximum_Monthly_Liters"].max()),
    )]
    ax.plot(lims, lims, "--", color=C_GREEN, lw=1.2, alpha=0.7,
            label="No-uplift line (y=x)")
    ax.set_title("Historical Median vs Predicted Potential", fontsize=10, fontweight="bold")
    ax.set_xlabel("log(1 + Historical Median Vol) [L]", fontsize=8)
    ax.set_ylabel("log(1 + Predicted Jan 2026) [L]", fontsize=8)
    ax.legend(fontsize=7)

    # Annotation
    pct_above = 100 * (df["Maximum_Monthly_Liters"] > df["hist_median_vol"]).mean()
    ax.text(0.05, 0.92, f"{pct_above:.1f}% of outlets\npredicted above historical",
            transform=ax.transAxes, color=C_GOLD, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C_PANEL, edgecolor=C_BORDER))


# ---------------------------------------------------------------------------
# Panel 2 — Uplift Ratio Distribution
# ---------------------------------------------------------------------------
def plot_uplift_dist(df, ax):
    df = df.copy()
    df["uplift_ratio"] = df["Maximum_Monthly_Liters"] / (df["hist_median_vol"] + 1e-9)
    df["uplift_ratio"] = df["uplift_ratio"].clip(0, 6)

    ax.hist(df["uplift_ratio"], bins=60, color=C_PURPLE, alpha=0.85, edgecolor="none")
    ax.axvline(1.0, color=C_GREEN, lw=1.5, linestyle="--", label="No uplift (ratio=1)")
    ax.axvline(df["uplift_ratio"].median(), color=C_GOLD, lw=1.5, linestyle="-.",
               label=f"Median: {df['uplift_ratio'].median():.2f}x")
    ax.axvline(df["uplift_ratio"].quantile(0.90), color=C_ORANGE, lw=1.2, linestyle=":",
               label=f"P90: {df['uplift_ratio'].quantile(0.90):.2f}x")

    pct_gt2 = 100 * (df["uplift_ratio"] > 2).mean()
    ax.text(0.60, 0.85, f"{pct_gt2:.1f}% outlets\n>2x uplift",
            transform=ax.transAxes, color=C_ORANGE, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C_PANEL, edgecolor=C_BORDER))
    ax.set_title("Uplift Ratio Distribution (Predicted / Historical)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Uplift Ratio (Predicted ÷ Historical Median)", fontsize=8)
    ax.set_ylabel("Outlets", fontsize=8)
    ax.legend(fontsize=7)


# ---------------------------------------------------------------------------
# Panel 3 — Box plots by Outlet Type
# ---------------------------------------------------------------------------
def plot_by_type(df, ax):
    types = df["Outlet_Type"].dropna().unique()
    types = sorted(types)

    positions = np.arange(len(types))
    w = 0.35

    for idx, ot in enumerate(types):
        sub = df[df["Outlet_Type"] == ot]
        hist_vals = np.log1p(sub["hist_median_vol"].values)
        pred_vals = np.log1p(sub["Maximum_Monthly_Liters"].values)

        bp1 = ax.boxplot(hist_vals, positions=[idx - w/2], widths=0.28,
                         patch_artist=True, manage_ticks=False,
                         medianprops=dict(color="white", lw=1.5),
                         boxprops=dict(facecolor=C_HIST, alpha=0.7),
                         whiskerprops=dict(color=C_HIST),
                         capprops=dict(color=C_HIST),
                         flierprops=dict(marker=".", color=C_HIST, alpha=0.2, markersize=2))

        bp2 = ax.boxplot(pred_vals, positions=[idx + w/2], widths=0.28,
                         patch_artist=True, manage_ticks=False,
                         medianprops=dict(color="white", lw=1.5),
                         boxprops=dict(facecolor=C_PRED, alpha=0.7),
                         whiskerprops=dict(color=C_PRED),
                         capprops=dict(color=C_PRED),
                         flierprops=dict(marker=".", color=C_PRED, alpha=0.2, markersize=2))

    ax.set_xticks(positions)
    ax.set_xticklabels(types, rotation=35, fontsize=7)
    ax.set_title("Historical vs Predicted by Outlet Type", fontsize=10, fontweight="bold")
    ax.set_xlabel("Outlet Type", fontsize=8)
    ax.set_ylabel("log(1 + Volume) [L]", fontsize=8)

    hist_patch = mpatches.Patch(color=C_HIST, alpha=0.8, label="Historical Median")
    pred_patch = mpatches.Patch(color=C_PRED, alpha=0.8, label="Predicted Jan 2026")
    ax.legend(handles=[hist_patch, pred_patch], fontsize=7)


# ---------------------------------------------------------------------------
# Panel 4 — Box plots by Outlet Size
# ---------------------------------------------------------------------------
def plot_by_size(df, ax):
    size_order = ["Small", "Medium", "Large", "Extra Large"]
    sizes      = [s for s in size_order if s in df["Outlet_Size"].values]

    positions = np.arange(len(sizes))
    w = 0.35

    for idx, sz in enumerate(sizes):
        sub = df[df["Outlet_Size"] == sz]
        hist_vals = np.log1p(sub["hist_median_vol"].values)
        pred_vals = np.log1p(sub["Maximum_Monthly_Liters"].values)

        ax.boxplot(hist_vals, positions=[idx - w/2], widths=0.28,
                   patch_artist=True, manage_ticks=False,
                   medianprops=dict(color="white", lw=1.5),
                   boxprops=dict(facecolor=C_HIST, alpha=0.7),
                   whiskerprops=dict(color=C_HIST), capprops=dict(color=C_HIST),
                   flierprops=dict(marker=".", color=C_HIST, alpha=0.2, markersize=2))

        ax.boxplot(pred_vals, positions=[idx + w/2], widths=0.28,
                   patch_artist=True, manage_ticks=False,
                   medianprops=dict(color="white", lw=1.5),
                   boxprops=dict(facecolor=C_PRED, alpha=0.7),
                   whiskerprops=dict(color=C_PRED), capprops=dict(color=C_PRED),
                   flierprops=dict(marker=".", color=C_PRED, alpha=0.2, markersize=2))

    ax.set_xticks(positions)
    ax.set_xticklabels(sizes, fontsize=8)
    ax.set_title("Historical vs Predicted by Outlet Size", fontsize=10, fontweight="bold")
    ax.set_xlabel("Outlet Size", fontsize=8)
    ax.set_ylabel("log(1 + Volume) [L]", fontsize=8)

    hist_patch = mpatches.Patch(color=C_HIST, alpha=0.8, label="Historical Median")
    pred_patch = mpatches.Patch(color=C_PRED, alpha=0.8, label="Predicted Jan 2026")
    ax.legend(handles=[hist_patch, pred_patch], fontsize=7)


# ---------------------------------------------------------------------------
# Panel 5 — Uplift by Censoring Score Decile
# ---------------------------------------------------------------------------
def plot_uplift_by_censoring(df, ax):
    df = df.copy()
    df["uplift_ratio"]     = (df["Maximum_Monthly_Liters"]
                               / (df["hist_median_vol"] + 1e-9)).clip(0, 6)
    df["cens_decile"]      = pd.qcut(df["censoring_score"], q=10,
                                     labels=False, duplicates="drop") + 1

    summary = df.groupby("cens_decile").agg(
        mean_uplift=("uplift_ratio", "mean"),
        median_uplift=("uplift_ratio", "median"),
        mean_cens=("censoring_score", "mean"),
    ).reset_index()

    xs = summary["cens_decile"].values
    bars = ax.bar(xs, summary["mean_uplift"], color=C_PURPLE, alpha=0.8,
                  edgecolor="none", label="Mean uplift")
    ax.plot(xs, summary["median_uplift"], "o--", color=C_GOLD, lw=1.5,
            markersize=5, label="Median uplift")
    ax.axhline(1.0, color=C_GREEN, lw=1, ls="--", alpha=0.7, label="No-uplift line")

    for bar, val in zip(bars, summary["mean_uplift"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}x", ha="center", va="bottom", fontsize=6.5, color=C_TEXT)

    ax.set_title("Uplift by Censoring Score Decile", fontsize=10, fontweight="bold")
    ax.set_xlabel("Censoring Decile (1=low constraint → 10=high constraint)", fontsize=7)
    ax.set_ylabel("Uplift Ratio (Predicted ÷ Historical)", fontsize=8)
    ax.legend(fontsize=7)


# ---------------------------------------------------------------------------
# Panel 6 — Residual Variance (Predicted - Historical) vs Historical
# ---------------------------------------------------------------------------
def plot_residual_variance(df, ax):
    sample = df.sample(min(6000, len(df)), random_state=7)
    residual = sample["Maximum_Monthly_Liters"] - sample["hist_median_vol"]
    hist_log = np.log1p(sample["hist_median_vol"])

    ax.scatter(hist_log, residual, s=3, alpha=0.35,
               c=sample["censoring_score"], cmap="coolwarm", vmin=0, vmax=1)
    ax.axhline(0, color=C_GREEN, lw=1.2, ls="--", label="Zero residual")

    # Rolling mean of residuals
    order     = np.argsort(hist_log.values)
    x_sorted  = hist_log.values[order]
    r_sorted  = residual.values[order]
    window    = max(1, len(r_sorted) // 40)
    roll_mean = pd.Series(r_sorted).rolling(window, center=True, min_periods=1).mean().values
    ax.plot(x_sorted, roll_mean, color=C_GOLD, lw=1.5, label="Rolling mean residual")

    ax.set_title("Residual Variance: (Predicted − Historical) vs Historical",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("log(1 + Historical Median Vol) [L]", fontsize=8)
    ax.set_ylabel("Residual (Predicted − Historical) [L]", fontsize=8)
    ax.legend(fontsize=7)
    pct_pos = 100 * (residual > 0).mean()
    ax.text(0.65, 0.90, f"{pct_pos:.1f}% positive residuals",
            transform=ax.transAxes, color=C_GOLD, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C_PANEL, edgecolor=C_BORDER))


# ---------------------------------------------------------------------------
# Panel 7 — Cumulative Distribution: Historical vs Predicted
# ---------------------------------------------------------------------------
def plot_cdf_comparison(df, ax):
    hist_sorted = np.sort(df["hist_median_vol"].clip(upper=2000).values)
    pred_sorted = np.sort(df["Maximum_Monthly_Liters"].clip(upper=2000).values)
    cdf         = np.linspace(0, 1, len(hist_sorted))

    ax.plot(hist_sorted, cdf, color=C_HIST, lw=2, label="Historical Median")
    ax.plot(pred_sorted, cdf, color=C_PRED, lw=2, label="Predicted Jan 2026")

    # Shade the gap — that's the "uncapped demand" area
    ax.fill_betweenx(cdf, hist_sorted, pred_sorted,
                     where=(pred_sorted >= hist_sorted),
                     alpha=0.15, color=C_GOLD, label="Uncapped demand gap")

    ax.set_title("CDF: Historical vs Predicted (capped at 2000 L)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Monthly Volume [L]", fontsize=8)
    ax.set_ylabel("Cumulative Fraction of Outlets", fontsize=8)
    ax.legend(fontsize=7)

    # Key percentile markers
    for p, lbl in [(0.5, "P50"), (0.9, "P90")]:
        h_val = np.percentile(df["hist_median_vol"], p * 100)
        r_val = np.percentile(df["Maximum_Monthly_Liters"], p * 100)
        ax.annotate("", xy=(r_val, p), xytext=(h_val, p),
                    arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=1.2))
        ax.text((h_val + r_val) / 2, p + 0.01, lbl, ha="center",
                fontsize=7, color=C_GOLD)


# ---------------------------------------------------------------------------
# Panel 8 — Potential Multiplier Decomposition
# ---------------------------------------------------------------------------
def plot_multiplier_decomposition(df, ax):
    """
    Show how much each multiplier component contributes to the overall uplift,
    grouped by outlet type. Helps identify which signal is driving predictions.
    """
    # Compute component contributions as (factor - 1.0) expressed as % uplift
    df = df.copy()
    components = {
        "Size":        ("size_factor",        C_HIST),
        "Type":        ("type_factor",        C_GREEN),
        "Censoring":   ("censoring_uplift",   C_PURPLE),
        "Behavioral":  ("behavioral_uplift",  C_ORANGE),
        "POI":         ("poi_uplift",         C_GOLD),
        "SFA Peer":    ("sfa_uplift",         C_PRED),
    }

    types    = sorted(df["Outlet_Type"].dropna().unique())
    x        = np.arange(len(types))
    bar_w    = 0.12
    offsets  = np.linspace(-(len(components) - 1) * bar_w / 2,
                            (len(components) - 1) * bar_w / 2,
                            len(components))

    for (label, (col, color)), offset in zip(components.items(), offsets):
        if col not in df.columns:
            continue
        means = [df[df["Outlet_Type"] == t][col].mean() for t in types]
        ax.bar(x + offset, means, width=bar_w, color=color, alpha=0.8,
               edgecolor="none", label=label)

    ax.axhline(1.0, color="white", lw=0.8, ls="--", alpha=0.5, label="Neutral (1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=35, fontsize=7)
    ax.set_title("Multiplier Decomposition by Outlet Type", fontsize=10, fontweight="bold")
    ax.set_xlabel("Outlet Type", fontsize=8)
    ax.set_ylabel("Mean Factor Value", fontsize=8)
    ax.legend(fontsize=6, ncol=3)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("COMPARISON PLOTS - Historical Input vs Predicted Output")
    print("=" * 60)

    print("\n[1/3] Loading gold features...")
    df = load_gold()

    # -----------------------------------------------------------------------
    # Print summary statistics first
    # -----------------------------------------------------------------------
    print("\n[2/3] Key comparison statistics:")
    df["uplift_ratio"] = (
        df["Maximum_Monthly_Liters"] / (df["hist_median_vol"] + 1e-9)
    ).clip(0, 10)

    print(f"\n  {'Metric':<35} {'Historical':>15}  {'Predicted':>15}")
    print(f"  {'-'*65}")
    print(f"  {'Mean monthly volume (L)':<35} {df['hist_median_vol'].mean():>15,.1f}  "
          f"{df['Maximum_Monthly_Liters'].mean():>15,.1f}")
    print(f"  {'Median monthly volume (L)':<35} {df['hist_median_vol'].median():>15,.1f}  "
          f"{df['Maximum_Monthly_Liters'].median():>15,.1f}")
    print(f"  {'90th percentile (L)':<35} {df['hist_median_vol'].quantile(0.9):>15,.1f}  "
          f"{df['Maximum_Monthly_Liters'].quantile(0.9):>15,.1f}")
    print(f"  {'Std deviation':<35} {df['hist_median_vol'].std():>15,.1f}  "
          f"{df['Maximum_Monthly_Liters'].std():>15,.1f}")
    print(f"  {'CV (std/mean)':<35} "
          f"{df['hist_median_vol'].std()/df['hist_median_vol'].mean():>15.3f}  "
          f"{df['Maximum_Monthly_Liters'].std()/df['Maximum_Monthly_Liters'].mean():>15.3f}")
    print(f"\n  {'Uplift ratio — mean':<35} {df['uplift_ratio'].mean():>15.3f}x")
    print(f"  {'Uplift ratio — median':<35} {df['uplift_ratio'].median():>15.3f}x")
    print(f"  {'% outlets predicted > historical':<35} "
          f"{100*(df['Maximum_Monthly_Liters']>df['hist_median_vol']).mean():>15.1f}%")
    print(f"  {'% outlets with >2x uplift':<35} "
          f"{100*(df['uplift_ratio']>2).mean():>15.1f}%")

    # -----------------------------------------------------------------------
    # Build the 8-panel comparison figure
    # -----------------------------------------------------------------------
    print("\n[3/3] Generating 8-panel comparison figure...")
    fig = plt.figure(figsize=(22, 16), facecolor=C_BG)
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.40)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    ax5 = fig.add_subplot(gs[1, 0])
    ax6 = fig.add_subplot(gs[1, 1])
    ax7 = fig.add_subplot(gs[1, 2])
    ax8 = fig.add_subplot(gs[1, 3])

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        style_ax(ax)

    plot_scatter(df, ax1)
    plot_uplift_dist(df, ax2)
    plot_by_type(df, ax3)
    plot_by_size(df, ax4)
    plot_uplift_by_censoring(df, ax5)
    plot_residual_variance(df, ax6)
    plot_cdf_comparison(df, ax7)
    plot_multiplier_decomposition(df, ax8)

    fig.suptitle(
        "DataStorm 2026 — Input vs Prediction Comparison\n"
        "Historical Observed Sales  ↔  Latent Demand Estimate (Jan 2026)",
        fontsize=14, color=C_TEXT, fontweight="bold", y=0.995
    )

    out_path = OUTPUT / "comparison_analysis.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)

    print(f"\n  Saved: {out_path}")
    print("\n[OK]  Comparison plots complete.\n")


if __name__ == "__main__":
    main()
