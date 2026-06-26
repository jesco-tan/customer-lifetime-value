"""Plotting helpers for CLV project, Thornbury & Co. scenario."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path


sns.set_theme(style="whitegrid", palette="muted")

# Colour palette consistent across all channel plots
CHANNEL_COLOURS = {
    "Enthusiast": "#2166ac",
    "Mainstream": "#4dac26",
    "Deal-Seeker": "#d7191c",
    "Overall": "#888888",
    "Unknown": "#aaaaaa",
}


# ── Existing plots ────────────────────────────────────────────────────────────

def plot_retention_heatmap(
    retention: pd.DataFrame,
    title: str = "Cohort Retention",
    save_path: Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, max(6, len(retention) * 0.4)))
    mask = retention.isna()
    sns.heatmap(
        retention,
        annot=True,
        fmt=".0%",
        cmap="YlGnBu",
        mask=mask,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Retention Rate"},
    )
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Period Index (0 = acquisition cohort)")
    ax.set_ylabel("Acquisition Cohort")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_clv_distribution(
    clv_df: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure:
    vals = clv_df["predicted_clv"].dropna()
    cap = vals.quantile(0.99)
    vals_capped = vals.clip(upper=cap)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].hist(vals_capped, bins=50, color="#4472C4", edgecolor="white")
    axes[0].set_title("12-Month CLV Distribution (99th pct cap)")
    axes[0].set_xlabel("Predicted CLV (£)")
    axes[0].set_ylabel("Customers")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))

    prob_alive = clv_df["prob_alive"]
    axes[1].hist(prob_alive, bins=40, color="#ED7D31", edgecolor="white")
    axes[1].set_title("Probability Customer is Still Active")
    axes[1].set_xlabel("P(alive)")
    axes[1].set_ylabel("Customers")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_segment_bar(
    segment_summary: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("tab10", n_colors=len(segment_summary))
    ax.barh(segment_summary["Segment"], segment_summary["Total_Revenue"], color=colors)
    ax.set_xlabel("Total Revenue (£)")
    ax.set_title("Revenue by RFM Segment")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_umap_scatter(
    rfm: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure | None:
    if rfm["UMAP1"].isna().all():
        return None
    fig, ax = plt.subplots(figsize=(9, 7))
    segments = rfm["Segment"].unique()
    palette = dict(zip(segments, sns.color_palette("tab10", n_colors=len(segments))))
    for seg, grp in rfm.groupby("Segment"):
        ax.scatter(grp["UMAP1"], grp["UMAP2"], label=seg, alpha=0.6, s=10, color=palette[seg])
    ax.set_title("Customer Segments, UMAP Projection")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=3, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── New disaggregation plots ──────────────────────────────────────────────────

def plot_channel_clv_boxplot(
    clv_df: pd.DataFrame,
    channel_map: pd.Series,
    save_path: Path | None = None,
) -> plt.Figure:
    """Violin plot of predicted CLV by acquisition channel.

    The Deal-Seeker violin sits near zero despite occasional high-AOV purchases,
    illustrating why P(alive) is the critical moderator.
    """
    df = clv_df.copy()
    df["Channel"] = df.index.map(channel_map)
    df = df.dropna(subset=["Channel", "predicted_clv"])

    channel_order = ["Enthusiast", "Mainstream", "Deal-Seeker"]
    present = [c for c in channel_order if c in df["Channel"].unique()]
    if not present:
        present = sorted(df["Channel"].unique())

    cap = df["predicted_clv"].quantile(0.98)
    df["predicted_clv_capped"] = df["predicted_clv"].clip(upper=cap)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: violin by channel
    colours = [CHANNEL_COLOURS.get(c, "#888888") for c in present]
    parts = axes[0].violinplot(
        [df.loc[df["Channel"] == c, "predicted_clv_capped"].values for c in present],
        positions=range(len(present)),
        showmedians=True,
        showextrema=True,
    )
    for i, (pc, col) in enumerate(zip(parts["bodies"], colours)):
        pc.set_facecolor(col)
        pc.set_alpha(0.7)
    axes[0].set_xticks(range(len(present)))
    axes[0].set_xticklabels(present)
    axes[0].set_title("12-Month Predicted CLV by Channel (98th pct cap)")
    axes[0].set_ylabel("Predicted CLV (£)")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))

    # Right: P(alive) by channel
    for ch in present:
        sub = df[df["Channel"] == ch]["prob_alive"]
        axes[1].hist(sub, bins=30, alpha=0.6, label=ch, color=CHANNEL_COLOURS.get(ch, "#888"))
    axes[1].set_title("P(alive) Distribution by Channel")
    axes[1].set_xlabel("P(alive)")
    axes[1].set_ylabel("Customers")
    axes[1].legend()
    axes[1].axvline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
    axes[1].text(0.51, axes[1].get_ylim()[1] * 0.9, "Alive threshold", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_rfm_vs_clv_tension(
    rfm_df: pd.DataFrame,
    clv_df: pd.DataFrame,
    channel_map: pd.Series,
    save_path: Path | None = None,
) -> plt.Figure:
    """Scatter plot: RFM Monetary score (M) vs predicted CLV, coloured by channel.

    The analytical tension lives here: Deal-Seekers cluster in the high-M /
    low-CLV quadrant. A naive M-score ranking would prioritise the wrong segment
    for win-back spend.
    """
    rfm = rfm_df.set_index("Customer ID") if "Customer ID" in rfm_df.columns else rfm_df
    clv = clv_df.copy()
    clv["Channel"] = clv.index.map(channel_map)

    merged = rfm[["M", "Monetary"]].join(
        clv[["predicted_clv", "prob_alive", "Channel"]], how="inner"
    )
    merged = merged.dropna(subset=["M", "predicted_clv", "Channel"])

    cap_clv = merged["predicted_clv"].quantile(0.98)
    merged["predicted_clv_capped"] = merged["predicted_clv"].clip(upper=cap_clv)

    channel_order = ["Enthusiast", "Mainstream", "Deal-Seeker"]
    present = [c for c in channel_order if c in merged["Channel"].unique()]
    if not present:
        present = sorted(merged["Channel"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: M score vs predicted CLV
    for ch in present:
        sub = merged[merged["Channel"] == ch]
        axes[0].scatter(
            sub["M"] + np.random.default_rng(0).uniform(-0.25, 0.25, len(sub)),
            sub["predicted_clv_capped"],
            alpha=0.25, s=8,
            color=CHANNEL_COLOURS.get(ch, "#888"),
            label=ch,
        )
    axes[0].set_xlabel("RFM Monetary Score (M, 1=lowest → 5=highest)")
    axes[0].set_ylabel("Predicted 12-Month CLV (£)")
    axes[0].set_title("The Tension: M Score vs Predicted CLV by Channel")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    axes[0].legend(markerscale=3)

    # Right: mean CLV vs mean M score per channel (summary dots)
    ch_stats = merged.groupby("Channel").agg(
        mean_M=("M", "mean"),
        mean_CLV=("predicted_clv", "mean"),
        mean_prob_alive=("prob_alive", "mean"),
    ).reset_index()

    for _, row in ch_stats.iterrows():
        ch = row["Channel"]
        col = CHANNEL_COLOURS.get(ch, "#888")
        axes[1].scatter(row["mean_M"], row["mean_CLV"], color=col, s=200, zorder=5)
        axes[1].annotate(
            f"{ch}\n(P(alive)={row['mean_prob_alive']:.0%})",
            (row["mean_M"], row["mean_CLV"]),
            textcoords="offset points", xytext=(8, 4), fontsize=9,
        )

    axes[1].set_xlabel("Mean M Score")
    axes[1].set_ylabel("Mean Predicted CLV (£)")
    axes[1].set_title("Channel Summary: Mean M vs Mean CLV")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_month1_retention_by_channel(
    retention_summary: pd.DataFrame,
    guardrail: float = 0.20,
    save_path: Path | None = None,
) -> plt.Figure:
    """Bar chart of mean Month-1 retention by channel with guardrail threshold line.

    Channels whose bar falls below the red dashed line breach the guardrail.
    """
    df = retention_summary[retention_summary["Channel"] != "Overall"].copy()
    channel_order = ["Enthusiast", "Mainstream", "Deal-Seeker"]
    present = [c for c in channel_order if c in df["Channel"].values]
    if not present:
        present = df["Channel"].tolist()

    df = df.set_index("Channel").reindex(present).reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    colours = [CHANNEL_COLOURS.get(c, "#888") for c in df["Channel"]]
    bars = ax.bar(df["Channel"], df["mean_month1_retention"], color=colours, edgecolor="white", width=0.5)

    ax.axhline(guardrail, color="#d7191c", linestyle="--", linewidth=2, label=f"Guardrail ({guardrail:.0%})")

    for bar, val in zip(bars, df["mean_month1_retention"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.1%}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylim(0, max(df["mean_month1_retention"].max() * 1.25, guardrail * 1.5))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_title("Mean Month-1 Cohort Retention by Acquisition Channel")
    ax.set_ylabel("Month-1 Retention Rate")
    ax.set_xlabel("Acquisition Channel")
    ax.legend()

    # Annotate guardrail breach
    for bar, val, ch in zip(bars, df["mean_month1_retention"], df["Channel"]):
        if val < guardrail:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                guardrail / 2,
                "BREACH",
                ha="center", va="center",
                fontsize=10, color="white", fontweight="bold",
            )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_holdout_validation(
    comparison_df: pd.DataFrame,
    channel_map: pd.Series | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Compare BG/NBD predicted vs actual purchases in the holdout period.

    Left: scatter of predicted vs actual (diagonal = perfect prediction).
    Right: mean absolute error by channel (if channel_map provided).
    """
    df = comparison_df.dropna(subset=["predicted_purchases_holdout", "actual_purchases_holdout"])
    cap = df["actual_purchases_holdout"].quantile(0.98)
    df = df[df["actual_purchases_holdout"] <= cap]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: scatter
    axes[0].scatter(
        df["actual_purchases_holdout"],
        df["predicted_purchases_holdout"],
        alpha=0.15, s=6, color="#4472C4",
    )
    maxval = max(df["actual_purchases_holdout"].max(), df["predicted_purchases_holdout"].max())
    axes[0].plot([0, maxval], [0, maxval], "r--", linewidth=1.5, label="Perfect prediction")
    axes[0].set_xlabel("Actual Purchases (holdout)")
    axes[0].set_ylabel("BG/NBD Predicted Purchases (holdout)")
    axes[0].set_title("Holdout Validation: Predicted vs Actual")
    axes[0].legend()

    # Overall error stats
    mae = np.abs(df["error"]).mean()
    axes[0].text(
        0.05, 0.92,
        f"MAE = {mae:.3f} purchases",
        transform=axes[0].transAxes,
        fontsize=10, bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    # Right: MAE by channel (or distribution of errors if no channel)
    if channel_map is not None and not channel_map.empty:
        df2 = df.copy()
        df2["Channel"] = df2.index.map(channel_map)
        channel_mae = df2.groupby("Channel")["error"].apply(
            lambda x: np.abs(x).mean()
        ).reset_index(name="MAE")

        channel_order = ["Enthusiast", "Mainstream", "Deal-Seeker"]
        present = [c for c in channel_order if c in channel_mae["Channel"].values]
        channel_mae = channel_mae.set_index("Channel").reindex(present).reset_index()
        colours = [CHANNEL_COLOURS.get(c, "#888") for c in channel_mae["Channel"]]

        axes[1].bar(channel_mae["Channel"], channel_mae["MAE"], color=colours, edgecolor="white", width=0.5)
        axes[1].set_title("Holdout MAE by Acquisition Channel")
        axes[1].set_ylabel("Mean Absolute Error (purchases)")
        axes[1].set_xlabel("Acquisition Channel")

        for bar, val in zip(axes[1].patches, channel_mae["MAE"]):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=10,
            )
    else:
        axes[1].hist(df["error"], bins=40, color="#4472C4", edgecolor="white")
        axes[1].axvline(0, color="red", linestyle="--")
        axes[1].set_title("Distribution of Prediction Errors")
        axes[1].set_xlabel("Predicted − Actual Purchases")
        axes[1].set_ylabel("Customers")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_bgnbd_matrices(
    bgf,
    max_frequency: int = 15,
    max_recency: int = 400,
    horizon: int = 365,
    save_path: Path | None = None,
) -> plt.Figure:
    """Recreate BG/NBD frequency-recency and P(alive) matrices.

    Replaces lifetimes.plotting functions that break under newer matplotlib.
    Uses vectorized BGF calls to avoid scalar-vs-array issues.
    """
    T = float(max_recency)
    freq_range = np.arange(0, max_frequency + 1, dtype=float)
    rec_range = np.linspace(0, max_recency, 41)

    # Meshgrid: shape (n_rec, n_freq) after meshgrid
    F, R = np.meshgrid(freq_range, rec_range)
    T_arr = np.full(F.size, T)

    raw_purchases = bgf.conditional_expected_number_of_purchases_up_to_time(
        horizon, F.ravel(), R.ravel(), T_arr
    )
    raw_alive = bgf.conditional_probability_alive(F.ravel(), R.ravel(), T_arr)

    Z_purchases = np.array(raw_purchases, dtype=float).reshape(F.shape).T  # (n_freq, n_rec)
    Z_alive = np.array(raw_alive, dtype=float).reshape(F.shape).T

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].imshow(
        Z_purchases, interpolation="none", aspect="auto",
        extent=[float(rec_range[0]), float(rec_range[-1]),
                float(freq_range[-1]) + 0.5, float(freq_range[0]) - 0.5],
        origin="upper",
    )
    plt.colorbar(im0, ax=axes[0], label="Expected purchases")
    axes[0].set_xlabel("Recency (days since first purchase)")
    axes[0].set_ylabel("Frequency (repeat purchases)")
    axes[0].set_title(f"Expected Purchases (next {horizon} days), F vs R")

    im1 = axes[1].imshow(
        Z_alive, interpolation="none", aspect="auto",
        extent=[float(rec_range[0]), float(rec_range[-1]),
                float(freq_range[-1]) + 0.5, float(freq_range[0]) - 0.5],
        vmin=0, vmax=1, origin="upper",
    )
    plt.colorbar(im1, ax=axes[1], label="P(alive)")
    axes[1].set_xlabel("Recency (days since first purchase)")
    axes[1].set_ylabel("Frequency (repeat purchases)")
    axes[1].set_title("P(Alive), Frequency vs Recency")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_campaign_roi(
    roi_df: pd.DataFrame,
    guardrail: float = 0.0,
    save_path: Path | None = None,
) -> plt.Figure:
    """Horizontal bar chart of estimated win-back campaign ROI by channel.

    Negative ROI bars are red; positive are green. Guardrail at ROI=0.
    """
    df = roi_df.copy().sort_values("roi")
    colours = [
        CHANNEL_COLOURS.get(c, "#888") if v >= guardrail else "#d7191c"
        for c, v in zip(df["Channel"], df["roi"])
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    bars = axes[0].barh(df["Channel"], df["roi"], color=colours, edgecolor="white")
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_xlabel("Estimated ROI per Contact (£ return per £ spent)")
    axes[0].set_title("Win-Back Campaign ROI by Channel")
    for bar, val in zip(bars, df["roi"]):
        x = bar.get_width()
        axes[0].text(
            x + (0.05 if x >= 0 else -0.05),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}×",
            va="center", ha="left" if x >= 0 else "right", fontsize=10, fontweight="bold",
        )

    axes[1].barh(df["Channel"], df["expected_return_per_contact"],
                 color=colours, edgecolor="white", label="Expected Return")
    axes[1].axvline(df["cost_per_contact"].iloc[0], color="black", linestyle="--",
                    linewidth=1.5, label=f"Cost per contact (£{df['cost_per_contact'].iloc[0]:.2f})")
    axes[1].set_xlabel("£ per Contact")
    axes[1].set_title("Expected Return vs Cost per Contact")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
