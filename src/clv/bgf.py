"""BG/NBD purchase model + Gamma-Gamma spend model wrappers.

Includes descriptive channel aggregation and a hypothetical revenue-weighting
heuristic. Neither predicts incremental campaign ROI or manages an actual budget.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from lifetimes import BetaGeoFitter, GammaGammaFitter


def fit_bgnbd(summary: pd.DataFrame, penalizer_coef: float = 0.001) -> BetaGeoFitter:
    bgf = BetaGeoFitter(penalizer_coef=penalizer_coef)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])
    return bgf


def fit_gamma_gamma(summary: pd.DataFrame, penalizer_coef: float = 0.001) -> GammaGammaFitter:
    repeat = summary[summary["frequency"] > 0]
    ggf = GammaGammaFitter(penalizer_coef=penalizer_coef)
    ggf.fit(repeat["frequency"], repeat["monetary_value"])
    return ggf


def predict_clv(
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter,
    summary: pd.DataFrame,
    horizon_days: int = 365,
    discount_rate: float = 0.01,
) -> pd.DataFrame:
    """Append nominal annual discounted revenue, purchase count and model survival.

    Nonrepeaters are assigned zero value by policy, not by a Gamma-Gamma theorem.
    The fractional monthly horizon can expand to 390 days in lifetimes 0.11.3;
    do not claim exact 365-day value without reconciling endpoint semantics.
    """
    df = summary.copy()

    df["predicted_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        horizon_days,
        df["frequency"],
        df["recency"],
        df["T"],
    )

    df["prob_alive"] = bgf.conditional_probability_alive(
        df["frequency"],
        df["recency"],
        df["T"],
    )

    repeat_mask = df["frequency"] > 0
    df.loc[repeat_mask, "predicted_clv"] = ggf.customer_lifetime_value(
        bgf,
        df.loc[repeat_mask, "frequency"],
        df.loc[repeat_mask, "recency"],
        df.loc[repeat_mask, "T"],
        df.loc[repeat_mask, "monetary_value"],
        time=horizon_days / 30,
        discount_rate=discount_rate,
        freq="D",
    )
    df.loc[~repeat_mask, "predicted_clv"] = 0.0

    return df


def clv_summary_stats(clv_df: pd.DataFrame) -> dict:
    vals = clv_df["predicted_clv"].dropna()
    return {
        "mean_clv": round(float(vals.mean()), 2),
        "median_clv": round(float(vals.median()), 2),
        "p90_clv": round(float(vals.quantile(0.90)), 2),
        "total_predicted_revenue": round(float(vals.sum()), 2),
        "pct_alive": round(float((clv_df["prob_alive"] > 0.5).mean() * 100), 1),
    }


def clv_by_channel(clv_df: pd.DataFrame, channel_map: pd.Series) -> pd.DataFrame:
    """Aggregate predicted CLV and P(alive) statistics per acquisition channel.

    Parameters
    ----------
    clv_df : DataFrame indexed by Customer ID, output of predict_clv()
    channel_map : Series mapping Customer ID → Channel

    Returns
    -------
    DataFrame with one row per channel and columns:
    n_customers, mean_clv, median_clv, p90_clv, total_clv,
    mean_prob_alive, pct_alive, mean_predicted_purchases
    """
    df = clv_df.copy()
    df["Channel"] = df.index.map(channel_map)

    result = (
        df.groupby("Channel")
        .agg(
            n_customers=("predicted_clv", "count"),
            mean_clv=("predicted_clv", "mean"),
            median_clv=("predicted_clv", "median"),
            p90_clv=("predicted_clv", lambda x: x.quantile(0.90)),
            total_clv=("predicted_clv", "sum"),
            mean_monetary=("monetary_value", "mean"),
            mean_prob_alive=("prob_alive", "mean"),
            pct_alive=("prob_alive", lambda x: round((x > 0.5).mean() * 100, 1)),
            mean_predicted_purchases=("predicted_purchases", "mean"),
        )
        .round(2)
        .reset_index()
    )
    return result


def holdout_comparison(
    bgf: BetaGeoFitter,
    cal_summary: pd.DataFrame,
    hold_df: pd.DataFrame,
    cal_end: pd.Timestamp,
    obs_end: pd.Timestamp,
) -> pd.DataFrame:
    """Compare BG/NBD predicted purchases vs actual purchases in holdout period.

    Returns a DataFrame indexed by Customer ID with columns:
    predicted_purchases_holdout, actual_purchases_holdout, error
    """
    holdout_days = (obs_end - cal_end).days

    predicted = bgf.conditional_expected_number_of_purchases_up_to_time(
        holdout_days,
        cal_summary["frequency"],
        cal_summary["recency"],
        cal_summary["T"],
    )

    actual = hold_df.groupby("Customer ID")["Invoice"].nunique()
    actual = actual.reindex(cal_summary.index, fill_value=0)

    df = pd.DataFrame({
        "predicted_purchases_holdout": predicted,
        "actual_purchases_holdout": actual,
    })
    df["error"] = df["predicted_purchases_holdout"] - df["actual_purchases_holdout"]
    return df


def campaign_heuristic_by_channel(
    clv_df: pd.DataFrame,
    channel_map: pd.Series,
    cost_per_contact: float,
    response_rates: dict[str, float],
) -> pd.DataFrame:
    """Illustrative revenue-weighting heuristic, not incremental campaign ROI.

    Multiplies mean value by the fraction above a survival cutoff and an assumed
    response rate. Purchase predictions already integrate survival; extra
    weighting may discount twice. No contacted/uncontacted effect is estimated.
    This score cannot justify campaign allocation or exclusion.
    """
    channel_stats = clv_by_channel(clv_df, channel_map)

    rows = []
    for _, row in channel_stats.iterrows():
        ch = row["Channel"]
        rr = response_rates.get(ch, 0.05)
        expected_return = row["mean_clv"] * (row["pct_alive"] / 100) * rr
        net_per_contact = expected_return - cost_per_contact
        roi = net_per_contact / cost_per_contact if cost_per_contact > 0 else np.nan
        max_contactable = int(row["n_customers"])
        rows.append({
            "Channel": ch,
            "n_customers": max_contactable,
            "response_rate": rr,
            "mean_clv": row["mean_clv"],
            "pct_alive": row["pct_alive"],
            "revenue_proxy_per_contact": round(expected_return, 2),
            "cost_per_contact": cost_per_contact,
            "net_per_contact": round(net_per_contact, 2),
            "heuristic_score": round(roi, 2),
        })

    return pd.DataFrame(rows).sort_values("heuristic_score", ascending=False).reset_index(drop=True)


def campaign_roi_by_channel(*args, **kwargs):
    """Deprecated compatibility alias returning labelled heuristic columns.

    There is no ROI column because incremental returns are not identified.
    """
    import warnings
    warnings.warn("Use campaign_heuristic_by_channel; this is not campaign ROI",
                  DeprecationWarning, stacklevel=2)
    return campaign_heuristic_by_channel(*args, **kwargs)
