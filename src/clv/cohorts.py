"""Cohort retention matrix builder with channel disaggregation."""

from __future__ import annotations

import pandas as pd
import numpy as np


def build_retention_matrix(df: pd.DataFrame, freq: str = "M", observation_end=None) -> pd.DataFrame:
    """
    Build a cohort × period retention matrix.

    Parameters
    ----------
    df : DataFrame with columns [Customer ID, InvoiceDate]
    freq : 'M' for monthly cohorts, 'W' for weekly

    Returns
    -------
    retention : DataFrame where index = cohort period, columns = period index (0, 1, 2, ...)
                values = retention rate (0–1)
    """
    df = df.copy()
    df["period"] = df["InvoiceDate"].dt.to_period(freq)
    cohort_map = df.groupby("Customer ID")["period"].min().rename("cohort")
    df = df.join(cohort_map, on="Customer ID")
    df["period_index"] = (df["period"] - df["cohort"]).apply(lambda x: x.n)
    cohort_sizes = df.groupby("cohort")["Customer ID"].nunique()
    retention_counts = df.groupby(["cohort", "period_index"])["Customer ID"].nunique()
    end = pd.Timestamp(observation_end if observation_end is not None else df.InvoiceDate.max()).to_period(freq)
    if end < df.period.max():
        raise ValueError("Observation end precedes a transaction")
    max_age = max((end - c).n for c in cohort_sizes.index)
    retention = retention_counts.unstack("period_index").reindex(columns=range(max_age+1)).divide(cohort_sizes, axis=0)
    for cohort in retention.index:
        observed = (end - cohort).n
        retention.loc[cohort, :observed] = retention.loc[cohort, :observed].fillna(0)
        retention.loc[cohort, observed+1:] = np.nan
    return retention


def cohort_revenue_matrix(df: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """Average revenue per customer per cohort-period."""
    df = df.copy()
    df["period"] = df["InvoiceDate"].dt.to_period(freq)
    cohort_map = df.groupby("Customer ID")["period"].min().rename("cohort")
    df = df.join(cohort_map, on="Customer ID")
    df["period_index"] = (df["period"] - df["cohort"]).apply(lambda x: x.n)
    revenue = df.groupby(["cohort", "period_index"])["Revenue"].sum()
    cohort_sizes = df.groupby("cohort")["Customer ID"].nunique()
    avg_revenue = revenue.unstack("period_index").divide(cohort_sizes, axis=0)
    return avg_revenue


def retention_by_channel(
    df: pd.DataFrame,
    freq: str = "M",
) -> dict[str, pd.DataFrame]:
    """Build separate retention matrices for each acquisition channel.

    Returns
    -------
    dict mapping channel name → retention DataFrame (same format as build_retention_matrix)
    Returns empty dict if Channel column is missing.
    """
    if "Channel" not in df.columns:
        return {}

    result: dict[str, pd.DataFrame] = {}
    for channel, grp in df.groupby("Channel"):
        result[str(channel)] = build_retention_matrix(grp, freq=freq, observation_end=df.InvoiceDate.max())
    return result


def month1_retention_summary(df: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """Return Month-1 retention rate per channel (and overall) as a summary table.

    Month-1 retention = fraction of the acquisition cohort that made at least one
    additional purchase in the period immediately following their acquisition period.

    Returns
    -------
    DataFrame with columns: Channel, mean_month1_retention, median_month1_retention,
    n_cohorts, n_customers
    """
    channels = df["Channel"].unique().tolist() if "Channel" in df.columns else ["All"]
    rows = []

    def _month1(sub_df: pd.DataFrame) -> dict:
        ret = build_retention_matrix(sub_df, freq=freq, observation_end=df.InvoiceDate.max())
        if 1 not in ret.columns:
            return {"mean": np.nan, "median": np.nan, "n_cohorts": 0}
        col = ret[1].dropna()
        return {
            "mean": float(col.mean()),
            "median": float(col.median()),
            "n_cohorts": len(col),
        }

    if "Channel" in df.columns:
        for ch in sorted(df["Channel"].unique()):
            sub = df[df["Channel"] == ch]
            stats = _month1(sub)
            rows.append({
                "Channel": ch,
                "mean_month1_retention": round(stats["mean"], 4),
                "median_month1_retention": round(stats["median"], 4),
                "n_cohorts": stats["n_cohorts"],
                "n_customers": int(sub["Customer ID"].nunique()),
            })

    stats_all = _month1(df)
    rows.append({
        "Channel": "Overall",
        "mean_month1_retention": round(stats_all["mean"], 4),
        "median_month1_retention": round(stats_all["median"], 4),
        "n_cohorts": stats_all["n_cohorts"],
        "n_customers": int(df["Customer ID"].nunique()),
    })

    return pd.DataFrame(rows)
