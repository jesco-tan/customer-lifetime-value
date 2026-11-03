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


def expected_purchases(bgf, days, summary):
    """Signed hypergeometric form of BG/NBD equation 10.

    lifetimes 0.11.3 logs hyp2f1; when its value is negative this yields NaN
    despite a finite positive expectation. Preserve its sign instead.
    """
    from scipy.special import hyp2f1
    r, alpha, a, b = (float(bgf.params_[k]) for k in ("r","alpha","a","b"))
    x=summary.frequency.to_numpy();T=summary["T"].to_numpy();rec=summary.recency.to_numpy()
    if abs(a-1)<1e-8:
        raise ValueError("BG/NBD a near 1 requires a limit calculation")
    z=days/(alpha+T+days)
    term=hyp2f1(r+x,b+x,a+b+x-1,z)*((alpha+T)/(alpha+T+days))**(r+x)
    denom=1+np.where(x>0,a/(b+x-1)*((alpha+T)/(alpha+rec))**(r+x),0.)
    result=(a+b+x-1)/(a-1)*(1-term)/denom
    if not np.isfinite(result).all() or (result < -1e-8).any():
        raise ValueError("Nonfinite or negative BG/NBD purchase prediction")
    return np.maximum(result,0.)


def predict_clv(
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter,
    summary: pd.DataFrame,
    horizon_days: int = 365,
    discount_rate: float = 0.01,
    fallback_spend: pd.Series | float = 0.0,
) -> pd.DataFrame:
    """Exact daily discounted revenue for days 1..horizon_days inclusive.

    Monthly discount convention: (1 + discount_rate) ** (day / 30).
    Spend fallback must be supplied from training data for non-zero policies.
    """
    if not isinstance(horizon_days, int) or horizon_days <= 0 or discount_rate < 0:
        raise ValueError("Positive integer days and nonnegative discount required")
    df = summary.copy()
    cumulative = np.array([expected_purchases(bgf,day,df)
        for day in range(horizon_days+1)])
    increments = np.diff(cumulative,axis=0)
    if not np.isfinite(increments).all() or (increments < -1e-7).any():
        raise ValueError("Invalid purchase increments")
    df["predicted_purchases"] = cumulative[-1]
    df["prob_alive"] = bgf.conditional_probability_alive(df.frequency,df.recency,df["T"])
    repeat = df.frequency > 0
    spend = pd.Series(fallback_spend,index=df.index,dtype=float)
    spend.loc[repeat] = ggf.conditional_expected_average_profit(df.loc[repeat,"frequency"],df.loc[repeat,"monetary_value"])
    df["predicted_clv"] = (increments / (1 + discount_rate) ** (np.arange(1,horizon_days+1)[:,None]/30)).sum(axis=0) * spend

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
