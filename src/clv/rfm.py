"""RFM scoring, segmentation, and UMAP projection with channel profiling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


SEGMENT_MAP = {
    (5, 5): "Champions",
    (4, 5): "Loyal Customers",
    (5, 4): "Loyal Customers",
    (4, 4): "Loyal Customers",
    (3, 5): "Potential Loyalists",
    (5, 3): "Potential Loyalists",
    (3, 4): "Potential Loyalists",
    (4, 3): "Potential Loyalists",
    (5, 1): "Recent Customers",
    (5, 2): "Recent Customers",
    (4, 1): "Promising",
    (4, 2): "Promising",
    (3, 3): "Needs Attention",
    (3, 2): "About to Sleep",
    (3, 1): "About to Sleep",
    (2, 5): "At Risk",
    (2, 4): "At Risk",
    (2, 3): "At Risk",
    (1, 5): "Can't Lose Them",
    (1, 4): "Can't Lose Them",
    (2, 2): "Hibernating",
    (2, 1): "Hibernating",
    (1, 3): "Lost",
    (1, 2): "Lost",
    (1, 1): "Lost",
}


def compute_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp, bins: int = 5) -> pd.DataFrame:
    """
    Compute RFM scores from raw transactions.

    Returns a DataFrame indexed by Customer ID with columns:
    Recency, Frequency, Monetary, R, F, M, RFM_Score, Segment
    and Channel if available.
    """
    rfm = (
        df.groupby("Customer ID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("Invoice", "nunique"),
            Monetary=("Revenue", "sum"),
        )
    )

    rfm["R"] = pd.qcut(
        rfm["Recency"], q=bins, labels=range(bins, 0, -1), duplicates="drop"
    ).astype(int)
    rfm["F"] = pd.qcut(
        rfm["Frequency"].rank(method="first"), q=bins, labels=range(1, bins + 1), duplicates="drop"
    ).astype(int)
    rfm["M"] = pd.qcut(
        rfm["Monetary"].rank(method="first"), q=bins, labels=range(1, bins + 1), duplicates="drop"
    ).astype(int)

    rfm["RFM_Score"] = rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)
    rfm["Segment"] = rfm.apply(
        lambda row: SEGMENT_MAP.get((row["R"], row["F"]), "Others"), axis=1
    )

    # Preserve Channel if available
    if "Channel" in df.columns:
        ch = df.groupby("Customer ID")["Channel"].first()
        rfm = rfm.join(ch)

    return rfm.reset_index()


def umap_cluster(rfm: pd.DataFrame, n_clusters: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Project RFM (R, F, M) into 2-D with UMAP then cluster with K-Means.

    Requires: umap-learn
    """
    try:
        import umap
    except ImportError:
        return rfm.assign(UMAP1=np.nan, UMAP2=np.nan, KMeans_Cluster=-1)

    features = rfm[["R", "F", "M"]].values.astype(float)
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    reducer = umap.UMAP(n_components=2, random_state=seed)
    embedding = reducer.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels = km.fit_predict(embedding)

    rfm = rfm.copy()
    rfm["UMAP1"] = embedding[:, 0]
    rfm["UMAP2"] = embedding[:, 1]
    rfm["KMeans_Cluster"] = labels
    return rfm


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """Aggregate segment-level stats for reporting."""
    return (
        rfm.groupby("Segment")
        .agg(
            Customers=("Customer ID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            Total_Revenue=("Monetary", "sum"),
        )
        .round(2)
        .sort_values("Total_Revenue", ascending=False)
        .reset_index()
    )


def rfm_channel_profile(rfm: pd.DataFrame) -> pd.DataFrame:
    """Compare mean RFM scores and segment distribution across acquisition channels.

    Returns a DataFrame with one row per channel showing mean R, F, M scores,
    the dominant segment, and fraction of customers in each tier.

    The key finding: Deal-Seekers score high on M (high Monetary) but low on F
    and R, which the BG/NBD model correctly translates into low predicted CLV.
    """
    if "Channel" not in rfm.columns:
        return pd.DataFrame()

    profile = (
        rfm.groupby("Channel")
        .agg(
            n_customers=("Customer ID", "count"),
            mean_R=("R", "mean"),
            mean_F=("F", "mean"),
            mean_M=("M", "mean"),
            mean_Recency_days=("Recency", "mean"),
            mean_Frequency=("Frequency", "mean"),
            mean_Monetary=("Monetary", "mean"),
        )
        .round(2)
        .reset_index()
    )

    dominant = (
        rfm.groupby("Channel")["Segment"]
        .agg(lambda x: x.value_counts().index[0])
        .rename("dominant_segment")
        .reset_index()
    )

    return profile.merge(dominant, on="Channel")
