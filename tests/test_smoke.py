"""Smoke tests, fast, no full pipeline needed."""

import pandas as pd
import numpy as np
import pytest
from src.clv.etl import load_synthetic, split_calibration_holdout, get_customer_channels
from src.clv.rfm import compute_rfm, segment_summary, rfm_channel_profile
from src.clv.cohorts import build_retention_matrix, retention_by_channel, month1_retention_summary
from src.clv.bgf import clv_by_channel, campaign_roi_by_channel


@pytest.fixture
def transactions():
    return load_synthetic(n_customers=300, seed=0)


# ── ETL ───────────────────────────────────────────────────────────────────────

def test_synthetic_load(transactions):
    assert len(transactions) > 0
    assert {"Customer ID", "InvoiceDate", "Revenue", "Channel"}.issubset(transactions.columns)


def test_synthetic_channel_distribution(transactions):
    channels = transactions["Customer ID"].map(
        transactions.groupby("Customer ID")["Channel"].first()
    ).unique()
    assert set(channels).issubset({"Enthusiast", "Mainstream", "Deal-Seeker"})


def test_synthetic_latent_correlation(transactions):
    """Enthusiasts should have higher invoice frequency than Deal-Seekers."""
    freq = (
        transactions.groupby(["Customer ID", "Channel"])["Invoice"]
        .nunique()
        .reset_index(name="invoices")
    )
    mean_freq = freq.groupby("Channel")["invoices"].mean()
    assert mean_freq.get("Enthusiast", 0) > mean_freq.get("Deal-Seeker", 0), (
        "Enthusiasts should buy more often than Deal-Seekers"
    )


def test_get_customer_channels(transactions):
    ch = get_customer_channels(transactions)
    assert ch.index.name == "Customer ID"
    assert set(ch.unique()).issubset({"Enthusiast", "Mainstream", "Deal-Seeker"})


def test_calibration_split(transactions):
    cutoff = pd.Timestamp("2011-03-01")
    cal, hold = split_calibration_holdout(transactions, cutoff)
    assert (cal["InvoiceDate"] <= cutoff).all()
    assert (hold["InvoiceDate"] > cutoff).all()


# ── RFM ──────────────────────────────────────────────────────────────────────

def test_rfm_smoke(transactions):
    snapshot = transactions["InvoiceDate"].max()
    rfm = compute_rfm(transactions, snapshot_date=snapshot, bins=4)
    assert "Segment" in rfm.columns
    assert "Channel" in rfm.columns
    assert rfm["R"].between(1, 4).all()
    assert rfm["F"].between(1, 4).all()


def test_segment_summary(transactions):
    snapshot = transactions["InvoiceDate"].max()
    rfm = compute_rfm(transactions, snapshot_date=snapshot, bins=4)
    summary = segment_summary(rfm)
    assert "Customers" in summary.columns
    assert summary["Customers"].sum() == rfm.shape[0]


def test_rfm_channel_profile(transactions):
    snapshot = transactions["InvoiceDate"].max()
    rfm = compute_rfm(transactions, snapshot_date=snapshot, bins=4)
    profile = rfm_channel_profile(rfm)
    assert "Channel" in profile.columns
    assert "mean_M" in profile.columns
    assert "dominant_segment" in profile.columns
    assert len(profile) == 3  # three channels


# ── Cohorts ───────────────────────────────────────────────────────────────────

def test_retention_matrix(transactions):
    retention = build_retention_matrix(transactions, freq="M")
    assert retention.shape[0] > 0
    assert retention.iloc[:, 0].dropna().between(0, 1).all()


def test_retention_by_channel(transactions):
    by_ch = retention_by_channel(transactions, freq="M")
    assert set(by_ch.keys()) == {"Enthusiast", "Mainstream", "Deal-Seeker"}
    for ret in by_ch.values():
        assert ret.iloc[:, 0].dropna().between(0, 1).all()


def test_month1_retention_summary(transactions):
    summary = month1_retention_summary(transactions)
    assert "Channel" in summary.columns
    assert "mean_month1_retention" in summary.columns
    # Should include all three channels + Overall
    assert "Overall" in summary["Channel"].values
    # Enthusiasts should retain better than Deal-Seekers
    ent = summary.loc[summary["Channel"] == "Enthusiast", "mean_month1_retention"].values
    ds = summary.loc[summary["Channel"] == "Deal-Seeker", "mean_month1_retention"].values
    if len(ent) and len(ds) and not (np.isnan(ent[0]) or np.isnan(ds[0])):
        assert ent[0] > ds[0], "Enthusiasts should have higher month-1 retention than Deal-Seekers"


# ── CLV disaggregation (no model fitting) ────────────────────────────────────

def _make_mock_clv_df(transactions):
    """Build a minimal clv_df-like DataFrame for testing disaggregation functions."""
    ch = get_customer_channels(transactions)
    customers = transactions["Customer ID"].unique()
    rng = np.random.default_rng(99)
    df = pd.DataFrame({
        "predicted_clv": rng.uniform(0, 500, len(customers)),
        "prob_alive": rng.uniform(0, 1, len(customers)),
        "predicted_purchases": rng.uniform(0, 5, len(customers)),
        "monetary_value": rng.uniform(10, 200, len(customers)),
    }, index=pd.Index(customers, name="Customer ID"))
    return df, ch


def test_clv_by_channel(transactions):
    clv_df, ch = _make_mock_clv_df(transactions)
    result = clv_by_channel(clv_df, ch)
    assert "Channel" in result.columns
    assert "mean_clv" in result.columns
    assert set(result["Channel"]).issubset({"Enthusiast", "Mainstream", "Deal-Seeker"})


def test_campaign_roi_by_channel(transactions):
    clv_df, ch = _make_mock_clv_df(transactions)
    roi = campaign_roi_by_channel(
        clv_df, ch,
        cost_per_contact=3.50,
        response_rates={"Enthusiast": 0.18, "Mainstream": 0.09, "Deal-Seeker": 0.04},
    )
    assert "roi" in roi.columns
    assert "net_per_contact" in roi.columns
    assert len(roi) == 3
