"""Load, clean, and prepare transaction data for CLV analysis.

Synthetic generator uses a latent engagement_score per customer that simultaneously
drives purchase rate (positive), basket size (negative, low-engagement Deal-Seekers
spend more per visit), and churn probability (negative, low engagement = higher
dropout risk). This shared latent structure is what makes channel disaggregation
mechanistically meaningful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# Per-channel parameters for the latent engagement structure.
# engagement_score ~ Gamma(eng_shape, eng_scale)
# monthly_rate     = rate_mult * engagement_score
# avg_basket       = basket_base * (ref_engagement / engagement_score) ** basket_power
# churn_prob       = min(0.99, churn_base * (ref_engagement / engagement_score) ** 0.50)
_CHANNEL_PARAMS: dict[str, dict] = {
    "Enthusiast": {
        "eng_shape": 2.5, "eng_scale": 0.80,   # mean engagement ≈ 2.0
        "rate_mult": 0.22,                       # purchases/month per unit engagement
        "basket_base": 28.0,                     # median basket at mean engagement
        "basket_power": 0.40,                    # basket ∝ (ref/eng)^power
        "ref_engagement": 2.0,
        "churn_base": 0.05,                      # mean churn prob per purchase at ref engagement
        "country_probs": [0.85, 0.08, 0.05, 0.02],
    },
    "Mainstream": {
        "eng_shape": 1.5, "eng_scale": 0.60,   # mean engagement ≈ 0.9
        "rate_mult": 0.18,
        "basket_base": 36.0,
        "basket_power": 0.35,
        "ref_engagement": 0.9,
        "churn_base": 0.17,
        "country_probs": [0.78, 0.12, 0.07, 0.03],
    },
    "Deal-Seeker": {
        "eng_shape": 0.8, "eng_scale": 0.40,   # mean engagement ≈ 0.32
        "rate_mult": 0.10,                       # rarely repeat even at max engagement
        "basket_base": 72.0,                     # high basket, they buy on deals
        "basket_power": 0.50,                    # stronger inverse: very low eng → very high basket
        "ref_engagement": 0.32,
        "churn_base": 0.62,                      # 62% chance of dropping after each purchase
        "country_probs": [0.70, 0.15, 0.10, 0.05],
    },
}

_COUNTRIES = ["United Kingdom", "Germany", "France", "Netherlands"]


def load_uci_online_retail(path: str | Path) -> pd.DataFrame:
    """Read the UCI Online Retail II XLSX and return a clean transactions DataFrame."""
    path = Path(path)
    sheets = []
    for sheet in ["Year 2009-2010", "Year 2010-2011"]:
        try:
            sheets.append(pd.read_excel(path, sheet_name=sheet))
        except Exception:
            pass
    if not sheets:
        raise FileNotFoundError(f"Could not read any sheets from {path}")
    raw = pd.concat(sheets, ignore_index=True)
    df = _clean(raw)
    df["Channel"] = "Unknown"
    return df


def load_synthetic(n_customers: int = 2500, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic transaction log for Thornbury & Co.

    The generator plants a latent engagement_score per customer that correlates
    purchase rate, basket size, and churn probability through a shared mechanism
    rather than independent draws. This ensures channel disaggregation produces
    structurally meaningful differences that the BG/NBD model can detect.
    """
    rng = np.random.default_rng(seed)

    OBS_START = pd.Timestamp("2010-01-01")
    OBS_END = pd.Timestamp("2011-12-09")

    channel_names = list(_CHANNEL_PARAMS.keys())
    channels = rng.choice(channel_names, size=n_customers, p=[0.30, 0.45, 0.25])

    records: list[dict] = []
    invoice_counter = 0

    for cid_idx, channel in enumerate(channels):
        cid = f"C{cid_idx + 1:05d}"
        p = _CHANNEL_PARAMS[channel]

        # ── Latent engagement score ───────────────────────────────────────────
        engagement = float(rng.gamma(shape=p["eng_shape"], scale=p["eng_scale"]))
        engagement = max(0.05, engagement)

        # ── Purchase rate (purchases/month) ───────────────────────────────────
        monthly_rate = p["rate_mult"] * engagement
        monthly_rate = max(0.005, monthly_rate)

        # ── Churn probability per purchase ────────────────────────────────────
        # Higher engagement → lower churn (sqrt scaling keeps it bounded)
        churn_prob = min(0.99, p["churn_base"] * (p["ref_engagement"] / engagement) ** 0.50)

        # ── Average basket (inversely correlated with engagement) ─────────────
        basket_at_ref = p["basket_base"]
        basket_scale = (p["ref_engagement"] / engagement) ** p["basket_power"]
        avg_basket = float(rng.lognormal(
            mean=np.log(max(3.0, basket_at_ref * basket_scale)),
            sigma=0.55,
        ))
        avg_basket = max(3.0, avg_basket)

        # ── Country (stable for each customer) ───────────────────────────────
        country = str(rng.choice(_COUNTRIES, p=p["country_probs"]))

        # ── Acquisition date (uniform over first 12 months) ───────────────────
        acq_days = int(rng.integers(0, 365))
        acq_date = OBS_START + pd.Timedelta(days=acq_days)
        if acq_date >= OBS_END:
            acq_date = OBS_END - pd.Timedelta(days=1)

        # ── Generate purchase sequence ────────────────────────────────────────
        purchase_dates = [acq_date]
        current_date = acq_date
        alive = True

        while alive and current_date < OBS_END:
            inter_days = max(1, int(rng.exponential(scale=30.0 / monthly_rate)))
            next_date = current_date + pd.Timedelta(days=inter_days)
            if next_date > OBS_END:
                break
            purchase_dates.append(next_date)
            current_date = next_date
            if rng.random() < churn_prob:
                alive = False

        # ── Generate invoice line-items for each purchase ─────────────────────
        for purchase_date in purchase_dates:
            invoice_counter += 1
            invoice_id = f"INV{invoice_counter:07d}"
            n_items = max(1, int(rng.poisson(3.0)))
            for _ in range(n_items):
                qty = max(1, int(rng.poisson(2)))
                unit_price = float(rng.lognormal(
                    mean=np.log(max(1.0, avg_basket / 3.0)),
                    sigma=0.55,
                ))
                records.append({
                    "Customer ID": cid,
                    "Channel": channel,
                    "Invoice": invoice_id,
                    "InvoiceDate": purchase_date,
                    "Quantity": qty,
                    "Price": round(unit_price, 2),
                    "Country": country,
                })

    df = pd.DataFrame(records)
    df["Revenue"] = df["Quantity"] * df["Price"]
    cols = ["Customer ID", "Channel", "Invoice", "InvoiceDate",
            "Quantity", "Price", "Revenue", "Country"]
    return df[cols].reset_index(drop=True)


def get_customer_channels(df: pd.DataFrame) -> pd.Series:
    """Return a Series mapping Customer ID → Channel."""
    if "Channel" not in df.columns:
        return pd.Series(dtype=str)
    return df.groupby("Customer ID")["Channel"].first()


def _clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    rename = {
        "Customer ID": "Customer ID",
        "CustomerID": "Customer ID",
        "InvoiceDate": "InvoiceDate",
        "Invoice": "Invoice",
        "Quantity": "Quantity",
        "Price": "Price",
        "UnitPrice": "Price",
        "Country": "Country",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df = df.dropna(subset=["Customer ID"])
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]
    df["Customer ID"] = df["Customer ID"].astype(str).str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["Price"]
    return df[["Customer ID", "Invoice", "InvoiceDate",
               "Quantity", "Price", "Revenue", "Country"]].reset_index(drop=True)


def split_calibration_holdout(
    df: pd.DataFrame,
    calibration_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split transactions into calibration and holdout periods."""
    cal = df[df["InvoiceDate"] <= calibration_end].copy()
    hold = df[df["InvoiceDate"] > calibration_end].copy()
    return cal, hold


def to_summary(df: pd.DataFrame, observation_period_end: pd.Timestamp) -> pd.DataFrame:
    """Aggregate transactions to BG/NBD summary (frequency, recency, T, monetary_value)."""
    from lifetimes.utils import summary_data_from_transaction_data
    return summary_data_from_transaction_data(
        df,
        customer_id_col="Customer ID",
        datetime_col="InvoiceDate",
        monetary_value_col="Revenue",
        observation_period_end=observation_period_end,
        freq="D",
    )
