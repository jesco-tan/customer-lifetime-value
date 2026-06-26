"""Streamlit dashboard for CLV analysis.

Run with: streamlit run src/clv/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"

st.set_page_config(page_title="Customer Lifetime Value Dashboard", layout="wide")
st.title("Customer Lifetime Value & Cohort Analytics")

# ── Load outputs ──────────────────────────────────────────────────────────────

@st.cache_data
def load_summary() -> dict | None:
    p = OUTPUTS / "run_summary.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def load_segment_profiles() -> pd.DataFrame | None:
    p = OUTPUTS / "segment_profiles.csv"
    return pd.read_csv(p) if p.exists() else None


summary = load_summary()
seg_df = load_segment_profiles()

if summary is None:
    st.warning("No outputs found. Run the walkthrough notebook first to generate results.")
    st.stop()

# ── KPI row ───────────────────────────────────────────────────────────────────
clv_stats = summary.get("clv_stats", {})
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean 12-Month CLV", f"£{clv_stats.get('mean_clv', 0):,.2f}")
col2.metric("Median 12-Month CLV", f"£{clv_stats.get('median_clv', 0):,.2f}")
col3.metric("90th Pct CLV", f"£{clv_stats.get('p90_clv', 0):,.2f}")
col4.metric("% Customers Likely Alive", f"{clv_stats.get('pct_alive', 0)}%")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Cohort Retention", "RFM Segments", "CLV Distribution"])

with tab1:
    img = OUTPUTS / "retention_heatmap.png"
    if img.exists():
        st.image(str(img), use_column_width=True)
    else:
        st.info("Retention heatmap not yet generated.")

with tab2:
    if seg_df is not None:
        selected = st.multiselect("Filter segments", seg_df["Segment"].tolist(), default=seg_df["Segment"].tolist())
        filtered = seg_df[seg_df["Segment"].isin(selected)]
        st.dataframe(filtered.set_index("Segment"), use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(filtered["Segment"], filtered["Total_Revenue"], color=sns.color_palette("tab10", len(filtered)))
        ax.set_xlabel("Total Revenue (£)")
        st.pyplot(fig)
    else:
        st.info("Segment profiles not yet generated.")

with tab3:
    img = OUTPUTS / "clv_distribution.png"
    if img.exists():
        st.image(str(img), use_column_width=True)
    else:
        st.info("CLV distribution chart not yet generated.")
