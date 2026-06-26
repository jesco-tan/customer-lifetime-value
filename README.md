# Thornbury & Co.: Customer Lifetime Value & Cohort Analytics

**Headline result:** Enthusiast-channel customers generate a mean 12-month CLV of **£255**: 5.3× the £49 mean for Deal-Seeker-channel customers acquired via flash sales. A naive RFM Monetary ranking inverts this ordering, because Deal-Seekers score high on historical spend despite having near-zero repeat probability. The BG/NBD model resolves the conflict; the win-back campaign recommendation follows from it.

---

## What this demonstrates

| Technique | Implementation note |
|---|---|
| **Cohort retention heatmap** | Monthly cohort × period matrix; `cohorts.build_retention_matrix` |
| **Guardrail tracking** | Month-1 retention per channel vs 20% floor; `cohorts.month1_retention_summary` |
| **RFM segmentation** | Quantile scoring → 11 named segments via standard (R, F) decision matrix |
| **UMAP + K-Means** | Optional 2-D projection of (R, F, M) space; falls back gracefully without `umap-learn` |
| **BG/NBD model** | `lifetimes.BetaGeoFitter`; fit on 20-month calibration, validated on 91-day holdout |
| **Gamma-Gamma model** | `lifetimes.GammaGammaFitter`; repeat buyers only (frequency > 0) |
| **Discounted 12-month CLV** | BG/NBD × Gamma-Gamma, 1%/month discount rate, horizon 365 days |
| **Holdout validation** | Predicted vs actual invoices in holdout period; MAE = 0.4745 purchases/customer |
| **Channel disaggregation** | CLV, P(alive), retention, and campaign ROI broken out by acquisition channel |
| **Win-back ROI model** | Expected return per contact by channel; £3.50 cost, channel-specific response rates |
| **Latent engagement generator** | Synthetic data with correlated purchase rate / basket / churn via shared latent variable |
| **Self-contained HTML report** | `scripts/generate_html_report.py` → `outputs/report.html` (no external dependencies) |
| **Streamlit dashboard** | Reads `outputs/`; run after notebook |

---

## Notebook sections

| Section | What it covers |
|---|---|
| **1. Load Data** | Synthetic (default) or UCI Online Retail II; channel assignment explained |
| **2. Channel Distributions** | Checks that Enthusiasts buy more often, Deal-Seekers spend more per visit |
| **3. Cohort Retention** | Monthly heatmap; overall Month-1 retention 16.1% (below 20% guardrail) |
| **4. Guardrail Check** | Month-1 retention by channel: Enthusiast 31.0%, Mainstream 13.0%, Deal-Seeker 4.5% |
| **5. RFM Segmentation** | 11 segments; Loyal Customers is the top revenue segment |
| **6. BG/NBD Model** | Calibration/holdout split; model diagnostics; fitted params (r=0.71, a=0.21, b=1.38) |
| **7. Gamma-Gamma Model** | Spend model fit; calibration scatter; avg transaction value by channel |
| **8. CLV Prediction** | Mean £143, median £63.25, P90 £377; top 18% generate 80% of predicted revenue |
| **9. The Tension** | M-score vs CLV scatter by channel, Deal-Seekers cluster in high-M / low-CLV quadrant |
| **10. Holdout Validation** | MAE 0.4745 purchases; bias +0.035 (slight over-prediction); MAE by channel |
| **11. Disaggregated CLV** | Violin plots; Enthusiast £255 vs Deal-Seeker £49; Deal-Seeker median = £0 |
| **12. Campaign ROI** | Enthusiast +10.0×, Mainstream +1.61×, Deal-Seeker −0.49× |
| **13. Decision** | Full evidence table; budget allocation recommendation |
| **14. Export** | `outputs/run_summary.json` with all key metrics |

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt

# Run the full pipeline
jupyter notebook notebooks/clv_walkthrough.ipynb  # Run All

# Generate the self-contained HTML report
python scripts/generate_html_report.py

# Launch the Streamlit dashboard (after notebook)
streamlit run src/clv/app.py

# Tests
pytest tests -q
```

No downloads required, the notebook ships with the Thornbury & Co. synthetic data. For the real UCI Online Retail II dataset, download `online_retail_II.xlsx` from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and place in `data/`.

---

## Output file inventory

| File | Description |
|---|---|
| `outputs/retention_heatmap.png` | Monthly cohort retention rates (all channels combined) |
| `outputs/month1_retention_by_channel.png` | Month-1 retention by channel with 20% guardrail line |
| `outputs/eda_distributions.png` | Transaction volume, invoices per customer, log-revenue distribution |
| `outputs/segment_revenue.png` | Revenue bar chart by RFM segment |
| `outputs/umap_scatter.png` | UMAP projection coloured by RFM segment (if umap-learn installed) |
| `outputs/bgnbd_matrices.png` | BG/NBD expected purchases and P(alive) heatmaps |
| `outputs/ggf_calibration.png` | Gamma-Gamma calibration scatter + avg transaction value by channel |
| `outputs/clv_distribution.png` | 12-month CLV histogram + P(alive) distribution |
| `outputs/rfm_vs_clv_tension.png` | Scatter: M-score vs CLV by channel (the analytical tension) |
| `outputs/holdout_validation.png` | Predicted vs actual purchases in holdout; MAE by channel |
| `outputs/channel_clv_comparison.png` | Violin plots of CLV and P(alive) distribution by channel |
| `outputs/campaign_roi.png` | Win-back ROI per channel and expected return vs cost |
| `outputs/segment_profiles.csv` | Aggregate stats per RFM segment |
| `outputs/clv_predictions.csv` | Per-customer CLV, P(alive), predicted purchases |
| `outputs/run_summary.json` | All key numeric results (single source of truth for docs) |
| `outputs/report.html` | Self-contained stakeholder HTML report |

---

## Tests

```bash
pytest tests -q   # 13 tests, ~3s
```

Covers: synthetic data schema and latent correlation structure, channel distribution, RFM scoring, segment summary, retention matrix, channel disaggregation, campaign ROI. BG/NBD and Gamma-Gamma fitting are exercised in the notebook only (marked as slow).

---

## Tech stack

`lifetimes` · `pandas` · `numpy` · `scikit-learn` · `umap-learn` · `matplotlib` · `seaborn` · `duckdb` · `streamlit` · `pyyaml` · `jinja2`

See [`REPORT.md`](REPORT.md) for the full analyst memo with methodology, findings, and limitations.
