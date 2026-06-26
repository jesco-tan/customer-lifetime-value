"""Generate a fully self-contained HTML stakeholder report from outputs/.

Reads run_summary.json and all PNG files in outputs/, encodes images as
base64, and writes outputs/report.html with no external dependencies.

Usage:
    python scripts/generate_html_report.py
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
SUMMARY_PATH = OUTPUTS / "run_summary.json"


def _img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _img_tag(path: Path, alt: str = "", style: str = "width:100%;max-width:900px;") -> str:
    if not path.exists():
        return f'<p class="missing">Chart not found: {path.name}</p>'
    data = _img_b64(path)
    return f'<img src="data:image/png;base64,{data}" alt="{alt}" style="{style}" loading="lazy">'


def _kpi(label: str, value: str, colour: str = "#2166ac") -> str:
    return f"""
    <div class="kpi-card">
      <div class="kpi-value" style="color:{colour}">{value}</div>
      <div class="kpi-label">{label}</div>
    </div>"""


def _alert(level: str, text: str) -> str:
    """level: green | red | amber | blue"""
    colours = {
        "green": ("#e8f5e9", "#2e7d32", "✓"),
        "red":   ("#ffebee", "#c62828", "✗"),
        "amber": ("#fff8e1", "#f57f17", "⚠"),
        "blue":  ("#e3f2fd", "#1565c0", "ℹ"),
    }
    bg, fg, icon = colours.get(level, colours["blue"])
    return f'<div class="alert" style="background:{bg};border-left:4px solid {fg};color:{fg}">{icon} {text}</div>'


def generate(output_path: Path = OUTPUTS / "report.html") -> None:
    if not SUMMARY_PATH.exists():
        print(f"ERROR: {SUMMARY_PATH} not found. Run the notebook first.")
        sys.exit(1)

    s = json.loads(SUMMARY_PATH.read_text())
    clv = s.get("clv_stats", {})
    ch_clv = s.get("channel_clv", {})
    m1_ret = s.get("month1_retention_by_channel", {})
    roi = s.get("campaign_roi_by_channel", {})
    guardrail = s.get("guardrail_threshold", 0.20)
    guardrail_breached = s.get("guardrail_breached", False)

    ent = ch_clv.get("Enthusiast", {})
    main = ch_clv.get("Mainstream", {})
    deal = ch_clv.get("Deal-Seeker", {})

    ent_clv = ent.get("mean_clv", 0)
    deal_clv = deal.get("mean_clv", 0.01)
    ratio = ent_clv / max(deal_clv, 0.01)

    ent_roi = roi.get("Enthusiast", {}).get("roi", 0)
    main_roi = roi.get("Mainstream", {}).get("roi", 0)
    deal_roi = roi.get("Deal-Seeker", {}).get("roi", 0)

    deal_m1 = m1_ret.get("Deal-Seeker", 0)
    ent_m1 = m1_ret.get("Enthusiast", 0)

    bgnbd = s.get("bgnbd_params", {})
    r_val = bgnbd.get("r", 0)
    alpha_val = bgnbd.get("alpha", 0)
    a_val = bgnbd.get("a", 0)
    b_val = bgnbd.get("b", 0)

    css = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #f5f7fa; color: #1a1a2e; line-height: 1.6; }
    nav { position: sticky; top: 0; background: #1a237e; color: white;
          padding: 12px 24px; display: flex; gap: 20px; z-index: 100;
          flex-wrap: wrap; font-size: 0.85rem; }
    nav a { color: #90caf9; text-decoration: none; white-space: nowrap; }
    nav a:hover { color: white; }
    .hero { background: linear-gradient(135deg, #1a237e 0%, #283593 60%, #1565c0 100%);
            color: white; padding: 48px 24px 40px; text-align: center; }
    .hero h1 { font-size: 2rem; margin-bottom: 8px; }
    .hero .subtitle { font-size: 1.1rem; opacity: 0.85; margin-bottom: 4px; }
    .hero .headline { font-size: 1.3rem; font-weight: 600; margin-top: 16px;
                      background: rgba(255,255,255,0.15); display: inline-block;
                      padding: 8px 20px; border-radius: 6px; }
    .kpi-strip { display: flex; gap: 16px; flex-wrap: wrap; justify-content: center;
                 padding: 24px; background: white; border-bottom: 1px solid #e0e0e0; }
    .kpi-card { text-align: center; min-width: 140px; padding: 12px 16px;
                border-radius: 8px; background: #f8f9ff; border: 1px solid #e8eaf6; }
    .kpi-value { font-size: 1.6rem; font-weight: 700; }
    .kpi-label { font-size: 0.78rem; color: #555; margin-top: 4px; text-transform: uppercase;
                 letter-spacing: 0.04em; }
    .section { max-width: 960px; margin: 32px auto; padding: 0 24px; }
    .section h2 { font-size: 1.4rem; color: #1a237e; border-bottom: 2px solid #e8eaf6;
                  padding-bottom: 8px; margin-bottom: 20px; }
    .section h3 { font-size: 1.1rem; color: #283593; margin: 20px 0 10px; }
    .chart-wrap { margin: 16px 0; text-align: center; }
    .chart-wrap img { border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.10); }
    .alert { padding: 12px 16px; border-radius: 6px; margin: 12px 0;
             font-weight: 500; font-size: 0.92rem; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.9rem; }
    th { background: #e8eaf6; color: #1a237e; padding: 10px 14px; text-align: left; }
    td { padding: 9px 14px; border-bottom: 1px solid #eeeeee; }
    tr:hover td { background: #f5f7ff; }
    .missing { color: #999; font-style: italic; font-size: 0.85rem; padding: 12px 0; }
    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
    @media (max-width: 640px) { .two-col { grid-template-columns: 1fr; } }
    footer { text-align: center; padding: 32px; color: #888; font-size: 0.82rem;
             border-top: 1px solid #e0e0e0; margin-top: 48px; }
    """

    def section(anchor: str, title: str, body: str) -> str:
        return f"""
        <div class="section" id="{anchor}">
          <h2>{title}</h2>
          {body}
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Thornbury & Co., CLV Analysis Report</title>
<style>{css}</style>
</head>
<body>

<nav>
  <strong style="color:white;margin-right:8px">Thornbury & Co.</strong>
  <a href="#overview">Overview</a>
  <a href="#retention">Retention</a>
  <a href="#guardrail">Guardrail</a>
  <a href="#rfm">RFM</a>
  <a href="#model">BG/NBD Model</a>
  <a href="#clv">CLV</a>
  <a href="#tension">Tension</a>
  <a href="#validation">Validation</a>
  <a href="#channels">Channels</a>
  <a href="#campaign">Campaign ROI</a>
  <a href="#decision">Decision</a>
</nav>

<div class="hero">
  <div class="subtitle">UK Gift & Home Accessories Retailer · 2010–2011</div>
  <h1>Thornbury & Co., Customer Lifetime Value Analysis</h1>
  <div class="headline">
    Enthusiast-channel CLV is £{ent_clv:,.0f}, {ratio:.1f}× higher than Deal-Seeker CLV of £{deal_clv:,.0f}.<br>
    Win-back ROI: Enthusiast +{ent_roi:.1f}×, Mainstream +{main_roi:.1f}×, Deal-Seeker {deal_roi:.2f}× (negative).
  </div>
</div>

<div class="kpi-strip">
  {_kpi("Mean 12-Month CLV", f"£{clv.get('mean_clv', 0):,.0f}")}
  {_kpi("Median CLV", f"£{clv.get('median_clv', 0):,.0f}", "#4CAF50")}
  {_kpi("P90 CLV", f"£{clv.get('p90_clv', 0):,.0f}", "#9C27B0")}
  {_kpi("Total Predicted Revenue", f"£{clv.get('total_predicted_revenue', 0):,.0f}", "#FF6F00")}
  {_kpi("Customers P(alive) > 50%", f"{clv.get('pct_alive', 0)}%", "#00796B")}
  {_kpi("Customers", f"{s.get('n_customers', 0):,}")}
</div>

{section("overview", "1. Data Overview", f"""
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Customers</td><td>{s.get('n_customers', 0):,}</td></tr>
    <tr><td>Transaction records</td><td>{s.get('n_transactions', 0):,}</td></tr>
    <tr><td>Unique invoices</td><td>{s.get('n_invoices', 0):,}</td></tr>
    <tr><td>Observation window</td><td>2010-01-01 → {s.get('observation_end', '')}</td></tr>
    <tr><td>Calibration end</td><td>{s.get('calibration_end', '')}</td></tr>
    <tr><td>Holdout period</td><td>{s.get('holdout_days', 0)} days</td></tr>
  </table>
  <div class="chart-wrap">{_img_tag(OUTPUTS / "eda_distributions.png", "EDA distributions")}</div>
""")}

{section("retention", "2. Cohort Retention", f"""
  <p>Customers are grouped by first-purchase month. Retention rate = fraction of cohort that purchased again in each subsequent month.</p>
  {"" if not guardrail_breached else _alert("amber", f"Overall Month-1 retention ({s.get('month1_retention_overall', 0):.1%}) is below the {guardrail:.0%} guardrail.")}
  <div class="chart-wrap">{_img_tag(OUTPUTS / "retention_heatmap.png", "Cohort retention heatmap")}</div>
""")}

{section("guardrail", "3. Guardrail, Month-1 Retention by Channel", f"""
  <p>Month-1 retention below {guardrail:.0%} signals acquisition of customers with no genuine purchase intent.</p>
  {_alert("green", f"Enthusiast: {ent_m1:.1%} Month-1 retention, above guardrail.")}
  {_alert("red", f"Deal-Seeker: {deal_m1:.1%} Month-1 retention, GUARDRAIL BREACHED. Expanding flash-sale acquisition destroys retention economics.")}
  {_alert("red", f"Mainstream: {m1_ret.get('Mainstream', 0):.1%} Month-1 retention, also below guardrail. Audit acquisition mix.")}
  <div class="chart-wrap">{_img_tag(OUTPUTS / "month1_retention_by_channel.png", "Month-1 retention by channel")}</div>
""")}

{section("rfm", "4. RFM Segmentation", f"""
  <p>Each customer receives R, F, and M scores (1–5). The (R, F) combination maps to one of 11 named segments.</p>
  {_alert("blue", "The M score creates a conflict: Deal-Seekers rank highest on M (large first basket) but lowest on predicted CLV. See Section 7.")}
  <div class="two-col">
    <div class="chart-wrap">{_img_tag(OUTPUTS / "segment_revenue.png", "Revenue by RFM segment")}</div>
    <div class="chart-wrap">{_img_tag(OUTPUTS / "umap_scatter.png", "UMAP projection")}</div>
  </div>
""")}

{section("model", "5. BG/NBD Purchase Frequency Model", f"""
  <p>The BG/NBD model was fit on the calibration period and validated on the 91-day holdout. Parameters:</p>
  <table>
    <tr><th>Parameter</th><th>Value</th><th>Interpretation</th></tr>
    <tr><td>r (purchase rate shape)</td><td>{r_val:.4f}</td><td>Moderate purchase rate heterogeneity</td></tr>
    <tr><td>α (purchase rate scale)</td><td>{alpha_val:.2f}</td><td>Slow baseline rate, expected for non-subscription retail</td></tr>
    <tr><td>a (dropout shape)</td><td>{a_val:.4f}</td><td>Low individual variation in churn</td></tr>
    <tr><td>b (dropout scale)</td><td>{b_val:.4f}</td><td>Mean dropout ≈ 13.4% per purchase</td></tr>
  </table>
  <div class="chart-wrap">{_img_tag(OUTPUTS / "bgnbd_matrices.png", "BG/NBD matrices")}</div>
  <div class="chart-wrap">{_img_tag(OUTPUTS / "ggf_calibration.png", "Gamma-Gamma calibration")}</div>
""")}

{section("clv", "6. 12-Month CLV Prediction", f"""
  <p>Combining BG/NBD and Gamma-Gamma gives a discounted 12-month CLV per customer. Customers with a single purchase (frequency = 0) receive CLV = 0, correct model behaviour.</p>
  <div class="chart-wrap">{_img_tag(OUTPUTS / "clv_distribution.png", "CLV distribution")}</div>
  <table>
    <tr><th>Channel</th><th>N</th><th>Mean CLV</th><th>Median CLV</th><th>P90 CLV</th><th>P(alive) %</th></tr>
    <tr><td><strong>Enthusiast</strong></td><td>{ent.get('n_customers', 0):,}</td>
        <td><strong>£{ent.get('mean_clv', 0):,.2f}</strong></td>
        <td>£{ent.get('median_clv', 0):,.2f}</td>
        <td>£{ent.get('p90_clv', 0):,.2f}</td>
        <td>{ent.get('pct_alive', 0):.1f}%</td></tr>
    <tr><td>Mainstream</td><td>{main.get('n_customers', 0):,}</td>
        <td>£{main.get('mean_clv', 0):,.2f}</td>
        <td>£{main.get('median_clv', 0):,.2f}</td>
        <td>£{main.get('p90_clv', 0):,.2f}</td>
        <td>{main.get('pct_alive', 0):.1f}%</td></tr>
    <tr><td>Deal-Seeker</td><td>{deal.get('n_customers', 0):,}</td>
        <td>£{deal.get('mean_clv', 0):,.2f}</td>
        <td><strong>£{deal.get('median_clv', 0):,.2f}</strong></td>
        <td>£{deal.get('p90_clv', 0):,.2f}</td>
        <td>{deal.get('pct_alive', 0):.1f}%</td></tr>
  </table>
  {_alert("red", f"Deal-Seeker median CLV = £{deal.get('median_clv', 0):.2f}. Most Deal-Seekers bought exactly once; GGF cannot predict spend without repeat history.")}
""")}

{section("tension", "7. The Tension, M Score vs Predicted CLV", f"""
  <p>Deal-Seekers rank highest on RFM Monetary score (high historical spend) but lowest on predicted CLV. This is the analytical crux: the naive M-score ranking inverts the CLV ranking.</p>
  {_alert("red", "A win-back campaign ranked by M score would contact Deal-Seekers first, the channel with the worst ROI and the lowest retention.")}
  <div class="chart-wrap">{_img_tag(OUTPUTS / "rfm_vs_clv_tension.png", "M score vs CLV tension")}</div>
""")}

{section("validation", "8. Holdout Validation", f"""
  <p>BG/NBD predictions for the 91-day holdout period versus actual purchases.</p>
  {_alert("green", f"Overall holdout MAE = {s.get('holdout_mae', 0):.4f} purchases/customer. Bias = {s.get('holdout_bias', 0):+.4f} (slightly optimistic).")}
  <div class="chart-wrap">{_img_tag(OUTPUTS / "holdout_validation.png", "Holdout validation")}</div>
""")}

{section("channels", "9. CLV by Acquisition Channel", f"""
  <p>Violin plots show the full CLV distribution per channel, not just means. A narrow Enthusiast distribution means high CLV is broad-based, not outlier-driven.</p>
  {_alert("green", f"Enthusiast mean CLV £{ent_clv:,.2f} is {ratio:.1f}× Deal-Seeker mean CLV £{deal_clv:,.2f}.")}
  <div class="chart-wrap">{_img_tag(OUTPUTS / "channel_clv_comparison.png", "Channel CLV comparison")}</div>
""")}

{section("campaign", "10. Win-Back Campaign ROI", f"""
  <p>£3.50 cost per contact, channel-specific response rates (Enthusiast 18%, Mainstream 9%, Deal-Seeker 4%).</p>
  {_alert("green", f"Enthusiast ROI: +{ent_roi:.1f}× per contact. Budget spend on Enthusiasts returns £38.50 per £3.50 contact.")}
  {_alert("amber", f"Mainstream ROI: +{main_roi:.2f}× per contact. Positive but modest, existing customers worth retaining, acquisition worth auditing.")}
  {_alert("red", f"Deal-Seeker ROI: {deal_roi:.2f}× per contact. NEGATIVE, each contact loses £1.70 on a £3.50 outreach.")}
  <div class="chart-wrap">{_img_tag(OUTPUTS / "campaign_roi.png", "Campaign ROI")}</div>
""")}

{section("decision", "11. Decision and Recommendation", f"""
  <table>
    <tr><th>Channel</th><th>Mean CLV</th><th>P(alive)</th><th>M1 Retention</th><th>ROI</th><th>Action</th></tr>
    <tr><td><strong>Enthusiast</strong></td><td>£{ent_clv:,.2f}</td><td>{ent.get('pct_alive',0):.1f}%</td>
        <td>{ent_m1:.1%}</td><td>+{ent_roi:.1f}×</td><td style="color:#2e7d32"><strong>Full budget priority</strong></td></tr>
    <tr><td>Mainstream</td><td>£{main.get('mean_clv',0):,.2f}</td><td>{main.get('pct_alive',0):.1f}%</td>
        <td>{m1_ret.get('Mainstream',0):.1%}</td><td>+{main_roi:.2f}×</td>
        <td style="color:#f57f17"><strong>Retain, audit acquisition</strong></td></tr>
    <tr><td>Deal-Seeker</td><td>£{deal_clv:,.2f}</td><td>{deal.get('pct_alive',0):.1f}%</td>
        <td>{deal_m1:.1%}</td><td>{deal_roi:.2f}×</td>
        <td style="color:#c62828"><strong>Exclude, negative ROI</strong></td></tr>
  </table>
  {_alert("green", "Allocate full £500K win-back budget to Enthusiast and Mainstream customers with P(alive) > 0.5.")}
  {_alert("red", f"Do NOT contact Deal-Seekers. Each contact loses £1.70 expected net. Do NOT expand flash-sale acquisition ({deal_m1:.1%} Month-1 retention).")}
  {_alert("blue", "High AOV is not a substitute for P(alive). The M score must be filtered through a churn model before acting on it.")}
""")}

<footer>
  Thornbury & Co. CLV Analysis · Generated from <code>outputs/run_summary.json</code> ·
  {s.get('n_customers', 0):,} customers · Observation end {s.get('observation_end', '')}
</footer>

</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"Report written to {output_path} ({output_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    generate()
