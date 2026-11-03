# Customer analytics on a synthetic retailer: cohorts, RFM and probabilistic value

Implemented RFM, cohort summaries and BG/NBD plus Gamma-Gamma models for a fictional retailer with 2,500 generated customers. A separate 91-day purchase-count holdout is recorded. CLV and campaign outputs depend on explicit modelling policies and assumed commercial inputs; no real budget or campaign return is evidenced.

Thornbury & Co. is a fictional retailer scenario. I built a customer-analytics workflow using 2,500 generated customers, 9,632 invoices and 29,336 invoice line items. It includes RFM segmentation, cohort summaries, BG/NBD purchase predictions, a Gamma-Gamma spend model and a separate purchase-count holdout.

The data are synthetic, not a real retailer's transaction log. The GBP500,000 budget, contact cost and response rates are assumed scenario inputs. No campaign budget was managed and no campaign returns were observed.

## Read the results

[REPORT.md](REPORT.md) gives the supported results, assumptions and limitations. [report.html](outputs/report.html) contains the same interpretation with embedded figures. The [walkthrough notebook](notebooks/clv_walkthrough.ipynb) retains numerical evidence and corrected narrative. Modified code cells have cleared execution state until rerun. Old wording and reports are available through Git history.

No real users, customers, bank-disclosure measurements or policy effects are represented by this saved run.

## Reproduce or inspect

Render the current report without dependencies or analytical execution:

```powershell
python scripts/generate_html_report.py
```

Run the synthetic notebook in an isolated environment (this overwrites analytical outputs):

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
# Install notebook tooling inside the same virtual environment if needed.
.venv/Scripts/python.exe -m pip install jupyter
.venv/Scripts/python.exe -m jupyter notebook notebooks/clv_walkthrough.ipynb
.venv/Scripts/python.exe -m pytest tests -q
```

Install analytical dependencies only in a project virtual environment.

## Evidence and implementation limits

Customer analytics on a fictional synthetic retailer, with cohorts, RFM, BG/NBD and Gamma-Gamma models, a purchase-count holdout and explicit campaign-scenario limitations.

See [REPORT.md](REPORT.md) for method definitions, saved-run provenance and remaining analytical limitations.
