# Validation

I checked observed-zero retention separately from unobserved months and tested the exact daily forecast endpoint and discount convention. All 18 Python tests passed, including an independent numerical-integration check of the purchase expectation.

I evaluated purchase counts and realised revenue at three chronological forecast origins. The saved holdouts compare zero, population and shrinkage spend policies for single-purchase customers against a historical-rate/spend baseline. Individual predictions reconcile with the reported error metrics.

## Scope

The transactions are synthetic. The 90-day monetary holdouts do not validate a full annual forecast or identify incremental campaign returns. Fallback policies have different error and bias tradeoffs. See [REPORT.md](REPORT.md), `outputs/forecast_validation.csv` and `outputs/forecast_manifest.json` for the supported conclusions and reproduction.
