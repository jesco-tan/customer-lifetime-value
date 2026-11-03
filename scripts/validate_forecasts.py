"""Frozen design: three chronological 90-day monetary holdouts; no policy selection."""
from pathlib import Path
import sys,json,hashlib,importlib.metadata,platform
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.clv.etl import load_synthetic,to_summary
from src.clv.bgf import fit_bgnbd,fit_gamma_gamma,predict_clv,clv_summary_stats
from src.clv.cohorts import build_retention_matrix,month1_retention_summary
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"
ORIGINS=["2011-03-01","2011-06-01","2011-09-01"]

def main():
    df=load_synthetic(n_customers=2500,seed=42)
    metrics=[];predictions=[];diagnostics=[]
    for origin in ORIGINS:
        cutoff=pd.Timestamp(origin);end=cutoff+pd.Timedelta(days=90)
        train=df[df.InvoiceDate<=cutoff];hold=df[(df.InvoiceDate>cutoff)&(df.InvoiceDate<=end)]
        summary=to_summary(train,cutoff);bg=fit_bgnbd(summary);gg=fit_gamma_gamma(summary)
        # All parameters use training only. Shrink single-customer invoice spend toward pooled mean, prior strength 5 fixed in advance.
        invoices=train.groupby(["Customer ID","Invoice"])["Revenue"].sum()
        pop=float(invoices.mean());own=invoices.groupby(level=0).mean().reindex(summary.index);n=invoices.groupby(level=0).size().reindex(summary.index)
        shrink=(n*own+5*pop)/(n+5)
        actual_n=hold.groupby("Customer ID").InvoiceDate.nunique().reindex(summary.index,fill_value=0)
        actual_rev=hold.groupby("Customer ID").Revenue.sum().reindex(summary.index,fill_value=0)
        rate=(summary.frequency+1)/(summary["T"]+1)*90
        groups=pd.Series(np.where(summary.frequency>0,"repeat","single"),index=summary.index)
        channels=train.groupby("Customer ID").Channel.first().reindex(summary.index)
        diagnostics.append({"origin":origin,"training_population_invoice_mean":pop,"frequency_spend_spearman":float(summary.loc[summary.frequency>0,["frequency","monetary_value"]].corr(method="spearman").iloc[0,1]),"n":len(summary)})
        for policy,fallback in [("zero",0.),("population",pop),("shrinkage",shrink),("historical_rate_spend",None)]:
            if policy=="historical_rate_spend": pn=rate;pr=rate*own
            else:
                pred=predict_clv(bg,gg,summary,horizon_days=90,discount_rate=0.,fallback_spend=fallback)
                pn=pred.predicted_purchases;pr=pred.predicted_clv
            frame=pd.DataFrame({"origin":origin,"policy":policy,"group":groups,"channel":channels,"predicted_count":pn,"actual_count":actual_n,"predicted_revenue":pr,"actual_revenue":actual_rev})
            assert np.isfinite(frame[["predicted_count","predicted_revenue","actual_count","actual_revenue"]]).all().all()
            predictions.append(frame.reset_index())
            for label,mask in [("all",np.ones(len(frame),dtype=bool)),("repeat",groups=="repeat"),("single",groups=="single")]+[(str(c),channels==c) for c in sorted(channels.unique())]:
                f=frame.loc[mask]
                row={"origin":origin,"policy":policy,"group":label,"n":len(f)}
                for target in ["count","revenue"]:
                    err=f["predicted_"+target]-f["actual_"+target]
                    row.update({target+"_mae":float(err.abs().mean()),target+"_bias":float(err.mean()),target+"_rmse":float(np.sqrt((err**2).mean()))})
                metrics.append(row)
    validation=pd.DataFrame(metrics);validation.to_csv(OUT/"forecast_validation.csv",index=False, lineterminator="\n")
    pd.concat(predictions).to_csv(OUT/"holdout_predictions.csv",index=False, lineterminator="\n")
    (OUT/"forecast_diagnostics.json").write_text(json.dumps(diagnostics,indent=2)+"\n", newline="\n", encoding="utf-8")
    retention=build_retention_matrix(df);retention.to_csv(OUT/"retention.csv", lineterminator="\n")
    month1_retention_summary(df).to_csv(OUT/"month1_retention.csv",index=False, lineterminator="\n")
    fig,ax=plt.subplots(figsize=(10,6));im=ax.imshow(retention,aspect="auto",vmin=0,vmax=1);fig.colorbar(im,ax=ax,label="Retention");ax.set_xlabel("Months since acquisition (observed zero months included)");ax.set_ylabel("Acquisition cohort");fig.tight_layout();fig.savefig(OUT/"retention_heatmap.png");plt.close(fig)
    final=to_summary(df,pd.Timestamp("2011-12-09"));bg=fit_bgnbd(final);gg=fit_gamma_gamma(final)
    # Population fallback fixed as the reporting policy before evaluating holdouts; no best-of-holdout selection.
    pop=float(df.groupby(["Customer ID","Invoice"]).Revenue.sum().mean())
    forecast=predict_clv(bg,gg,final,365,.01,pop);forecast.to_csv(OUT/"clv_predictions.csv", lineterminator="\n")
    stats=clv_summary_stats(forecast)
    output={"data_source":"synthetic","seed":42,"n_customers":2500,"forecast_days":365,"discount":"Daily increments discounted by (1.01)^(day/30), days 1..365","reporting_fallback":"training population invoice mean; specified before holdout comparison","clv_stats":stats,"validation_origins":ORIGINS,"holdout_days":90,"validation_target":"undiscounted revenue and daily purchase counts, existing customers only","origins_overlap":False}
    (OUT/"run_summary.json").write_text(json.dumps(output,indent=2)+"\n", newline="\n", encoding="utf-8")
    fig,ax=plt.subplots(figsize=(9,5));forecast.predicted_clv.plot.hist(bins=50,ax=ax);ax.set_xlabel("365-day discounted revenue, synthetic GBP");fig.tight_layout();fig.savefig(OUT/"clv_distribution.png");plt.close(fig)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    inputs={p.relative_to(ROOT).as_posix():digest(p) for p in (ROOT/"src").rglob("*.py")};inputs["scripts/validate_forecasts.py"]=digest(Path(__file__))
    manifest={"status":"complete","python":platform.python_version(),"parameters":{"seed":42,"origins":ORIGINS,"horizon":90,"shrinkage_strength":5,"penalizer":.001},"inputs":inputs,"packages":{d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},"outputs":{name:digest(OUT/name) for name in ["forecast_validation.csv","holdout_predictions.csv","run_summary.json","clv_predictions.csv","retention.csv"]},"tolerance":"Metrics absolute 1e-6 with pinned packages. No interval claims. Synthetic results only."}
    (OUT/"forecast_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n", newline="\n", encoding="utf-8")
    print(validation[validation.group=="all"].to_string(index=False))

if __name__=="__main__":main()
