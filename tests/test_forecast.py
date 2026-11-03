import numpy as np
import pandas as pd
from src.clv.cohorts import build_retention_matrix, retention_by_channel
from src.clv.bgf import predict_clv


def test_observed_zero_and_unobserved_missing():
    df=pd.DataFrame({"Customer ID":["a","a","b"],"InvoiceDate":pd.to_datetime(["2020-01-01","2020-03-01","2020-03-01"])})
    r=build_retention_matrix(df)
    assert r.loc[pd.Period("2020-01","M"),1] == 0
    assert r.loc[pd.Period("2020-01","M"),2] == 1
    assert np.isnan(r.loc[pd.Period("2020-03","M"),1])


def test_channels_share_observation_window():
    df=pd.DataFrame({"Customer ID":["a","b"],"InvoiceDate":pd.to_datetime(["2020-01-01","2020-03-01"]),"Channel":["A","B"]})
    r=retention_by_channel(df)
    assert r["A"].iloc[0,1] == 0
    assert r["A"].iloc[0,2] == 0


def test_exact_endpoint_and_discount(monkeypatch):
    seen=[]
    class B:
        def conditional_expected_number_of_purchases_up_to_time(self,t,*args):
            seen.append(t);return np.array([float(t),float(t)])
        def conditional_probability_alive(self,*args):return np.ones(2)
    class G:
        def conditional_expected_average_profit(self,*args):return np.array([2.])
    summary=pd.DataFrame({"frequency":[1.,0.],"recency":[2.,0.],"T":[3.,3.],"monetary_value":[2.,0.]})
    monkeypatch.setattr("src.clv.bgf.expected_purchases", lambda bg,t,s: bg.conditional_expected_number_of_purchases_up_to_time(t))
    result=predict_clv(B(),G(),summary,horizon_days=365,discount_rate=.01,fallback_spend=3.)
    expected=sum((1.01)**(-d/30) for d in range(1,366))
    assert max(seen)==365 and min(seen)==0
    np.testing.assert_allclose(result.predicted_clv,[2*expected,3*expected])
    np.testing.assert_allclose(result.predicted_purchases,[365,365])


def test_signed_hypergeometric_predictions_finite():
    from src.clv.bgf import expected_purchases
    class B:
        params_={"r":.5,"alpha":10.,"a":.2,"b":.3}
    s=pd.DataFrame({"frequency":[0.,2.],"recency":[0.,20.],"T":[60.,60.]})
    result=expected_purchases(B(),90,s)
    assert np.isfinite(result).all() and (result>0).all()


def test_prediction_against_independent_beta_integral():
    from scipy.integrate import quad
    from scipy.special import beta
    from src.clv.bgf import expected_purchases
    class B:
        params_={"r":.5,"alpha":10.,"a":.2,"b":.3}
    s=pd.DataFrame({"frequency":[0.,2.],"recency":[0.,20.],"T":[60.,60.]})
    actual=expected_purchases(B(),90,s)
    for i,row in s.iterrows():
        x=row.frequency;r=.5;alpha=10.;a=.2;b=.3;T=row["T"]
        alive=1/(1+(a/(b+x-1)*((alpha+T)/(alpha+row.recency))**(r+x) if x>0 else 0))
        def integrand(p):
            expected=-np.expm1(-(r+x)*np.log1p(p*90/(alpha+T)))/p
            return expected*p**(a-1)*(1-p)**(b+x-1)/beta(a,b+x)
        expected=alive*quad(integrand,0,1,epsabs=1e-7)[0]
        assert np.isclose(actual[i],expected,rtol=1e-6)
