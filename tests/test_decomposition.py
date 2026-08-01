import numpy as np
import pandas as pd
from src.decomposition import fit_global_model, ols_contributions, structural_tracking_difference, compounded_log_return

def sample():
    rng=np.random.default_rng(7); idx=pd.date_range("2020-01-03", periods=120, freq="W-FRI")
    ivv=rng.normal(.001,.02,120); fx=rng.normal(0,.015,120)
    return pd.DataFrame({"IVV":ivv,"USDBRL":fx,"IVVB11":.0002+1.1*ivv+.9*fx+rng.normal(0,.002,120)},index=idx)

def test_contributions_reconstruct_observed():
    r=sample(); c=ols_contributions(r,fit_global_model(r)); assert np.allclose(c["Modelado"],c["Observado"],atol=1e-12)
def test_tracking_difference():
    r=sample(); assert np.allclose(structural_tracking_difference(r),r.IVVB11-r.IVV-r.USDBRL)
def test_log_compounding():
    x=pd.Series(np.log([1.1,.9,1.05])); assert compounded_log_return(x)==pytest.approx(1.1*.9*1.05-1)
import pytest

