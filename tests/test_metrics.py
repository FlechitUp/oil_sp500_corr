import numpy as np
import pandas as pd
import pytest
from src.investment_metrics import reference_peak, drawdown_series, risk_return_metrics

def test_peak_latest_tie_and_drawdown():
    s=pd.Series([100,120,110,120,90],index=pd.date_range("2024-01-05",periods=5,freq="W-FRI"))
    d,p=reference_peak(s); assert d==s.index[3] and p==120
    assert drawdown_series(s).iloc[-1]==pytest.approx(-.25)
def test_metrics_controlled_series():
    p=pd.Series(100*1.01**np.arange(53),index=pd.date_range("2024-01-05",periods=53,freq="W-FRI"))
    m=risk_return_metrics(p,.02); assert m["CAGR"]==pytest.approx(1.01**52-1)
    assert m["Volatilidad anualizada"]==pytest.approx(0,abs=1e-12)
    assert np.isnan(m["Sharpe anualizado"])

