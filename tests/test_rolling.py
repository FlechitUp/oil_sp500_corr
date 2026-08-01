import numpy as np
import pandas as pd
from src.rolling_diagnostics import lagged_rolling_zscore

def test_zscore_has_no_lookahead():
    s=pd.Series(np.arange(20,dtype=float)); z1=lagged_rolling_zscore(s,10,5)
    changed=s.copy(); changed.iloc[-1]=999; z2=lagged_rolling_zscore(changed,10,5)
    assert z1.iloc[:-1].equals(z2.iloc[:-1])
    expected=(999-s.iloc[-11:-1].mean())/s.iloc[-11:-1].std(ddof=1)
    assert z2.iloc[-1]==expected

