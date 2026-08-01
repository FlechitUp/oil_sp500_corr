import pandas as pd
import pytest
from src.portfolio import validate_portfolio, weighted_portfolio_returns

def test_weights_and_weighted_return():
    good=pd.DataFrame({"Ticker":["A","B"],"Peso (%)":[60,40]}); valid,errors=validate_portfolio(good)
    assert not errors
    r=pd.DataFrame({"A":[.1,0],"B":[0,.1]}); out=weighted_portfolio_returns(r,valid.set_index("Ticker")["Peso (%)"])
    assert out.tolist()==pytest.approx([.06,.04])
    _,errors=validate_portfolio(pd.DataFrame({"Ticker":["A"],"Peso (%)":[90]})); assert errors

