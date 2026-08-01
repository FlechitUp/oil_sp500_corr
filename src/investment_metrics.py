from __future__ import annotations

import numpy as np
import pandas as pd


def reference_peak(prices: pd.Series, period: str = "Todo") -> tuple[pd.Timestamp, float]:
    clean = prices.dropna()
    lookback = {"52 semanas": 52, "104 semanas": 104}.get(period)
    sample = clean.tail(lookback) if lookback else clean
    peak = sample.max()
    date = sample[sample == peak].index[-1]
    return date, float(peak)


def drawdown_series(prices: pd.Series) -> pd.Series:
    clean = prices.dropna()
    return (clean / clean.cummax() - 1).rename("Drawdown")


def risk_return_metrics(prices: pd.Series, annual_rf: float, periods: int = 52) -> dict[str, float]:
    p = prices.dropna()
    simple = p.pct_change().dropna()
    years = len(simple) / periods
    total = p.iloc[-1] / p.iloc[0] - 1
    cagr = (p.iloc[-1] / p.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = simple.std(ddof=1) * np.sqrt(periods)
    weekly_rf = (1 + annual_rf) ** (1 / periods) - 1
    excess = simple - weekly_rf
    weekly_std = simple.std(ddof=1)
    # Una serie matemáticamente constante puede dejar ruido ~1e-16 en coma flotante.
    sharpe = excess.mean() / weekly_std * np.sqrt(periods) if weekly_std > 1e-12 else np.nan
    dd = drawdown_series(p)
    return {"Retorno total": float(total), "CAGR": float(cagr), "Volatilidad anualizada": float(vol),
            "Sharpe anualizado": float(sharpe), "Máximo drawdown": float(dd.min()),
            "Drawdown actual": float(dd.iloc[-1]), "Mejor semana": float(simple.max()),
            "Peor semana": float(simple.min()), "Semanas positivas": float((simple > 0).mean()),
            "Valor final de R$10.000": float(10000 * (1 + total)), "Tasa libre anual": annual_rf}
