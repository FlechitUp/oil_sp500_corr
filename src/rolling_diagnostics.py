from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def rolling_regression(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    clean = returns[["IVVB11", "IVV", "USDBRL"]].dropna()
    rows = []
    for end in range(window, len(clean) + 1):
        sample = clean.iloc[end-window:end]
        x = sm.add_constant(sample[["IVV", "USDBRL"]], has_constant="add")
        fit = sm.OLS(sample["IVVB11"], x).fit()
        date = sample.index[-1]
        rows.append({"Fecha": date, "Alpha": fit.params["const"],
                     "Beta IVV": fit.params["IVV"], "Beta USD/BRL": fit.params["USDBRL"],
                     "R²": fit.rsquared, "Residuo": fit.resid.iloc[-1]})
    return pd.DataFrame(rows).set_index("Fecha") if rows else pd.DataFrame(
        columns=["Alpha", "Beta IVV", "Beta USD/BRL", "R²", "Residuo"])


def historical_model_residuals(returns: pd.DataFrame, min_train: int = 26) -> pd.Series:
    """Residuo one-step-ahead con modelo estimado solo hasta t-1."""
    clean = returns[["IVVB11", "IVV", "USDBRL"]].dropna()
    out = pd.Series(np.nan, index=clean.index, name="Residuo")
    for i in range(min_train, len(clean)):
        train = clean.iloc[:i]
        fit = sm.OLS(train["IVVB11"], sm.add_constant(train[["IVV", "USDBRL"]], has_constant="add")).fit()
        row = clean.iloc[i]
        pred = fit.params["const"] + fit.params["IVV"]*row["IVV"] + fit.params["USDBRL"]*row["USDBRL"]
        out.iloc[i] = row["IVVB11"] - pred
    return out


def lagged_rolling_zscore(values: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Normaliza t usando exclusivamente valores anteriores a t (sin look-ahead)."""
    min_periods = min_periods or max(10, window // 2)
    past = values.shift(1)
    mean = past.rolling(window, min_periods=min_periods).mean()
    std = past.rolling(window, min_periods=min_periods).std(ddof=1).replace(0, np.nan)
    return ((values - mean) / std).rename("Z-score")


def anomaly_label(z: float) -> str:
    if pd.isna(z): return "Sin datos suficientes"
    if abs(z) >= 3: return "Extremo"
    if abs(z) >= 2: return "Inusual"
    return "Normal"

