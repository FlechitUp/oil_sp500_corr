from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_global_model(returns: pd.DataFrame, hac_lags: int = 4):
    clean = returns[["IVVB11", "IVV", "USDBRL"]].dropna()
    if len(clean) < 10:
        raise ValueError("Se necesitan al menos 10 observaciones para la regresión.")
    x = sm.add_constant(clean[["IVV", "USDBRL"]], has_constant="add")
    return sm.OLS(clean["IVVB11"], x).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})


def ols_contributions(returns: pd.DataFrame, model) -> pd.DataFrame:
    """Contribuciones log aditivas; alpha+residuo completa la identidad observada."""
    idx = returns.index.intersection(model.model.data.row_labels)
    r = returns.loc[idx]
    out = pd.DataFrame(index=idx)
    out["IVV"] = model.params["IVV"] * r["IVV"]
    out["USDBRL"] = model.params["USDBRL"] * r["USDBRL"]
    out["Alpha y residual"] = model.params["const"] + pd.Series(model.resid, index=idx)
    out["Modelado"] = out.sum(axis=1)
    out["Observado"] = r["IVVB11"]
    return out


def structural_tracking_difference(returns: pd.DataFrame) -> pd.Series:
    return (returns["IVVB11"] - returns["IVV"] - returns["USDBRL"]).rename("Tracking difference")


def compounded_log_return(log_returns: pd.Series) -> float:
    clean = log_returns.dropna()
    return float(np.expm1(clean.sum())) if len(clean) else np.nan


def tracking_statistics(td: pd.Series, periods: int = 52) -> dict[str, float]:
    clean = td.dropna()
    return {
        "media_anualizada": float(clean.mean() * periods),
        "volatilidad_anualizada": float(clean.std(ddof=1) * np.sqrt(periods)),
        "acumulado_simple_equivalente": compounded_log_return(clean),
    }

