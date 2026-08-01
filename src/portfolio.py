from __future__ import annotations

import numpy as np
import pandas as pd


def validate_portfolio(table: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    errors = []
    if table is None or table.empty: return pd.DataFrame(), ["Introduce al menos un activo."]
    out = table.copy()
    out["Ticker"] = out["Ticker"].astype(str).str.strip().str.upper()
    out["Peso (%)"] = pd.to_numeric(out["Peso (%)"], errors="coerce")
    if out["Ticker"].eq("").any(): errors.append("Todos los tickers deben estar informados.")
    if out["Ticker"].duplicated().any(): errors.append("No repitas tickers.")
    if out["Peso (%)"].isna().any() or (out["Peso (%)"] < 0).any(): errors.append("Los pesos deben ser números no negativos.")
    if not np.isclose(out["Peso (%)"].sum(), 100, atol=0.01): errors.append(f"Los pesos suman {out['Peso (%)'].sum():.2f}%; deben sumar 100%.")
    return out, errors


def weighted_portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    missing = set(weights.index) - set(asset_returns.columns)
    if missing: raise ValueError("Faltan retornos para: " + ", ".join(sorted(missing)))
    aligned = asset_returns[list(weights.index)].dropna(how="any")
    if aligned.empty: raise ValueError("No hay fechas comunes para construir la cartera.")
    return aligned.mul(weights / weights.sum(), axis=1).sum(axis=1).rename("Mi cartera")

