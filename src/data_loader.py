from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = {"IVVB11.SA": "IVVB11", "IVV": "IVV", "BRL=X": "USDBRL"}


class DataDownloadError(RuntimeError):
    pass


def _close_frame(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extrae Close de las distintas formas que devuelve yfinance."""
    if raw.empty:
        raise DataDownloadError("Yahoo Finance no devolvió datos.")
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        level1 = raw.columns.get_level_values(1)
        if "Close" in level0:
            close = raw["Close"]
        elif "Close" in level1:
            close = raw.xs("Close", axis=1, level=1)
        else:
            raise DataDownloadError("La respuesta no contiene precios de cierre ajustados.")
    else:
        if "Close" not in raw.columns:
            raise DataDownloadError("La respuesta no contiene precios de cierre ajustados.")
        close = raw[["Close"]].copy()
        if len(tickers) == 1:
            close.columns = tickers
    return close


def download_adjusted_prices(tickers: list[str], start, end=None) -> pd.DataFrame:
    try:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                          progress=False, group_by="column", threads=True)
        close = _close_frame(raw, tickers)
    except Exception as exc:
        if isinstance(exc, DataDownloadError):
            raise
        raise DataDownloadError(f"No fue posible descargar datos: {exc}") from exc
    close = close.reindex(columns=tickers).apply(pd.to_numeric, errors="coerce")
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]


def prepare_weekly_data(daily: pd.DataFrame, names: dict[str, str] = TICKERS,
                        freq: str = "W-FRI") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    available = daily.rename(columns=names)
    weekly_unaligned = available.resample(freq).last()
    missing = weekly_unaligned.isna().sum().to_dict()
    aligned = weekly_unaligned.dropna(how="any")
    if len(aligned) < 3:
        raise DataDownloadError("No hay suficientes semanas coincidentes entre las series.")
    log_returns = np.log(aligned / aligned.shift(1)).dropna(how="any")
    report = {
        "semanas_antes_alinear": len(weekly_unaligned),
        "semanas_comunes": len(aligned),
        "semanas_eliminadas": len(weekly_unaligned) - len(aligned),
        "ausentes_por_serie": missing,
        "ultima_fecha": aligned.index.max(),
    }
    return aligned, log_returns, report

