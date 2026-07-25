"""
Brent (BZ=F) vs IVVB11.SA — sincronización, normalización base 100 y correlación.

Notas metodológicas:
- Los activos cotizan en mercados y monedas distintas (Brent en USD, IVVB11 en BRL),
  con calendarios de feriados diferentes. Por eso la sincronización se hace por
  intersección de fechas (inner join), no por relleno hacia adelante indiscriminado.
- La correlación entre NIVELES de precio está inflada por la tendencia común
  (correlación espuria). La medida estadísticamente válida es la correlación
  entre RETORNOS. Se reportan ambas para dejar clara la diferencia.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

TICKERS = {"BZ=F": "Brent (USD)", "IVVB11.SA": "IVVB11 (BRL)"}
START, END = "2020-01-01", None


def descargar(tickers, start=START, end=END) -> pd.DataFrame:
    """Descarga precios de cierre ajustados y devuelve un DataFrame de columnas por activo."""
    df = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,      # ajusta por splits/dividendos
        progress=False,
        group_by="column",
    )["Close"]

    # yfinance puede devolver Serie si hay un solo ticker
    if isinstance(df, pd.Series):
        df = df.to_frame()

    df = df[list(tickers)].rename(columns=tickers)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def sincronizar(df: pd.DataFrame, max_gap: int = 3) -> pd.DataFrame:
    """
    Alinea calendarios distintos:
    1) elimina filas totalmente vacías,
    2) rellena huecos cortos (feriado en un solo mercado) con el último precio,
       limitando el arrastre a `max_gap` días para no fabricar precios,
    3) se queda solo con las fechas donde AMBOS activos tienen dato.
    """
    df = df.dropna(how="all").sort_index()
    df = df.ffill(limit=max_gap)
    df = df.dropna(how="any")
    return df


def normalizar_base100(df: pd.DataFrame) -> pd.DataFrame:
    """Reescala cada serie para que su primer valor observado sea 100."""
    return df.div(df.iloc[0]).mul(100)


def correlaciones(df: pd.DataFrame) -> dict:
    """Correlación de niveles (referencial) y de retornos logarítmicos (válida)."""
    ret = np.log(df / df.shift(1)).dropna()
    return {
        "niveles_pearson": df.corr().iloc[0, 1],
        "retornos_pearson": ret.corr().iloc[0, 1],
        "retornos_spearman": ret.corr(method="spearman").iloc[0, 1],
        "retornos": ret,
    }


def graficar(base100: pd.DataFrame, ret: pd.DataFrame, rho: float, ventana: int = 60):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    base100.plot(ax=ax1, linewidth=1.5)
    ax1.axhline(100, color="grey", linestyle="--", linewidth=0.8)
    ax1.set_title(f"Brent vs IVVB11 — base 100 ({base100.index[0]:%d/%m/%Y})")
    ax1.set_ylabel("Índice (base 100)")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.3)

    rolling = ret.iloc[:, 0].rolling(ventana).corr(ret.iloc[:, 1])
    rolling.plot(ax=ax2, color="firebrick", linewidth=1.2)
    ax2.axhline(rho, color="black", linestyle="--", linewidth=0.9,
                label=f"correlación total = {rho:.3f}")
    ax2.axhline(0, color="grey", linewidth=0.8)
    ax2.set_title(f"Correlación móvil de retornos ({ventana} días)")
    ax2.set_ylabel("ρ")
    ax2.set_ylim(-1, 1)
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    precios = sincronizar(descargar(TICKERS))
    base100 = normalizar_base100(precios)
    res = correlaciones(precios)

    print(f"Observaciones sincronizadas: {len(precios)} "
          f"({precios.index[0]:%d/%m/%Y} a {precios.index[-1]:%d/%m/%Y})\n")
    print(base100.tail(3).round(2), "\n")
    print(f"Correlación de niveles  (Pearson) : {res['niveles_pearson']:.4f}  <- espuria")
    print(f"Correlación de retornos (Pearson) : {res['retornos_pearson']:.4f}")
    print(f"Correlación de retornos (Spearman): {res['retornos_spearman']:.4f}")

    fig = graficar(base100, res["retornos"], res["retornos_pearson"])
    fig.savefig("brent_ivvb11.png", dpi=150)
