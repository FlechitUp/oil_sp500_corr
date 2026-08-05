"""
Descomposicion de los retornos del IVVB11 en sus factores estructurales,
con panel de PRECIO encima para ver nivel y movimiento a la vez.

IVVB11 ≈ IVV (USD) × USD/BRL. Sus retornos se reparten en:
mercado americano (IVV), tipo de cambio (USD/BRL) y componente Residual (local).

Frecuencia SEMANAL (viernes): a diario el desfase horario entre B3, el FX y
EE.UU. rompe la relacion cambiaria (beta_FX colapsa a ~0). En semanal el
desfase se diluye y ambos factores vuelven a ser visibles. Para diario fiable
hacen falta FX y NAV con horario de corte de B3 (datos de pago), no Yahoo.

Paneles:
  ARRIBA  precio de cierre del IVVB11 (nivel, en BRL)
  ABAJO   descomposicion del retorno de cada semana (movimiento, en %)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm

TICKERS = {"IVVB11.SA": "IVVB11", "IVV": "IVV", "BRL=X": "USDBRL"}
START = "2020-01-01"
FREQ = "W-FRI"


def cargar(tickers=TICKERS, start=START, freq=FREQ):
    d = yf.download(list(tickers), start=start, auto_adjust=True, progress=False)["Close"]
    d = d[list(tickers)].rename(columns=tickers).dropna(how="any")
    precio = d.resample(freq).last().dropna()      # nivel = cierre semanal > last() = viernes , mean()= promedio semanal, first() = lunes
    ret = np.log(precio / precio.shift(1)).dropna()
    return precio, ret


def descomponer(ret):
    X = sm.add_constant(ret[["IVV", "USDBRL"]])
    m = sm.OLS(ret["IVVB11"], X).fit()
    contrib = pd.DataFrame({
        "S&P 500 (IVV)": m.params["IVV"] * ret["IVV"],
        "USD/BRL": m.params["USDBRL"] * ret["USDBRL"],
        "Local (residuo)": m.resid + m.params["const"],
    }, index=ret.index)
    return m, contrib


def importancia(ret, m):
    c_us = m.params["IVV"] * ret["IVV"]; c_fx = m.params["USDBRL"] * ret["USDBRL"]
    vt = ret["IVVB11"].var()
    return pd.Series({"S&P 500 (IVV)": c_us.var()/vt, "USD/BRL": c_fx.var()/vt,
                      "Covarianza": 2*c_us.cov(c_fx)/vt, "Local (residuo) / no explicado": m.resid.var()/vt})


def graficar(precio, contrib, ret, n=16):
    px = precio["IVVB11"].tail(n)
    sub = contrib.tail(n) * 100
    obs = ret["IVVB11"].tail(n) * 100
    x = np.arange(len(sub))
    fechas = [d.strftime("%d/%m/%y") for d in sub.index]
    col = {"S&P 500 (IVV)": "#1F618D", "USD/BRL": "#E67E22", "Local (residuo)": "#95A5A6"}

    fig, (axp, axd) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.3]})

    # --- panel superior: PRECIO (nivel) ---
    axp.plot(x, px.values, color="#111", linewidth=1.8, marker="o", markersize=5, zorder=3)
    for xi, yi in zip(x, px.values):
        axp.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=7.5, color="#333")
    axp.set_ylabel("Precio de cierre IVVB11 (BRL)")
    axp.set_title("IVVB11: precio (arriba) y de que vino cada movimiento (abajo)", fontsize=12)
    axp.grid(alpha=0.3)

    # --- panel inferior: DESCOMPOSICION (movimiento) ---
    pos = np.zeros(len(sub)); neg = np.zeros(len(sub))
    for c in ["S&P 500 (IVV)", "USD/BRL", "Local (residuo)"]:
        v = sub[c].values
        axd.bar(x, v, bottom=np.where(v >= 0, pos, neg), color=col[c], label=c, width=0.7)
        pos += np.where(v >= 0, v, 0); neg += np.where(v < 0, v, 0)
    axd.plot(x, obs.values, "ko", markersize=5, label="IVVB11 (movimiento)", zorder=5)
    for xi, yi in zip(x, obs.values):
        axd.annotate(f"{yi:+.2f}%", (xi, yi), textcoords="offset points",
                     xytext=(0, -12 if yi >= 0 else 8), ha="center",
                     fontsize=7.5, fontweight="bold", color="#000")
    axd.axhline(0, color="black", linewidth=0.8)
    axd.set_xticks(x); axd.set_xticklabels(fechas, rotation=45, ha="right", fontsize=8)
    axd.set_ylabel("Contribucion al retorno semanal (%)")
    axd.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    axd.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    precio, ret = cargar()
    m, contrib = descomponer(ret)

    print(f"Frecuencia: {FREQ} | {ret.index[0]:%d/%m/%Y} a {ret.index[-1]:%d/%m/%Y} (n={len(ret)})\n")
    print(f"beta IVV = {m.params['IVV']:.3f} | beta USD/BRL = {m.params['USDBRL']:.3f} | R2 = {m.rsquared:.3f}\n")
    print("Descomposición de la varianza del retorno")
    for k, v in importancia(ret, m).items():
        print(f"  {k:<22} {v*100:6.1f}%")
    print("\nUltimas 4 semanas:")
    t = (contrib.tail(4) * 100).round(2)
    t.insert(0, "Precio", precio["IVVB11"].tail(4).round(2).values)
    t["Mov.TOTAL %"] = (ret["IVVB11"].tail(4) * 100).round(2).values
    t.index = [d.strftime("%d/%m/%y") for d in t.index]
    print(t.to_string())

    graficar(precio, contrib, ret).savefig("descomposicion_ivvb11.png", dpi=150)