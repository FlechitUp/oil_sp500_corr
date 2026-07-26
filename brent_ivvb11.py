"""
Descomposicion de los retornos del IVVB11 en sus factores estructurales.

IVVB11 replica el S&P 500 (via el ETF estadounidense IVV), cotizado en BRL y
sin cobertura cambiaria. Por identidad:  IVVB11 ≈ IVV (USD) × USD/BRL.
Sus retornos tienen por tanto dos motores -- mercado americano y tipo de
cambio -- mas un componente LOCAL (prima/descuento del ETF, liquidez de B3,
ruido).

  ============================ LEER ANTES DE USAR ============================
  Con datos DIARIOS gratuitos (Yahoo) esta descomposicion FALLA: el cierre de
  B3 (17h BRT), el del FX y el de EE.UU. no estan sincronizados, y el retorno
  diario del USD/BRL de Yahoo no cuadra con la ventana del IVVB11. Resultado:
  el beta del FX colapsa a ~0 y su varianza aparente cae a ~0%, lo cual es
  FALSO (el dolar si mueve al IVVB11, solo que mal medido a diario).

  Evidencia del problema y su solucion:
     regresion en NIVELES   log(IVVB11)~log(IVV*USDBRL): R2=0.999, beta≈1  OK
     retornos DIARIOS:       beta_FX≈0.03  <- roto por desfase horario
     retornos SEMANALES:     beta_FX≈0.72  <- recuperado

  Por eso este script trabaja en frecuencia SEMANAL (viernes), donde el
  desfase intradiario se diluye y ambos factores vuelven a ser visibles.
  Para trabajo diario serio se necesita FX y NAV con horario de corte alineado
  a B3 (datos de pago), no Yahoo.
  ===========================================================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm

TICKERS = {"IVVB11.SA": "IVVB11", "IVV": "IVV", "BRL=X": "USDBRL"}
START = "2020-01-01"
FREQ = "W-FRI"          # semanal; pon "D" para diario (advertencia arriba)


def cargar(tickers=TICKERS, start=START, freq=FREQ) -> pd.DataFrame:
    d = yf.download(list(tickers), start=start, auto_adjust=True, progress=False)["Close"]
    d = d[list(tickers)].rename(columns=tickers).dropna(how="any")
    d = d.resample(freq).last().dropna()
    return np.log(d / d.shift(1)).dropna()


def descomponer(ret: pd.DataFrame):
    """r_ivvb = alpha + b_us*r_ivv + b_fx*r_fx + e. Devuelve modelo y contribuciones."""
    X = sm.add_constant(ret[["IVV", "USDBRL"]])
    m = sm.OLS(ret["IVVB11"], X).fit()
    contrib = pd.DataFrame({
        "S&P 500 (IVV)": m.params["IVV"] * ret["IVV"],
        "USD/BRL": m.params["USDBRL"] * ret["USDBRL"],
        "Local (residuo)": m.resid + m.params["const"],
    }, index=ret.index)
    return m, contrib


def importancia(ret, m) -> pd.Series:
    c_us = m.params["IVV"] * ret["IVV"]
    c_fx = m.params["USDBRL"] * ret["USDBRL"]
    vt = ret["IVVB11"].var()
    return pd.Series({
        "S&P 500 (IVV)": c_us.var() / vt,
        "USD/BRL": c_fx.var() / vt,
        "Covarianza": 2 * c_us.cov(c_fx) / vt,
        "Local / no explicado": m.resid.var() / vt,
    })


def graficar(contrib, ret, n=16):
    sub = contrib.tail(n) * 100
    obs = ret["IVVB11"].tail(n) * 100
    x = np.arange(len(sub))
    col = {"S&P 500 (IVV)": "#1F618D", "USD/BRL": "#E67E22", "Local (residuo)": "#95A5A6"}

    fig, ax = plt.subplots(figsize=(13, 6.5))
    pos = np.zeros(len(sub)); neg = np.zeros(len(sub))
    for c in ["S&P 500 (IVV)", "USD/BRL", "Local (residuo)"]:
        v = sub[c].values
        ax.bar(x, v, bottom=np.where(v >= 0, pos, neg), color=col[c], label=c, width=0.7)
        pos += np.where(v >= 0, v, 0); neg += np.where(v < 0, v, 0)
    ax.plot(x, obs.values, "ko", markersize=5, label="IVVB11 observado", zorder=5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%d/%m/%y") for d in sub.index], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Contribucion al retorno semanal (%)")
    ax.set_title(f"Descomposicion del retorno SEMANAL del IVVB11 - ultimas {n} semanas\n"
                 "(las barras suman el punto negro observado)", fontsize=11)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ret = cargar()
    m, contrib = descomponer(ret)

    print(f"Frecuencia: {FREQ} | Periodo: {ret.index[0]:%d/%m/%Y} a {ret.index[-1]:%d/%m/%Y} (n={len(ret)})\n")
    print("MODELO  r_ivvb = alpha + b_us*r_IVV + b_fx*r_USDBRL")
    print(f"  beta IVV     = {m.params['IVV']:.3f}   (teorico ~1)")
    print(f"  beta USD/BRL = {m.params['USDBRL']:.3f}   (teorico ~1)")
    print(f"  R2 = {m.rsquared:.3f}\n")

    print("Importancia relativa (descomposicion de varianza):")
    for k, v in importancia(ret, m).items():
        print(f"  {k:<22} {v*100:6.1f}%")

    print("\nUltimas 4 semanas (contribucion en %):")
    t = (contrib.tail(4) * 100).round(3)
    t["TOTAL"] = t.sum(axis=1)
    t.index = [d.strftime("%d/%m/%y") for d in t.index]
    print(t.to_string())

    graficar(contrib, ret).savefig("descomposicion_ivvb11.png", dpi=150)