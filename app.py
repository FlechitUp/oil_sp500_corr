from __future__ import annotations

from datetime import date
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.charts import *
from src.data_loader import TICKERS, DataDownloadError, download_adjusted_prices, prepare_weekly_data
from src.decomposition import fit_global_model, ols_contributions, structural_tracking_difference, tracking_statistics, compounded_log_return
from src.investment_metrics import reference_peak, drawdown_series, risk_return_metrics
from src.interpretation import classify_episode, residual_description
from src.portfolio import validate_portfolio, weighted_portfolio_returns
from src.rolling_diagnostics import rolling_regression, historical_model_residuals, lagged_rolling_zscore, anomaly_label

st.set_page_config(page_title="Análisis histórico de IVVB11", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: #EAF6FB;
        border: 1px solid #D3EAF4;
        border-radius: 10px;
        padding: 16px 18px;
        min-height: 118px;
    }
    div[data-testid="stMetric"] label {
        color: #315568;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Análisis histórico de IVVB11")
st.caption("Descomposición entre mercado estadounidense, USD/BRL y componentes no explicados por el modelo")

@st.cache_data(ttl=6*60*60, show_spinner=False)
def cached_download(tickers, start):
    return download_adjusted_prices(list(tickers), start)

def pct(v): return "—" if pd.isna(v) else f"{v*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")
def brl(v): return "—" if pd.isna(v) else f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
def metric_row(items):
    for col, (label, value, help_) in zip(st.columns(len(items)), items): col.metric(label, value, help=help_)

with st.sidebar:
    st.header("Parámetros")
    start = st.date_input("Fecha inicial", value=date(2020,1,1), max_value=date.today())
    freq = st.selectbox("Frecuencia", ["W-FRI"], help="Cierre semanal correspondiente al viernes; en feriados se usa la última observación disponible de la semana.")
    weeks = st.selectbox("Semanas mostradas", [8,16,26,52], index=1)
    window = st.selectbox("Ventana rolling", [26,52,104], index=1)
    peak_period = st.selectbox("Periodo para identificar el máximo", ["Todo", "52 semanas", "104 semanas"])
    rf_pct = st.number_input("Tasa libre de riesgo anual (%)", min_value=-20.0, max_value=100.0, value=10.0, step=.25,
                             help="Valor introducido por el usuario; no se presupone una tasa vigente.")
    if st.button("Actualizar / limpiar caché", use_container_width=True):
        st.cache_data.clear(); st.rerun()

try:
    daily = cached_download(tuple(TICKERS), str(start))
    prices, returns, report = prepare_weekly_data(daily, TICKERS, freq)
except (DataDownloadError, Exception) as exc:
    st.error(f"No fue posible preparar los datos de Yahoo Finance. Detalle: {exc}")
    st.stop()

model = fit_global_model(returns)
contrib = ols_contributions(returns, model)
rolling = rolling_regression(returns, window)
historical_resid = historical_model_residuals(returns, min_train=max(15, window//2))
zscore = lagged_rolling_zscore(historical_resid, window)
td = structural_tracking_difference(returns)
peak_date, peak_price = reference_peak(prices["IVVB11"], peak_period)
current_dd_ref = prices["IVVB11"].iloc[-1]/peak_price - 1

st.info(f"Última fecha disponible: **{report['ultima_fecha']:%d/%m/%Y}** · Semanas antes de alinear: {report['semanas_antes_alinear']} · comunes: {report['semanas_comunes']} · eliminadas: {report['semanas_eliminadas']}.")
tabs = st.tabs(["Resumen", "Movimientos semanales", "Desde el último máximo", "Diagnóstico rolling", "Riesgo y retorno", "Mi cartera"])

with tabs[0]:
    last_roll = rolling.iloc[-1] if not rolling.empty else pd.Series(dtype=float)
    current_z = zscore.dropna().iloc[-1] if zscore.notna().any() else np.nan
    last_week = np.expm1(returns["IVVB11"].iloc[-1])
    metric_row([("Precio reciente", brl(prices['IVVB11'].iloc[-1]), "Cierre semanal ajustado."),
                ("Última semana", pct(last_week), "Retorno simple equivalente al retorno log semanal."),
                ("Máximo de referencia", brl(peak_price), peak_period), ("Fecha del máximo", peak_date.strftime("%d/%m/%Y"), peak_period),
                ("Drawdown desde máximo", pct(current_dd_ref), "Respecto del máximo seleccionado.")])
    metric_row([("Beta IVV", f"{last_roll.get('Beta IVV', np.nan):.2f}", "Coeficiente descriptivo rolling."),
                ("Beta USD/BRL", f"{last_roll.get('Beta USD/BRL', np.nan):.2f}", "Coeficiente descriptivo rolling."),
                ("R²", pct(last_roll.get('R²', np.nan)), "Fracción explicada dentro de la ventana."),
                ("Z-score residual", f"{current_z:.2f}" if pd.notna(current_z) else "—", "Calculado solo con información anterior.")])
    since = contrib.loc[contrib.index > peak_date]
    vals = {"IVV": since["IVV"].sum(), "FX": since["USDBRL"].sum(), "Residual": since["Alpha y residual"].sum()}
    category = classify_episode(vals["IVV"], vals["FX"], vals["Residual"], np.log1p(current_dd_ref))
    st.subheader("Diagnóstico descriptivo")
    st.write(f"IVVB11 se encuentra {abs(current_dd_ref)*100:.2f}% por debajo del máximo registrado el {peak_date:%d/%m/%Y}. "
             f"La clasificación del episodio es: **{category.lower()}**. {residual_description(current_z)}")

with tabs[1]:
    st.plotly_chart(weekly_decomposition(prices["IVVB11"], contrib, weeks), use_container_width=True)
    st.caption("El componente ‘Residual, timing y tracking’ reúne alpha y residuo OLS y puede reflejar horarios de cierre, tracking error, gastos, diferencia entre precio y valor patrimonial, feriados, limitaciones de Yahoo Finance y errores no explicados.")
    st.info("Las barras están en puntos porcentuales logarítmicos y suman exactamente el retorno log observado. El porcentaje convencional equivalente es exp(suma de retornos log) − 1.")

with tabs[2]:
    since = contrib.loc[contrib.index > peak_date]
    log_total = returns.loc[returns.index > peak_date, "IVVB11"].sum()
    acc = {"IVVB11 observado": log_total, "IVV": since["IVV"].sum(), "USD/BRL": since["USDBRL"].sum(), "Residual": since["Alpha y residual"].sum()}
    weeks_elapsed = len(since)
    table = pd.DataFrame({"Componente": list(acc), "Acumulado (pp log)": np.array(list(acc.values()))*100})
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.plotly_chart(accumulated_bar(acc), use_container_width=True)
    st.plotly_chart(line_chart(prices.loc[peak_date:], ["IVVB11"], "Precio de IVVB11 desde el máximo", {"IVVB11": BLACK}, "R$"), use_container_width=True)
    category = classify_episode(acc["IVV"], acc["USD/BRL"], acc["Residual"], log_total)
    st.write(f"En {weeks_elapsed} semanas, el retorno convencional acumulado fue {pct(np.expm1(log_total))}. Clasificación: **{category}**.")
    st.info("Las contribuciones se agregan en espacio logarítmico. Por eso son aditivas; solo el total se convierte a retorno convencional con exp(suma log) − 1.")

with tabs[3]:
    if rolling.empty:
        st.warning(f"No hay {window} observaciones completas para estimar la ventana rolling.")
    else:
        lr = rolling.iloc[-1]
        metric_row([("Beta IVV actual", f"{lr['Beta IVV']:.3f}", "Descriptivo, no predictivo."), ("Beta USD/BRL actual", f"{lr['Beta USD/BRL']:.3f}", "Descriptivo, no predictivo."), ("R² actual", pct(lr['R²']), "Ajuste de la regresión rolling."), ("Alpha semanal", pct(lr['Alpha']), "Intercepto semanal log.")])
        st.plotly_chart(line_chart(rolling, ["Beta IVV"], "Beta rolling de IVV", {"Beta IVV": BLUE}), use_container_width=True)
        st.plotly_chart(line_chart(rolling, ["Beta USD/BRL"], "Beta rolling de USD/BRL", {"Beta USD/BRL": ORANGE}), use_container_width=True)
        st.plotly_chart(line_chart(rolling, ["R²"], "R² rolling", {"R²": BLACK}), use_container_width=True)
        st.dataframe(rolling.tail(12).sort_index(ascending=False).style.format("{:.4f}"), use_container_width=True)
    st.subheader("Regresión global con errores HAC/Newey-West")
    global_table = pd.DataFrame({"Coeficiente": model.params, "Error estándar HAC": model.bse, "p-valor": model.pvalues})
    st.dataframe(global_table.style.format("{:.4f}"), use_container_width=True)
    st.caption("Los coeficientes rolling son descriptivos. El resumen global usa errores HAC/Newey-West con 4 rezagos.")
    st.subheader("Tracking difference estructural")
    tstats = tracking_statistics(td)
    metric_row([("Acumulado", pct(tstats['acumulado_simple_equivalente']), "Equivalente simple de la suma log."), ("Media anualizada", pct(tstats['media_anualizada']), "Media semanal log × 52."), ("Volatilidad anualizada", pct(tstats['volatilidad_anualizada']), "Desvío semanal × √52.")])
    td_frame = pd.DataFrame({"Semanal": td, "Acumulado log": td.cumsum()})
    st.plotly_chart(line_chart(td_frame, ["Semanal", "Acumulado log"], "Tracking difference: semanal y acumulado", {"Semanal": GREY, "Acumulado log": BLACK}), use_container_width=True)
    st.caption("Tracking difference = retorno log IVVB11 − retorno log IVV − retorno log USD/BRL. No representa exclusivamente costos del fondo.")
    st.subheader("Anomalías del residuo OLS")
    current_z = zscore.dropna().iloc[-1] if zscore.notna().any() else np.nan
    metric_row([("Z-score actual", f"{current_z:.2f}" if pd.notna(current_z) else "—", "Sin información futura."), ("Estado", anomaly_label(current_z), "Normal <2; inusual 2–3; extremo ≥3.")])
    st.plotly_chart(anomaly_chart(zscore), use_container_width=True)
    anomalies = pd.concat([historical_resid, zscore], axis=1).dropna()
    anomalies["Clasificación"] = anomalies["Z-score"].map(anomaly_label)
    st.dataframe(anomalies.assign(Abs=anomalies["Z-score"].abs()).nlargest(15,"Abs").drop(columns="Abs"), use_container_width=True)
    st.caption("Una anomalía indica una desviación estadística, no una oportunidad de compra ni un error seguro de precio.")

with tabs[4]:
    metrics = risk_return_metrics(prices["IVVB11"], rf_pct/100)
    metric_row([("Retorno total", pct(metrics['Retorno total']), "Del primer al último precio."), ("CAGR", pct(metrics['CAGR']), "Retorno anualizado compuesto."), ("Volatilidad", pct(metrics['Volatilidad anualizada']), "Semanal anualizada con √52."), ("Sharpe", f"{metrics['Sharpe anualizado']:.2f}", f"Tasa libre anual introducida: {rf_pct:.2f}%")])
    metric_row([("Máximo drawdown", pct(metrics['Máximo drawdown']), "Peor caída desde un máximo previo."), ("Drawdown actual", pct(metrics['Drawdown actual']), "Respecto del máximo histórico acumulado."), ("Mejor semana", pct(metrics['Mejor semana']), "Retorno simple semanal."), ("Peor semana", pct(metrics['Peor semana']), "Retorno simple semanal."), ("Semanas positivas", pct(metrics['Semanas positivas']), "Proporción de semanas > 0.")])
    growth = 10000 * prices["IVVB11"] / prices["IVVB11"].iloc[0]
    st.plotly_chart(line_chart(growth.rename("R$10.000"), ["R$10.000"], "Crecimiento hipotético de R$10.000", {"R$10.000": BLACK}, "R$"), use_container_width=True)
    st.plotly_chart(line_chart(drawdown_series(prices["IVVB11"]), ["Drawdown"], "Underwater (drawdown)", {"Drawdown": RED}, "Drawdown"), use_container_width=True)
    display = pd.DataFrame({"Métrica": list(metrics), "Valor": [pct(v) if k != "Sharpe anualizado" and k != "Valor final de R$10.000" else (f"{v:.2f}" if k == "Sharpe anualizado" else brl(v)) for k,v in metrics.items()]})
    st.dataframe(display, hide_index=True, use_container_width=True)
    st.caption(f"Sharpe calculado con tasa libre anual de {rf_pct:.2f}%, convertida a tasa semanal efectiva. Datos semanales: 52 periodos por año.")

with tabs[5]:
    st.write("Introduce tickers compatibles con Yahoo Finance y pesos porcentuales.")
    #edited = st.data_editor(pd.DataFrame({"Ticker":["IVVB11.SA","BIEF39.SA"], "Peso (%)":[98.69,1.31]}), num_rows="dynamic", use_container_width=True)
    edited = st.data_editor(pd.DataFrame({"Ticker":["BOVA11.SA","GOLD11.SA"], "Peso (%)":[70.0,30.0]}), num_rows="dynamic", use_container_width=True)
    if st.button("Calcular correlación", type="primary"):
        valid, errors = validate_portfolio(edited)
        if errors:
            for error in errors: st.error(error)
        else:
            try:
                tickers = valid["Ticker"].tolist()
                raw_assets = cached_download(tuple(tickers), str(start))
                invalid = raw_assets.columns[raw_assets.notna().sum() == 0].tolist()
                if invalid: raise ValueError("Tickers sin datos válidos: " + ", ".join(invalid))
                weekly_assets = raw_assets.resample(freq).last()
                weekly_counts = weekly_assets.notna().sum()
                insufficient = weekly_counts[weekly_counts < 2]
                if not insufficient.empty:
                    detail = ", ".join(f"{ticker}: {int(count)} semana(s)" for ticker, count in insufficient.items())
                    raise ValueError(
                        "Yahoo Finance no ofrece historial suficiente para calcular retornos de: "
                        f"{detail}. Se requieren al menos dos semanas con precio."
                    )
                asset_ret = weekly_assets.pct_change(fill_method=None)
                weights = valid.set_index("Ticker")["Peso (%)"] / 100
                port_ret = weighted_portfolio_returns(asset_ret, weights)
                ivv_ret = prices["IVVB11"].pct_change().rename("IVVB11")
                common = pd.concat([ivv_ret, port_ret], axis=1).dropna()
                if len(common) < max(10, window//2): raise ValueError(f"Solo hay {len(common)} observaciones comunes; son insuficientes.")
                corr = common.corr().iloc[0,1]
                metric_row([("Correlación IVVB11/cartera", f"{corr:.3f}", "Correlación contemporánea histórica."), ("Observaciones comunes", str(len(common)), "Semanas tras alinear todas las series.")])
                rcorr = common["IVVB11"].rolling(window, min_periods=max(10,window//2)).corr(common["Mi cartera"]).rename("Correlación rolling")
                st.plotly_chart(line_chart(rcorr, ["Correlación rolling"], "Correlación rolling", {"Correlación rolling": BLUE}), use_container_width=True)
                matrix_data = pd.concat([ivv_ret, asset_ret[tickers]], axis=1).dropna(how="any")
                fig = px.imshow(matrix_data.corr(), text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", title="Matriz de correlación")
                st.plotly_chart(fig, use_container_width=True)
                removed = len(pd.concat([ivv_ret, asset_ret[tickers]], axis=1)) - len(matrix_data)
                st.caption(f"La matriz utiliza {len(matrix_data)} semanas comunes; {removed} filas se excluyeron por datos ausentes. No se interpolaron precios.")
            except Exception as exc: st.error(f"No fue posible calcular la cartera: {exc}")
    else:
        st.caption("El resto del dashboard funciona aunque no se calcule una cartera.")

st.divider()
st.warning("Los datos de Yahoo Finance pueden presentar diferencias de horario, ajustes y disponibilidad. Este dashboard es una herramienta educativa de análisis histórico y no constituye recomendación de inversión.")
