# Dashboard histórico de IVVB11

Dashboard local, educativo y descriptivo construido con Streamlit y Plotly. Descompone retornos semanales logarítmicos de IVVB11 entre IVV, USD/BRL y alpha/residuo; incluye drawdowns, regresiones rolling, tracking difference, anomalías, métricas de riesgo/retorno y correlación con una cartera manual.

## Instalación (PowerShell)

```powershell
cd "C:\Users\XIMENA\Documents\S&P500_analysis"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Ejecución

```powershell
.\fin_env\Scripts\Activate.ps1

streamlit run app.py
```

Para las pruebas:

```powershell
python -m pytest -q
```

## Decisiones estadísticas

- Se usan cierres ajustados (`auto_adjust=True`) y cierre semanal `W-FRI`. No se interpolan precios. Tras el resampleo se conservan únicamente las semanas coincidentes y se reportan las eliminadas.
- La regresión es `r_IVVB11 = alpha + beta_IVV*r_IVV + beta_FX*r_USDBRL + residuo`. Las contribuciones `beta*r` más `alpha + residuo` reconstruyen exactamente el retorno log observado. Los coeficientes rolling son descriptivos; el resumen global reporta errores HAC/Newey-West con cuatro rezagos.
- El tracking difference estructural, separado del residuo OLS, es `r_IVVB11-r_IVV-r_USDBRL`. No debe interpretarse exclusivamente como gastos.
- Los acumulados se suman en log-retornos. El retorno convencional se muestra como `exp(suma log)-1`.
- El z-score usa media y desviación rolling desplazadas una semana; por tanto, una observación nunca participa en su propia normalización. Normal: `|z|<2`; inusual: `2≤|z|<3`; extremo: `|z|≥3`.
- Episodios: drawdown menor a 1% = sin drawdown relevante; residual ≥50% de la magnitud absoluta total = predominantemente residual; factores con signos opuestos = efectos opuestos; un factor con ≥60% de la magnitud de IVV+FX = dominante; en otro caso = caída conjunta.
- Las métricas semanales se anualizan con 52. La tasa libre anual es introducida por el usuario y se convierte a semanal efectiva para el Sharpe.

## Origen y límites

`brent_ivvb11.py` se conserva como referencia original. Su regresión y visualización de dos paneles se reutilizaron conceptualmente. La etiqueta “local” se reemplazó porque el término reunía intercepto, residuo, timing y tracking; Matplotlib se migró a Plotly. Yahoo Finance puede tener diferencias de horarios, ajustes, feriados y disponibilidad. La herramienta no predice precios ni emite recomendaciones.
