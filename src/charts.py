from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE, ORANGE, GREY, BLACK, RED, GREEN = "#1F618D", "#E67E22", "#95A5A6", "#111111", "#C0392B", "#27864A"


def weekly_decomposition(prices: pd.Series, contrib: pd.DataFrame, weeks: int) -> go.Figure:
    c = contrib.tail(weeks)
    px = prices.reindex(c.index)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.08,
                        row_heights=[.42, .58], subplot_titles=("Precio semanal de IVVB11", "Contribuciones al retorno logarítmico"))
    precios_texto = [
        f"R$ {valor:,.2f}"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
        for valor in px
    ]
    fig.add_trace(go.Scatter(x=px.index, y=px, name="IVVB11 (price)", mode="lines+markers+text", line=dict(color=BLACK, width=2),
                             marker=dict(size=6), text=precios_texto, textposition="top center", textfont=dict(size=10, color="#333333"), cliponaxis=False, hovertemplate="%{x|%d/%m/%Y}<br>R$ %{y:,.2f}<extra></extra>"), row=1, col=1)
    labels = [("IVV", "Mercado estadounidense (IVV)", BLUE), ("USDBRL", "USD/BRL", ORANGE),
              ("Alpha y residual", "Residual, timing y tracking", GREY)]
    for col, label, color in labels:
        fig.add_trace(go.Bar(x=c.index, y=c[col]*100, name=label, marker_color=color,
                             hovertemplate="%{x|%d/%m/%Y}<br>%{y:+.3f} pp log<extra>"+label+"</extra>"), row=2, col=1)
    retornos_texto = [
        f"{valor:+.2f}%".replace(".", ",")
        for valor in c["Observado"] * 100
    ]
    fig.add_trace(go.Scatter(x=c.index, y=c["Observado"]*100, name="IVVB11 observado",
                             mode="lines+markers+text", line=dict(color=BLACK, width=1), marker=dict(size=7, color=BLACK,),
                             text=retornos_texto, texttemplate="%{text}", textposition="top center",textfont=dict(size=9,color=BLACK), cliponaxis=False,
                             hovertemplate="%{x|%d/%m/%Y}<br>%{y:+.2f} pp log<extra></extra>"), row=2, col=1)
    fig.update_layout(barmode="relative", hovermode="x unified", height=650, margin=dict(l=30,r=20,t=60,b=30), legend_orientation="h")
    fig.update_yaxes(title="Price (R$)", row=1, col=1); fig.update_yaxes(title="contribution to the weekly return % (pp log)", zeroline=True, row=2, col=1)
    return fig


def line_chart(frame, columns, title, colors=None, y_title=""):
    fig = go.Figure()
    data = frame.to_frame() if isinstance(frame, pd.Series) else frame
    for i, col in enumerate(columns):
        fig.add_trace(go.Scatter(x=data.index, y=data[col], name=col, mode="lines",
                                 line=dict(color=(colors or {}).get(col))))
    fig.update_layout(title=title, height=350, hovermode="x unified", margin=dict(l=25,r=15,t=55,b=25), yaxis_title=y_title)
    return fig


def anomaly_chart(z: pd.Series):
    fig = line_chart(z.to_frame(), [z.name], "Z-score rolling del residuo", {z.name: BLACK}, "Z-score")
    for level, color in [(2, ORANGE), (-2, ORANGE), (3, RED), (-3, RED)]:
        fig.add_hline(y=level, line_dash="dash", line_color=color)
    return fig


def accumulated_bar(values: dict[str, float]):
    colors = [BLACK, BLUE, ORANGE, GREY]
    fig = go.Figure(go.Bar(x=list(values), y=[v*100 for v in values.values()], marker_color=colors,
                           hovertemplate="%{x}<br>%{y:+.3f} pp log<extra></extra>"))
    fig.update_layout(title="Acumulación desde el máximo (espacio logarítmico)", yaxis_title="pp log", height=360)
    return fig

