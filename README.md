# oil_sp500_corr
Exploring macroeconomic asset correlations (Brent Oil vs. S&amp;P 500) using Python and yfinance.


El punto negro es cuánto subió o bajó el IVVB11 esa semana. Las barras de colores te dicen de dónde vino ese movimiento:

🔵 Azul = bolsa americana (S&P 500)
🟠 Naranja = dólar/real
⬜ Gris = local/ruido (lo que no explican los otros dos)

Las barras se suman hasta el punto negro. Así, de un vistazo, ves quién mandó cada semana.

# Interpretation

La barra azul hacia abajo significa que el precio del mercado americano bajó. No te dice cuántas acciones cambiaron de manos.

La clave está en separar dos cosas que no van juntas:

*Cuántas acciones se vendieron* = eso es el *volumen* (cantidad).
*Cuánto cayó el precio* = eso es lo que muestra la barra azul (dirección/magnitud del movimiento).

Y por qué no son lo mismo: cada acción vendida es a la vez una acción comprada — no puede venderse sin que alguien la compre. Así que el número de acciones "vendidas" siempre es igual al número "comprado". Ese número (el volumen) puede ser enorme o diminuto, y en ambos casos el precio puede subir, bajar o quedarse quieto.

Ejemplos para que quede claro:

Precio cae fuerte con *pocas* acciones negociadas (poca gente, pero los vendedores aceptan precios bajos).
Precio cae fuerte con *muchas* acciones negociadas.
Precio casi no se mueve aunque se negocien *muchísimas* acciones.

La barra azul no distingue entre esos casos. Solo dice "el precio bajó tanto por ciento".

Si tu pregunta real es "¿se negociaron muchas acciones esta semana?", eso necesita el dato de volumen, que es una columna aparte y no está en este gráfico. Podría añadírtelo si te interesa verlo junto al resto.