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

# Example 

En el 10/07: el IVVB11 cayó −0,62 %, pero el mercado americano en realidad aportó +1,10 % (subió). Lo que arrastró el precio hacia abajo fue el real fortaleciéndose (−1,21 % de contribución cambiaria).
 Sin el gráfico dirías "el IVVB11 bajó, el mercado debe andar mal" — y estarías equivocado: el mercado subió, fue la moneda.

Dos límites que conviene tener presentes al usarlo:

*El componente gris (local/residuo) no es interpretable como una causa*. Cuando es grande, significa que ni el mercado ni el dólar explican bien ese movimiento — puede ser ruido, liquidez de B3 o desajustes de los datos. No lo leas como "efecto Brasil"; léelo como "esto el modelo no lo explica".

*Es atribución estadística, no causa comprobada*. El gráfico reparte el movimiento entre factores correlacionados; no prueba que el S&P causó nada. Para movimientos semanales normales es una guía razonable; para semanas muy ruidosas, tómalo como orientación, no como veredicto.

Con eso en mente, sí: el gráfico es una herramienta legítima para responder cada semana "¿esto vino del mercado americano o del dólar?", que era justo la pregunta con la que empezaste.

# Run
.\fin_env\Scripts\Activate.ps1  
python brent_ivvb11.py
