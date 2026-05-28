# Nota de trabajo futuro — diversidad de atención vía Q·K

Registro de una línea de investigación abierta por los resultados de
la contribución A. No es para implementar ahora: es el núcleo de un
posible segundo trabajo, anotado para que no se pierda.

## El hallazgo que la motiva

La contribución A regulariza la dirección representativa de cada
cabeza ---el primer vector singular de la proyección de salida
$W_O^{(h)}$--- hacia la separación angular máxima. El regularizador
blando alcanza el óptimo del símplex sin signo (85°) sin coste de
exactitud (92,6% vs 92,5% de la base). Pero la evaluación de
diversidad funcional ---similitud entre mapas de atención--- muestra
un efecto marginal: la similitud media baja solo de 0,693 a 0,688
(−0,6% relativo), y la reducción se concentra casi entera en la última
capa (capa 11: −0,035), mientras varias capas intermedias incluso
suben.

Conclusión: la geometría de las direcciones de salida $W_O$ y la
diversidad funcional de la atención están **desacopladas**. Separar
$W_O$ angularmente apenas cambia a qué atiende cada cabeza.

## La hipótesis para el trabajo futuro

La función de una cabeza ---a qué atiende--- no reside en $W_O$, que
solo proyecta de vuelta el resultado, sino en el mecanismo que decide
la atención: $\mathrm{softmax}(QK^\top)$. Regularizar $W_O$ no toca ese
mecanismo, de ahí el desacople. La diversidad funcional, si se quiere
inducir geométricamente, habría que buscarla en el subespacio $Q\cdot
K$.

## Por qué es un proyecto independiente, no una extensión

- $Q\cdot K$ no es un conjunto de direcciones sobre una esfera, sino
  una forma bilineal: un operador que define una geometría de
  comparación entre tokens. "Separación angular máxima" sobre formas
  bilineales no tiene definición obvia y habría que construirla.
- Exige un estudio previo de atribución por componente: regularizar
  $Q$, $K$, $V$ y $W_O$ por separado y medir cuál mueve realmente los
  mapas de atención. Eso es, en sí mismo, un estudio de análisis.
- Cambia el objeto regularizado, la definición de separación, la
  implementación y la evaluación. Es un marco nuevo, no una variante
  del actual.

## Cómo entra en el paper actual

Solo como una frase de trabajo futuro en la sección de discusión o
conclusión, del tenor: "El desacople observado entre la geometría de
$W_O$ y la diversidad funcional sugiere que el componente que gobierna
la atención reside en el subespacio $Q\cdot K$; inducir separación
sobre ese subespacio, y el estudio de atribución por componente que lo
precede, quedan como continuación natural."

No se implementa nada de esto para el preprint actual.
