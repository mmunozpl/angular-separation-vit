# Preregistro B2 — estabilidad de la firma frente al tamaño de sonda

Congelado 26-08-2026, antes de leer número alguno (R1 bloque B).
~~NO CORRE hasta la ratificación del punto 3~~ — **RATIFICADO
28-08-2026, umbrales y resolución (a) tal cual: candado
levantado**. Origen: Revisor B §6.

## Diseño — sondas anidadas (decisión preregistrada)

P ∈ {250, 500, 1000, 2000, 5000}, congeladas y **anidadas**: cada P
es prefijo del siguiente bajo permutación fija.

- 1000 = la sonda existente (`imagenet100_val_1k.pt`);
- 250 y 500 = prefijos de la de 1000 bajo una permutación fijada por
  semilla escrita en el script antes de correr;
- 2000 y 5000 = extensiones estratificadas (20 y 50 por clase) con
  semilla fija, sin tocar las 1000 primeras.

El anidado aísla el efecto tamaño del efecto remuestreo. El
split-half dentro de cada P aparea la posición i con i+P/2 (el
mismo patrón posicional del half-swap de `radio_firma.py`).

**Esquema de estratificación a P=250, declarado antes de correr
(nota de diseño 27-08, pre-dato):** la permutación fija de la sonda
de 1000 es la **rotación por clases** (round-robin: las posiciones
ciclan por las 100 clases, orden intra-clase fijado con semilla
2026 escrita aquí). Así todo prefijo hereda cuasi-estratificación
por construcción: a P=250, 2 imágenes por clase en cincuenta
clases y 3 en las otras cincuenta —determinado por la semilla, no
decidido dentro del script—; a P=500, exactamente 5 por clase.
Las extensiones a 2000 y 5000 siguen el mismo orden rotado.

## Resolución del conflicto orden/gate — fechada 27-08, pre-dato

**Objeto leído sobre el árbol (27-08):** `imagenet100_val_1k.pt`
guarda paths+labels en **bloques puros por clase** (10 por clase,
99 cambios consecutivos) y sus mitades almacenadas son **disjuntas
en clases** (0–49 | 50–99, intersección cero). Consecuencias: el
8/12 publicado comparó mitades de clases disjuntas, y el ‖E‖ del
half-swap apareó cada imagen con una de otra clase (+50). La vía
(b) —rotar dentro de las mitades almacenadas— queda **descartada
por el dato**: mitades disjuntas jamás dan prefijos estratificados.

**Resolución adoptada: (a) enriquecida.**

1. **El orden almacenado corre solo para el gate de
   reproducción** a P=1000: split-half posicional i↔i+500 tal
   como se publicó (8/12, mediana 0,001). Nada de la curva usa
   este orden.
2. **Toda la curva usa el orden rotado** (round-robin por clases,
   orden intra-clase a semilla 2026): mitades balanceadas por
   construcción a todo P.
3. **Semántica uniforme por clase, no posicional:** bajo rotación
   el desplazamiento i↔i+P/2 cambia de relación de clase según P
   (a P=1000 aparea dentro de clase; a P=500, entre clases +50).
   Para que la curva mida lo mismo a todo P, split-half y
   apareamiento de diferencias se definen **dentro de cada
   clase** (primeras ⌊m/2⌋ ocurrencias contra últimas, orden
   rotado; con m impar la sobrante queda fuera del apareamiento).
4. **El par P=1000 (almacenado vs rotado) se reporta como
   contraste composición/tamaño:** mismas imágenes, mismas
   firmas base; la diferencia entre coincidencia con mitades
   disjuntas y con mitades balanceadas separa el efecto
   composición de clases del efecto tamaño. Expectativa:
   balanceada ≥ 8/12; se reporta tal cual salga.

**Observación para la lectura del autor (fuera del alcance R1, no
se edita nada):** la disjunción de clases de las mitades re-lee el
8/12 y el ‖E‖ publicados —«la perturbación más gruesa de su
clase» es literalmente cierta y ahora está cuantificada en su
génesis—; si el autor quiere una cláusula aclaratoria en el
veredicto del lem:radio o en Límites, es decisión de lectura.

## Métricas por P (ancla, 5 semillas; fijadas)

1. coincidencia split-half del par más redundante por firma
   (12 capas);
2. ídem del conjunto de poda top-3;
3. mediana de r_h/‖E‖ del lem:radio;
4. recuento fuera de dominio (‖E‖ ≥ hueco).

## Gates previos al veredicto

- Invariancia de escala de r/‖E‖ bajo C → 10C (heredado de
  `radio_firma.py`);
- a P=1000 **con el orden almacenado y su split posicional
  i↔i+500**, reproducción del 8/12 y de la mediana 0,001
  publicados; si no reproducen, el script aborta sin imprimir
  estadístico (la curva, en cambio, corre bajo el esquema por
  clase de la resolución (a)).

## Fe de parámetros y criterios — 28-08-2026

Tres piezas que el preregistro no fijó por escrito y que el script sí
fijó antes de correr. Se escriben aquí el 28-08, **después de la
corrida**, con la cronología dicha tal cual: la elección es
pre-dato (vive en el código desde antes del barrido), la redacción
es post-dato.

1. **`GATE_TOL = 0,02`** — tolerancia relativa con que el gate 2
   compara la mediana contra la publicada. Es parámetro **de gate,
   no de veredicto**: no entra en ninguna cifra que el paper afirme.
   En la corrida no soportó peso alguno: el acuerdo resultó exacto
   (0,00105 contra 0,00105, desvío ~1e-12 por cabeza).
2. **Lectura de «12/12»** — el script exige `par_m == 12,0`, esto es
   **las cinco semillas coincidiendo en las doce capas**. Es la
   lectura estricta de un enunciado ambiguo. No altera el desenlace:
   ninguna lectura ---estricta, por media o por semilla--- alcanza
   el 12/12 en la rejilla, de modo que se reporta techo.
3. **Conducta ante fallo del gate exacto de 8/12** — se deja exacto
   tal como se preregistró; si aborta, se investiga (recorrida en
   CPU, comparación capa a capa) y la discrepancia va a bitácora.
   **No se relaja sobre la marcha.** Ocurrió: la primera pasada
   abortó por la mediana (0,00108 contra 0,00105) y la
   investigación descartó jitter de GPU ---los ratios por cabeza
   coincidían a 8e-13 y `radio_firma.py` reproducía su CSV bit a
   bit--- y localizó un defecto del script nuevo: con 144 valores
   tomaba el central superior en vez de promediar los dos
   centrales. Corregida la mediana, el gate clava 8/12 y 0,00105.
   El gate cumplió su función: cazó un error de implementación
   antes de que produjera un veredicto.

## Veredictos — RATIFICADOS 28-08-2026 (punto 3)

- se reporta el P donde la coincidencia alcanza 12/12, o el techo
  alcanzado a P=5000 si no llega;
- expectativa de monotonía creciente, reportada tal cual si no se
  cumple;
- los 8/12 de P=1000 pasan de caveat a punto de una curva
  caracterizada.

## Coste y destino

Forwards de gram por P y semilla, segundos cada uno; sin
entrenamiento. Extiende el párrafo del veredicto del lem:radio y
actualiza la frase de Límites (de consejo a curva medida). Las
cifras titulares (techo de coincidencia) se bloquean en el oráculo
tras la ratificación del texto.
