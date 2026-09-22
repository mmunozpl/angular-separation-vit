# Preregistro B1 — curva de inestabilidad decisional vs fuerza del gauge

Congelado 26-08-2026, antes de leer número alguno (R1 bloque B).
~~NO CORRE hasta la ratificación del punto 2~~ — **RATIFICADO
28-08-2026, umbrales tal cual: candado levantado, el barrido
puede correr**. Origen: Revisor B §3, con la corrección de expectativa
escrita por delante.

## Predicción preregistrada — contra la expectativa del revisor

El R0 §2 estableció que el radio local NO es cero: con margen
positivo, Wedin garantiza que la decisión sobrevive a gauges
próximos a la clase inocua. Por tanto:

1. flip ≈ 0 en δ_R = 0,06 — **la estabilidad en régimen pequeño es
   el resultado esperado** y armoniza con la dicotomía (radio local
   positivo, radio funcional nulo por solidez de órbita);
2. crecimiento monótono del flip con δ_R;
3. δ* = primera fuerza con flip ≥ 40 % (umbral heredado de
   tab:decision, fijado antes de este dato) se reporta como
   «arranque de inestabilidad».

## Objetos

Las 420 combinaciones capa×semilla×fuerza de la fase G: 5 semillas
× 12 capas × 7 fuerzas, R reproducibles por semilla
`1000·capa + j` (patrón de `decision_rota.py`). Decisiones por
v₁(W_O) —par más redundante y solape top-3— materializadas a cada
fuerza. Estático, CPU, cero forwards.

## Estadísticos (fijados)

Por fuerza: fracción de combinaciones con cambio de par respecto a
la decisión sin gauge, y solape medio top-3, ambos con banda entre
semillas (media ± std sobre las 5).

## Gates previos al veredicto

~~Reproducción exacta: a la fuerza saturada, las decisiones
materializadas clavan las almacenadas en `decision_rota.csv`
(seed 42).~~ **Tachado 27-08, pre-dato:** presuponía identidad de
esquemas de R no verificada. Verificada sobre el árbol
(27-08-2026): generador compartido (`src/gauge_flip.py:18`,
`r = randn(semilla) + escala_id·I`), pero **fase G**
(`run_fase_G.py:163`) siembra `1000·capa+j` con j = índice de
fuerza (ESCALAS = [128,64,32,16,8,4,2]; una R por capa×fuerza,
compartida entre las 5 semillas de modelo), y **decision_rota**
(`decision_rota.py:125`) siembra `1000·capa+g` con g = réplica
0..4, todas a escala 8.0. Solape exacto: una R por capa
(`semilla 1000·capa+4, escala 8.0` = réplica g=4). El gate
enmendado, satisfacible por construcción:

1. **identidad exacta en la decisión base sin gauge** por
   (semilla, capa) contra la almacenada en `decision_rota.csv`;
2. **identidad exacta en el ancla de solape**: en el punto de la
   curva a escala 8.0, la decisión materializada con la R
   compartida clava la fila **gauge_idx 5** almacenada (el bucle
   de `decision_rota.py` guarda base=0 y réplicas g+1; semilla
   +4 → idx 5; verificado sobre el árbol 27-08), por
   (semilla, capa).

~~3. consistencia con el 92,7 % en banda binomial.~~ **Tachado
27-08, pre-dato: redundante dado el gate 2** — si el 2 pasa, el
punto saturado queda determinado por archivo. **Reencuadre como
sanity de archivo, ejecutado hoy sin correr nada:** flip por
réplica (criterio pesos) = 0,917 / 0,900 / 0,900 / 0,917 /
**1,000**; la rebanada del ancla (idx 5) es la más extrema de las
cinco: 60/60. P(60/60 | p=0,927) ≈ 0,011; con cinco rebanadas y
dependencia estructural entre ellas, atípica pero no
descalificante. Se reporta tal cual.

**Nota de presentación (pre-dato, obligada en prosa o caption):**
el punto saturado de la curva dirá **1,000 (60/60), no 92,7 %** —
protocolos distintos: la curva usa una R por capa (esquema fase G);
tab:decision agrega cinco réplicas por capa. La figura lleva el
punto de referencia de tab:decision a escala 8.0 (0,927) con su
protocolo declarado, o el caption explica la diferencia; un árbitro
yuxtapondrá ambos números y el texto debe habérselo dicho antes.

Si 1 o 2 fallan, el script aborta sin imprimir estadístico; si 3
falla, se aborta y la discrepancia va a bitácora antes de tocar
nada. El barrido se reejecuta, no se recuerda (regla de la casa).

## Veredictos — RATIFICADOS 28-08-2026 (punto 2)

- «seguridad de régimen pequeño» si flip < 5 % en δ_R ≤ 0,13;
- «arranque de inestabilidad» en δ*;
- si la curva no es monótona, se reporta tal cual, sin adorno.

## Destino en prosa

Párrafo en §7.2 + columna o figura junto a tab:gauge. Cualifica «la
decisión se rompe» por fuerza. Las cifras titulares (δ*) se
bloquean en el oráculo tras la ratificación del texto.
