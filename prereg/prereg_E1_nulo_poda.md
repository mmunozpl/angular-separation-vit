# Prerregistro E1 — nulo aleatorio de la poda (`tab:poda`)

Fijado el 21-09-2026 antes de ejecutar `scripts/poda_nulo.py`.
Origen: GUIA_REVISION_R3 v2.1, §3. Nada de lo de abajo se ajusta
después de leer un número.

## Diseño

- Mismo presupuesto que `tab:poda`: k=2 cabezas por capa, 24 de 144.
- Mismo mecanismo: se anulan las columnas de la cabeza en la
  proyección de salida (`scripts/poda_criterio.podar`).
- Cinco semillas base del ancla (42–46); val completo de
  ImageNet-100 (5000 imágenes).
- Sorteos: N=100 por semilla, `random.Random(s)` con semillas de
  sorteo s=0,…,99 (el generador de `seleccion_aleatoria`, sin
  cambios); los sorteos 42, 43 y 44 del protocolo anterior se
  reproducen como fila de regresión.

## Gate de reproducción (antes de leer nada más)

La media de las caídas de los sorteos 42/43/44 por semilla, agregada
sobre las cinco semillas, coincide con la celda «Aleatorio» de
`tab:poda` con tolerancia 1e-3 en top-1 (0,001 = 0,1 pp).

## Estadísticos

- Por semilla: media y desviación de la caída (pp) sobre los 100
  sorteos; percentiles 2,5, 5, 50, 95 y 97,5.
- Agregados sobre las cinco semillas.
- Para pesos y firma: percentil que ocupa su caída dentro de la
  distribución conjunta de los 500 sorteos.

## Criterio de veredicto (fijado ahora)

- Un criterio «bate al azar» si su caída queda por debajo del
  percentil 5 del nulo en al menos 4 de 5 semillas.
- «Daña más que el azar» si queda por encima del percentil 95 en al
  menos 4 de 5 semillas.

## Expectativa

Ninguno bate al azar; la firma queda por encima del percentil 95 en
la mayoría de las semillas. Si no se cumple, se reporta como
expectativa fallida, sin reajustar umbrales.

## Salida

`artifacts/logs/poda_criterio/poda_nulo_100.csv` (semilla, sorteo,
top1_podado, caida_pp) y `poda_nulo_resumen.csv`; 15 filas aleatorias
al terminar.
