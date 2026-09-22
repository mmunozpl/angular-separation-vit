# Prerregistro E2 — identidad de cabeza entre mitades de la sonda

Fijado el 21-09-2026 antes de ejecutar `scripts/identidad_mitades.py`.
Origen: GUIA_REVISION_R3 v2.1, §3c. Prueba gauge-invariante de
especificidad por cabeza, sin v1(W_O).

## Datos

- Firmas r_h^(A), r_h^(B) de las dos mitades de la sonda a P=1000:
  (primaria) mitades balanceadas por clase en orden rotado, con el
  apareamiento de `curva_sonda.particion_por_clase`; (robustez)
  mitades por bloques de clases de la sonda publicada
  (`imagenet100_val_1k.pt`, split posicional i <-> i+500).
- B2 no guardó los vectores de firma por mitad, solo sus agregados:
  se recomputan con el mismo código (`curva_sonda.grams` y
  `radio_firma.lectura_capa`), sin cambios.
- Cinco semillas del ancla, 12 capas. Extensión a P en
  {250, 500, 1000, 2000, 5000} (rejilla B2, orden rotado) como curva
  secundaria.

## Estadístico

- Por capa: M_hh' = |<r_h^(A), r_h'^(B)>| y tasa de identificación
  id = (1/H)·#{h : argmax_h' M_hh' = h}.
- Secundario: tasa con asignación óptima (húngaro,
  `scipy.optimize.linear_sum_assignment` sobre −M).

## Nulo

1000 permutaciones de las etiquetas de la mitad B por capa
(`torch.Generator` semilla 0); esperanza bajo el nulo 1/H = 1/12; se
reporta el percentil 99 del nulo.

## Veredicto (fijado ahora)

La firma es específica por cabeza si la tasa media sobre capas supera
el percentil 99 del nulo en las cinco semillas; se reporta además por
capa cuántas alcanzan id >= 10/12.

## Expectativa

Se cumple en las cinco semillas; las capas con discrepancia entre
mitades en B2 (4 de 12 a P=1000) son las de menor id. Si no se
cumple, se reporta como expectativa fallida.

## Salida

`artifacts/logs/identidad_mitades/identidad.csv` (semilla, capa, P,
particion, id, id_hungaro, p99_nulo) y `resumen.csv`; 15 filas
aleatorias al terminar.
