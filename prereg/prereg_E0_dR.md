# Prerregistro E0 — d(R) de la ecuación (2) sobre los gauges muestreados

Fijado el 21-09-2026 antes de ejecutar `scripts/gauge_dR.py`.
Origen: GUIA_REVISION_R3 v2.1, §3b. Postprocesado, sin entrenar ni
pasar imágenes.

## Diseño

- Se reproduce cada R muestreada de la fase G desde (capa, escala_id)
  con el mismo generador que `src/gauge_flip.aplica_gauge_ov`:
  `torch.Generator("cpu").manual_seed(1000*capa + j)`, una R por
  cabeza extraída en el mismo orden, r = randn + escala_id·I, en
  float64 y con d_h = 64. La R no depende del checkpoint.
- Gate: la delta_R recomputada (desviación media respecto al mejor
  múltiplo escalar de la identidad, misma fórmula) coincide con la
  columna `desv_R` de `gauge_flip.csv` a 1e-6, fila a fila.
- d(R) = (sum_i (sigma_i − sigma_media)^2)^{1/2} / (sum_i
  sigma_i^2)^{1/2}, por SVD de cada R; se agrega como media sobre
  cabezas, igual que delta_R.
- Lo mismo sobre los gauges de `decision_rota.csv` (escala 8,0,
  semillas 1000*capa+g, g=0..4) y `decision_rota_lm.csv` (d_h de
  Pythia-410M; se lee de su script).

## Salida

`artifacts/logs/fase_G/gauge_dR.csv` (arch, capa, semilla, escala_id,
delta_R, d_R); 15 filas aleatorias al terminar. `tab:gauge` recibe la
columna d(R) con media ± desviación por fuerza; delta_R sigue siendo
el mando (B1 está prerregistrado sobre él).
