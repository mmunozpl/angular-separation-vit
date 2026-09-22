# Changelog

## 2026-09-21

- `configs/attn_*_clean.yaml`: retirada la clave `warmup_epochs: 3`.
  Ningún código la consumía: el bucle de entrenamiento usa
  `CosineAnnealingLR` sin calentamiento, y la tasa de aprendizaje de
  la época 1 registrada en `run.csv` lo confirma. Los resultados
  publicados no cambian.
- `scripts/punto_balanceado.py`: `test_familia_alineada` (forma
  canónica QR y familia diagonal alineada; 10 instancias).
- Nuevos: `scripts/gauge_dR.py` (E0), `scripts/poda_nulo.py` (E1),
  `scripts/identidad_mitades.py` (E2), `paper/refs.bib`, `prereg/`.

## 2026-09-22 (v7.0.1)

- Declaración sobre el uso de herramientas de IA: redacción fijada por
  el autor en los dos idiomas; PDF de `paper/` regenerados. Sin cambio
  de código ni de resultados.
