# Separación angular máxima en Transformers de visión

Implementación de referencia del preprint *Separación angular máxima en
Transformers de visión: códigos esféricos para una atención diversa y
una detección generalizable de medios sintéticos*.

El repositorio implementa y valida tres piezas:

1. **Códigos esféricos** — descenso de gradiente proyectado sobre la
   energía de Riesz en `S^{d-1}`, validado contra el *kissing number*
   en `d ∈ {2, 3, 4, 8, 24}`.
2. **Contribución A** — regularizador de atención angularmente diversa
   sobre las direcciones representativas de las cabezas de un ViT.
3. **Contribución B** — cabeza de prototipos por generador anclada a un
   código esférico, con rechazo en conjunto abierto, para la detección
   de imagen sintética sobre GenImage en régimen
   *leave-one-generator-out*.

## Requisitos

- Python ≥ 3.10, PyTorch y [`timm`](https://github.com/huggingface/pytorch-image-models)
  para la columna ViT-B/16.
- Una sola GPU NVIDIA (desarrollado en una RTX 5090, 32 GB).
- Los datasets (ImageNet-1k / ImageNet-100, GenImage) se descargan a
  mano; las rutas se pasan por los configs YAML y nunca se descargan
  desde el código.

```bash
conda activate pytorch28
```

## Estructura

```
configs/   # configs YAML de códigos, contribución A y B
src/
  codes/   # energía de Riesz, descenso proyectado, kissing number
  models/  # columna ViT, ViT de atención diversa, cabeza de protos
  losses/  # margen angular y R_div
  data/    # loaders de ImageNet y GenImage (split LOGO)
  metrics/ # diversidad de atención, conjunto abierto (AUROC, OSCR)
  train/   # bucles de entrenamiento de A y B
scripts/   # puntos de entrada (gen_codes, run_attn, run_detect, …)
tests/     # batería pytest
```

## Uso

```bash
# códigos esféricos
python scripts/gen_codes.py --config configs/codes.yaml

# contribución A — atención diversa
python scripts/run_attn.py --config configs/attn_imagenet.yaml

# contribución B — detección en conjunto abierto (LOGO)
python scripts/run_detect.py --config configs/detect_genimage.yaml

# pruebas
pytest tests/ -q
```

## Licencia

Apache-2.0. Véase [LICENSE](LICENSE). Si utiliza este software, cítelo
como se indica en [CITATION.cff](CITATION.cff).
