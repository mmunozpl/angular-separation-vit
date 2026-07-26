# Dos modelos idénticos, dos podas distintas

Implementación de referencia del preprint *Dos modelos idénticos, dos
podas distintas: la dirección dominante de $W_O$ bajo intervención
libre, suave y dura* (EN: *Same function, different pruning: the
dominant direction of $W_O$ under free, soft, and hard intervention*).

La dirección dominante de la proyección de salida de una cabeza de
atención, $v_1(W_O)$, se usa habitualmente para podar cabezas
redundantes, interpretar su papel o enrutar información. Este trabajo
muestra que esa dirección no es identificable —una libertad de gauge
del sector valor-salida la desplaza a voluntad sin tocar la función—,
que esa no-identificabilidad tiene una consecuencia medible (la
decisión de poda que produce cambia bajo gauge en más del 90 % de los
casos, tanto en un ViT afinado en visión como en un transformer de
lenguaje preentrenado, sin entrenar nada), y que el acoplamiento entre
esa geometría y la función es unidireccional: una sonda que la separa
angularmente hasta el símplex no mueve la función, pero imponer esa
misma separación por construcción sí la daña. Como instrumento
correcto se propone la firma de respuesta gauge-invariante
$v_1(C_h^P)$.

El código y las medidas se publican aquí; los CSV que respaldan cada
tabla y los checkpoints de reproducción se publican por separado en
Hugging Face (ver `## Datos y pesos` más abajo).

## Requisitos

- Python ≥ 3.10, PyTorch y [`timm`](https://github.com/huggingface/pytorch-image-models)
  para las columnas ViT-B/16 y ViT-L/16; [`transformers`](https://github.com/huggingface/transformers)
  para la columna de lenguaje (Pythia-410M).
- Una sola GPU NVIDIA (desarrollado en una RTX 5090, 32 GB); la
  columna de lenguaje es forma cerrada sobre pesos y corre en CPU.
- Los datasets (ImageNet-1k / ImageNet-100) se descargan a mano; las
  rutas se pasan por los configs YAML y nunca se descargan desde el
  código.

```bash
conda activate pytorch28
```

## Estructura

```
configs/   # configs YAML de las columnas ViT-B, ViT-L y DINOv2
src/
  codes/     # energía de Riesz, descenso proyectado, kissing number
  models/    # columna ViT, ViT de atención diversa (sonda blanda/dura)
  losses/    # margen angular y R_div
  data/      # loader de ImageNet
  metrics/   # diversidad y similitud funcional entre cabezas
  train/     # bucle de entrenamiento de la sonda
scripts/   # puntos de entrada (colas de entrenamiento, sondas de
           # gauge, decisión rota, ablación, órbita certificada)
tests/     # batería pytest
archivo/   # fases de verificación ya cerradas (de-riesgo, sanity,
           # smoke), conservadas como evidencia del proceso, no como
           # pipeline vigente
```

## Uso

```bash
# columna ancla ViT-B (n=5 semillas): base, blanda y dura
bash scripts/cola_base_clean.sh && bash scripts/cola_finde.sh

# columna ViT-L (n=3) y columna DINOv2 congelada (n=1)
bash scripts/cola_vitl.sh
bash scripts/cola_dinov2.sh

# decisión rota bajo gauge: visión (ViT-B) y lenguaje (Pythia-410M)
python scripts/decision_rota.py
python scripts/decision_rota_lm.py

# órbita de gauge certificada sobre los pesos reales
python scripts/orbita_gauge.py

# pruebas
pytest tests/ -q
```

## Datos y pesos

- Checkpoints de reproducción (pares base/blanda/dura por semilla y
  arquitectura): <https://huggingface.co/ManuelPla/angular-separation-vit-checkpoints>
- CSV que respaldan cada tabla y figura del paper: <https://huggingface.co/datasets/ManuelPla/angular-separation-vit-results>

## Licencia

Apache-2.0. Véase [LICENSE](LICENSE). Si utiliza este software, cítelo
como se indica en [CITATION.cff](CITATION.cff).
