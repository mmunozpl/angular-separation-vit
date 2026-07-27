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

- Python 3.11; PyTorch y [`timm`](https://github.com/huggingface/pytorch-image-models)
  para las columnas ViT-B/16 y ViT-L/16, [`transformers`](https://github.com/huggingface/transformers)
  para la columna de lenguaje (Pythia-410M) — versiones exactas
  fijadas en [`pyproject.toml`](pyproject.toml)/[`uv.lock`](uv.lock),
  congeladas desde el entorno que produjo cada resultado del paper.
- Una sola GPU NVIDIA (desarrollado en una RTX 5090, 32 GB, CUDA
  13.0); la columna de lenguaje es forma cerrada sobre pesos y corre
  en CPU.
- Los datasets (ImageNet-1k / ImageNet-100) se descargan a mano; las
  rutas se pasan por los configs YAML y nunca se descargan desde el
  código.

```bash
conda activate pytorch28
uv pip install -r <(uv export --no-hashes --no-dev)
```

(`uv sync` se evita a propósito: `pytorch28` es un entorno conda
compartido con otros proyectos, y `uv sync` eliminaría cualquier
paquete no declarado en el `pyproject.toml` de este repositorio. El
comando anterior instala las versiones fijadas de forma aditiva.)

## Estructura

```
configs/
  codes.yaml                     # generación de códigos esféricos (Riesz)
  attn_vitb_{base,blanda,dura}_clean.yaml    # columna ViT-B, 3 variantes
  attn_vitl_{base,blanda,dura}_clean.yaml    # columna ViT-L, 3 variantes
  attn_dinov2_base_clean.yaml    # columna DINOv2 congelada
  imagenet100_cmc.txt            # lista canónica de 100 sinsets (CMC)
src/
  seed.py             # set_seed único, reproducibilidad
  config.py            # cargador de configs YAML
  carga.py              # loaders compartidos: base afinada + probe congelado
  firma_funcional.py    # firma de respuesta gauge-invariante v1(C_h^P)
  gauge_flip.py         # transformación de gauge valor-salida (Fase G)
  reg_funcional.py      # R_func: la sonda angular, blanda y dura
  codes/
    riesz.py       # energía de Riesz y su gradiente
    generate.py     # descenso de gradiente proyectado en S^{d-1}
    canonical.py     # inicializaciones canónicas para kissing numbers
    validate.py      # theta_min contra la cota del kissing number
  models/
    vit_backbone.py   # columna ViT-B/L/DINOv2, ganchos de atención
    attn_diverse.py    # AttnDiverseViT: sonda angular blanda/dura
  losses/
    angular.py    # R_div (ec. 5) y margen angular
  data/
    imagenet.py    # loader de ImageNet-100/1k
  metrics/
    attention.py    # redundancia, entropía, dirección representativa
  train/
    train_attn.py    # bucle de entrenamiento de la sonda
  viz/
    sphere.py    # proyecciones 2D/3D de códigos esféricos
scripts/   # un punto de entrada por resultado del paper — ver Uso
tests/     # batería pytest (atención, códigos, losses, SVD fallback/GPU)
archivo/   # vías cerradas conservadas como evidencia del proceso, no
           # como pipeline vigente: la contribución de detección
           # descartada, borradores de paper superados, src/scripts/
           # configs huérfanos de iteraciones anteriores, la autopsia
           # del crash de SVD, y otras sondas y gates ya cerrados
```

## Uso

```bash
# Fase 1 — códigos esféricos: generar, cachear, validar contra kissing number
python scripts/gen_codes.py --config configs/codes.yaml

# columna ancla ViT-B (n=5 semillas): base, blanda y dura
bash scripts/cola_base_clean.sh && bash scripts/cola_finde.sh

# columna ViT-L (n=3) y columna DINOv2 congelada (n=1)
bash scripts/cola_vitl.sh
bash scripts/cola_dinov2.sh

# tabla de ablación (base/blanda/dura, val_top1/theta_min/redundancia) — tab:divattn
python scripts/ablation_table_A.py

# probe set congelado de 1000 imágenes, compartido por los diagnósticos siguientes
python scripts/build_probe_set.py

# Fase G — invariancia gauge-flip: v1(W_O) deriva, el circuito OV no — tab:gauge
python scripts/run_fase_G.py

# Fase 0 — firma computada vs. estática por capa — tab:residuo
python scripts/run_fase_0.py

# decisión rota bajo gauge: visión (ViT-B) y lenguaje (Pythia-410M) — tab:decision
python scripts/decision_rota.py
python scripts/decision_rota_lm.py

# órbita de gauge certificada sobre los pesos reales — app:orbita
python scripts/orbita_gauge.py

# reubicación hacia Q·K con la profundidad — tab:inversion, fig:inversion
python scripts/dissociation_D.py
python scripts/principal_angles.py
python scripts/fig_inversion.py

# poda por criterio — pesos vs. firma vs. suelo aleatorio — tab:poda
python scripts/poda_criterio.py

# lecturas en frío: inercia de la sonda blanda y coste de la dura (§6.2, §6.3)
python scripts/inertia_read.py
python scripts/dura_cost_read.py

# pruebas
pytest tests/ -q
```

## Datos y pesos

- Checkpoints de reproducción (pares base/blanda/dura por semilla y
  arquitectura): <https://huggingface.co/ManuelPla/angular-separation-vit-checkpoints>
  (DOI: [10.57967/hf/9742](https://doi.org/10.57967/hf/9742))
- CSV que respaldan cada tabla y figura del paper: <https://huggingface.co/datasets/ManuelPla/angular-separation-vit-results>
  (DOI: [10.57967/hf/9743](https://doi.org/10.57967/hf/9743))
- Código (este repositorio): DOI pendiente, se archiva en Zenodo con
  el próximo release etiquetado.

## Licencia

Apache-2.0. Véase [LICENSE](LICENSE). Si utiliza este software, cítelo
como se indica en [CITATION.cff](CITATION.cff).
