# [Preprint] La dirección dominante de $W_O$ no es identificable: órbita de gauge, radio certificado nulo y consecuencias para la poda

🇪🇸 Español · 🇬🇧 [English](README.md)

**Manuel Muñoz Plá** · [ORCID 0009-0000-5714-912X](https://orcid.org/0009-0000-5714-912X)

[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.21630534-009e73)](https://doi.org/10.5281/zenodo.21630534)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Dataset-ffd21e)](https://huggingface.co/datasets/ManPla/angular-separation-vit-results)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Model-ffd21e)](https://huggingface.co/ManPla/angular-separation-vit-checkpoints)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Space-ffd21e)](https://huggingface.co/spaces/ManPla/angular-separation-vit-demo)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0000--5714--912X-a6ce39)](https://orcid.org/0009-0000-5714-912X)
[![Web](https://img.shields.io/badge/Web-manpla.net-009e73)](https://manpla.net)
[![License](https://img.shields.io/badge/License-Apache--2.0-009e73)](LICENSE)
[![Cite](https://img.shields.io/badge/Cite-BibTeX-009e73)](#cómo-citar)

**Resumen:** La factorización valor-salida de una cabeza de atención no es única: para toda $R\in GL(d_h)$, la sustitución $(W_v,b_v,W_O)\to(W_vR,b_vR,R^{-1}W_O)$ deja la función intacta, y hace de la dirección dominante $v_1(W_O^{(h)})$ un proxy estático tentador para poda, interpretación o routing. Probamos que esa lectura no es identificable: su órbita bajo el gauge es la esfera unitaria completa del espacio fila, y ningún umbral admisible de similitud sobre ella admite radio certificado funcional positivo. Con espectro no degenerado, solo la clase conforme ortogonal la deja invariante para toda $W_O$, y toda lectura invariante de $W_O$ sola factoriza por su espacio fila. La consecuencia se mide: el par que la poda por pesos declara más redundante cambia bajo reparametrización genérica en más del $90$ % de los casos en un ViT y —por la misma forma cerrada, sin entrenar— en un transformer de lenguaje. En el gauge que el entrenamiento deja, la factorización se acerca al balance sin que las decisiones coincidan. Una sonda angular que separa las direcciones hasta el umbral del símplex no produce coste funcional detectable; la imposición dura sí. La firma de respuesta $v_1(C_h^P)$, condicionada a la sonda y gauge-invariante, mide diversidad funcional por cabeza. Código, datos y demo con DOI.

El código y las medidas se publican aquí, con el prerregistro de cada
experimento en [`prereg/`](prereg/); los CSV que respaldan cada tabla
y los checkpoints de reproducción se publican por separado en Hugging
Face (ver `## Datos y pesos` más abajo).

## Requisitos

- Python 3.11; PyTorch y [`timm`](https://github.com/huggingface/pytorch-image-models)
  para las columnas ViT-B/16 y ViT-L/16, [`transformers`](https://github.com/huggingface/transformers)
  para la columna de lenguaje (Pythia-410M). Dos ficheros, con dos
  papeles distintos: [`requirements_frozen.txt`](requirements_frozen.txt)
  es el `pip freeze` del entorno que produjo cada resultado del paper
  ---certificado el 11-08-2026---, y es lo que hay que instalar para
  reproducir exactamente; [`pyproject.toml`](pyproject.toml) y
  [`uv.lock`](uv.lock) declaran una resolución compatible, útil para
  una instalación nueva, pero no idéntica a aquel entorno.
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
prereg/    # prerregistro de cada experimento, commiteado antes de ejecutarlo
CHANGELOG.md
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

# E0 — d(R) de la ec. (2) sobre cada gauge muestreado — tab:gauge (columna d(R))
python scripts/gauge_dR.py

# E1 — nulo de poda aleatoria, 100 sorteos por semilla — tab:poda (columna de percentil)
python scripts/poda_nulo.py

# E2 — identidad de cabeza entre mitades de la sonda — tab:identidad
python scripts/identidad_mitades.py

# punto balanceado del sector valor-salida (comprobaciones del lema) y radio de la firma
python scripts/punto_balanceado.py
python scripts/radio_firma.py

# B1/B2 — curva de inestabilidad de la decisión frente a la fuerza del gauge; curva de tamaño de sonda
python scripts/curva_inestabilidad.py --disp cpu
python scripts/build_probe_anidado.py && python scripts/curva_sonda.py

# pruebas
pytest tests/ -q
```

## Datos y pesos

- Checkpoints de reproducción (pares base/blanda/dura por semilla y
  arquitectura): <https://huggingface.co/ManPla/angular-separation-vit-checkpoints>
  (DOI: [10.57967/hf/9742](https://doi.org/10.57967/hf/9742))
- CSV que respaldan cada tabla y figura del paper: <https://huggingface.co/datasets/ManPla/angular-separation-vit-results>
  (DOI: [10.57967/hf/9743](https://doi.org/10.57967/hf/9743))
- Código (este repositorio), archivado en Zenodo: <https://github.com/mmunozpl/angular-separation-vit>
  (DOI: [10.5281/zenodo.21630534](https://doi.org/10.5281/zenodo.21630534))

## Cómo citar

Si menciona o usa esta obra, cítela así (GitHub también ofrece el
botón «Cite this repository», generado desde
[CITATION.cff](CITATION.cff)):

```bibtex
@software{munozpla2026dominantdirection,
  author  = {Muñoz Plá, Manuel},
  title   = {The dominant direction of W_O is not identifiable: gauge
             orbit, zero certified radius, and consequences for pruning},
  year    = {2026},
  version = {v7.0.1},
  doi     = {10.5281/zenodo.21630534},
  url     = {https://github.com/mmunozpl/angular-separation-vit}
}
```

El manuscrito tendrá su propia entrada en cuanto el envío reciba un
identificador público.

## Licencia

Apache-2.0. Véase [LICENSE](LICENSE).
