# Auditar este trabajo — correspondencia afirmación → script → dato

Cada tabla y cada figura del paper, con el script que la produce y el
fichero que la respalda. Es lo que convierte un repositorio en algo
comprobable por un tercero sin tener que creerse nada.

Versión **v6.0**. Autor: Manuel Muñoz Plá. Código bajo Apache-2.0
(ver `LICENSE`).

---

## 1. Dónde está cada cosa

Este repositorio lleva el **código**, los **tests** y los **PDF** del
paper. Los datos y los pesos viven aparte, con su propio DOI, porque
pesan:

| Pieza | Dónde | DOI |
|---|---|---|
| Código (este repositorio), archivado | [Zenodo](https://doi.org/10.5281/zenodo.21630534) | `10.5281/zenodo.21630534` |
| CSV que respaldan cada tabla y figura | [HF datasets](https://huggingface.co/datasets/ManPla/angular-separation-vit-results) | `10.57967/hf/9743` |
| Checkpoints de reproducción (por semilla y arquitectura) | [HF models](https://huggingface.co/ManPla/angular-separation-vit-checkpoints) | `10.57967/hf/9742` |

El DOI de Zenodo es el **concept**: resuelve siempre a la última
versión. Para citar una versión concreta, tómese su DOI hijo desde el
registro.

Rutas de la tabla de abajo: las de `artifacts/` son las que el script
escribe al correr, y las mismas con que los ficheros se publican en el
dataset de Hugging Face. `artifacts/` no se versiona aquí.

## 2. Correspondencia afirmación → script → dato

| Afirmación / tabla | Script | Dato |
|---|---|---|
| `tab:gauge` (invariancia bajo gauge, Fase G) | `scripts/run_fase_G.py` | `artifacts/logs/fase_G/gauge_flip.csv` |
| `tab:residuo` (firma computada vs. estática, Fase 0) | `scripts/run_fase_0.py` | `artifacts/logs/fase_0/computado_estatico.csv` |
| `tab:decision` (decisión de poda rota bajo gauge; visión 92,7 % / 0,378 y lenguaje 90,0 % / 0,379) | `scripts/decision_rota.py`, `scripts/decision_rota_lm.py` | `artifacts/logs/decision_rota/decision_rota{,_lm}.csv` |
| `tab:balance` (el gauge que el entrenamiento elige: alineación con el invariante) | `scripts/balance_espectral.py` | `artifacts/logs/balance/{balance_espectral,control_antes_afinado,balance_dinov2,control_dinov2}.csv` |
| `tab:natural` (acuerdo de decisión en el gauge natural) | `scripts/decision_natural.py` | `artifacts/logs/balance/decision_natural.csv` |
| §M2 (inercia verificada a la salida del modelo) | `scripts/inercia_salida.py` | `artifacts/logs/inercia_salida/inercia_salida.csv` |
| `tab:divattn` (ablación base/blanda/dura) | `scripts/ablation_table_A.py` | `artifacts/tables/ablation_A.csv`; crudo en `artifacts/logs/{vitb,vitl}_clean/attnA_*/{run,layerwise,entropy,head_norms}.csv` |
| `app:orbita` (órbita de gauge certificada) | `scripts/orbita_gauge.py` | `artifacts/logs/orbita_gauge/{orbita_gauge,certificado_simultaneo}.csv` |
| `tab:inversion`, `fig:inversion` (reubicación a Q·K con la profundidad) | `scripts/dissociation_D.py`, `scripts/principal_angles.py`, `scripts/fig_inversion.py` | `artifacts/tables/dissociation_D/by_layer.csv`, `artifacts/figs/` |
| `tab:poda` (poda por criterio: pesos vs. firma vs. azar) | `scripts/poda_criterio.py` | `artifacts/logs/poda_criterio/poda_criterio.csv` |
| §6.2 inercia de la sonda blanda | `scripts/inertia_read.py` | lee `artifacts/logs/<arch>_clean/attnA_{base,blanda}_seed*/run.csv` |
| §6.3 coste de la variante dura | `scripts/dura_cost_read.py` | lee los mismos `run.csv`, variante `dura` |
| *(legacy)* validación geométrica de los códigos esféricos | `scripts/gen_codes.py` | `artifacts/logs/codes.csv`, `artifacts/tables/tabla_codigos.tex` |

La fila marcada *legacy* no respalda ninguna tabla del paper vigente:
viene de una línea anterior del proyecto y se conserva porque el
código sigue en el árbol y corre.

Las dos **lecturas en frío** (§6.2 y §6.3) son reproducibles con solo
los `run.csv` del dataset. El resto de diagnósticos carga pesos
afinados y necesita los checkpoints.

## 3. Verificaciones que el propio repositorio corre

No son documentación: son scripts que fallan con código distinto de
cero si algo no cuadra.

| Qué comprueba | Script |
|---|---|
| Que las cifras bloqueadas del paper están en los tres PDF, con el separador decimal de cada idioma | `scripts/oraculo_cifras.py` |
| Que la demo reproduce las corridas publicadas **celda a celda** (264 celdas de `tab:gauge` y `tab:decision`), no dentro de banda | `scripts/paridad_demo.py` |
| Que el port de TMLR no arrastra ningún identificador del autor | `scripts/port_tmlr.py` |
| Suite de unidad y contratos de forma | `pytest tests/ -q` |

## 4. Reproducir

Entorno exacto en `pyproject.toml` / `uv.lock` (Python 3.11, PyTorch,
`timm`, `transformers`; una sola GPU NVIDIA, desarrollado en RTX 5090
32 GB, CUDA 13.0). La instalación es **aditiva**, no barre el entorno:

```bash
uv pip install -r <(uv export --no-hashes --no-dev)
pytest tests/ -q
```

`requirements_frozen.txt` es el `pip freeze` del entorno que produjo
los resultados. Es un entorno conda **compartido** con otros
proyectos, de ahí sus 382 paquetes: sirve para reproducir la
instalación tal cual, no como lista mínima de dependencias.

La lista completa de comandos, uno por resultado, está en `README.md`.
La columna de lenguaje (Pythia-410M) es cerrada sobre pesos y corre en
CPU.

## 5. Qué mirar con lupa

Los puntos donde una revisión independiente rinde más:

1. **La transformación de gauge** (`src/gauge_flip.py`): el sector
   valor-salida transforma también el sesgo de valor (`b_v ← Rᵀ b_v`);
   sin eso la salida no es invariante. La invariancia se mide con SVD
   exacta del circuito materializado, no con iteración de potencia.
2. **La firma funcional** (`src/firma_funcional.py`): es el
   instrumento que el paper propone como correcto; conviene comprobar
   que es efectivamente invariante bajo el gauge anterior.
3. **La lectura de la asimetría blanda/dura**: la sonda blanda no
   mueve la función y la dura sí la daña. El coste de la dura y la
   inercia de la blanda se leen con `n=5` (ViT-B) y `n=3` (ViT-L)
   desde los `run.csv` incluidos en el dataset.
4. **`scripts/poda_criterio.py`**: el desenlace es que ninguna
   similitud bate el suelo aleatorio; el suelo y su construcción son
   la parte a auditar.
5. **La afirmación negativa**: el resultado central es de
   no-identificabilidad, no un método. Contraejemplos que muestren que
   `v₁(W_O)` sí es identificable bajo alguna restricción razonable son
   la crítica más valiosa posible.
