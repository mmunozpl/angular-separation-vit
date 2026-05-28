# Pipeline: separación angular máxima en ViT

Documento técnico del modelo, las dos intervenciones, y el pipeline
operativo. Es una guía de implementación; el .tex es la versión
publicable. Mantenidos en sincronía (ver `notas_paper.md`).

## Tabla de contenidos

- [1. Resumen ejecutivo](#1-resumen-ejecutivo)
- [2. Fundamentos geométricos](#2-fundamentos-geometricos)
- [3. Arquitectura del modelo](#3-arquitectura-del-modelo)
- [4. Contribución A: regulador angular sobre cabezas](#4-contribucion-a)
- [5. Contribución B: prototipos en código esférico](#5-contribucion-b)
- [6. Pipeline operativo (fases 0-5)](#6-pipeline-operativo)
- [7. Métricas y artefactos](#7-metricas-y-artefactos)
- [8. Reproducibilidad](#8-reproducibilidad)
- [9. Estado actual de la corrida](#9-estado-actual)
- [10. Auditoría de métricas — gaps conocidos](#10-auditoria-gaps)

---

## 1. Resumen ejecutivo

El proyecto encuadra dos problemas distantes como un único problema
geométrico:

1. **Redundancia entre cabezas de un Transformer de visión** —
   las direcciones representativas de las `H=12` cabezas en cada
   capa de ViT-B/16 tienden a agruparse y duplicar trabajo.
2. **Fragilidad de los detectores de imagen sintética** frente a
   generadores no observados — los prototipos del clasificador no
   reservan espacio angular para lo desconocido.

La hipótesis común: ambos se resuelven imponiendo **separación
angular máxima** entre las direcciones relevantes sobre la
hiperesfera `S^{d-1}`. Se implementan dos intervenciones:

| Intervención | Dónde actúa | Mecanismo |
|---|---|---|
| **A** — atención angularmente diversa | cabezas multi-cabeza | regulador `R_div` que penaliza pares con cos > cos θ* |
| **B** — prototipos por generador | cabeza de clasificación | M+1 prototipos fijos sobre código esférico + margen angular |

Ambas son **aditivos geométricos**: no añaden parámetros ni cambian
la arquitectura. Compatibles con un ViT preentrenado afinado.

---

## 2. Fundamentos geométricos

### 2.1 Energía de Riesz

Dadas `K` direcciones unitarias `{u_1, ..., u_K} ⊂ S^{d-1}`, la
**energía de Riesz** con exponente `s > 0` es:

```
E_s(u_1, ..., u_K) = Σ_{i < j} 1 / ||u_i - u_j||^s
```

Minimizar `E_s` reparte las direcciones uniformemente sobre la
esfera. Implementación: [`src/codes/riesz.py`](src/codes/riesz.py).

### 2.2 Dos regímenes

La separación angular óptima `θ_min` depende de la relación entre
`K` y `d`:

| Régimen | Condición | Configuración óptima | θ_min |
|---|---|---|---|
| **holgado** | K ≤ d+1 | símplex regular / ETF | `arccos(-1/(K-1))` |
| **saturado** | K > d+1 | empaquetamiento, kissing number como caso ideal | ≥ 60° |

```
θ_min en función de K (régimen holgado, K ≤ d+1)
180° ┤ ●
     │  \
150° ┤   \
     │    ●
120° ┤      ●  (K=3, tetraedro/triángulo equilátero)
     │       \
 90° ┤        ●●●●●─────────────  (asíntota K → ∞)
     │            \___
 60° ┤                ───
     └─┬───┬───┬───┬───┬───┬───┬─── K
       2   4   8   12  24  64  256
                   ↑
                   H=12 cabezas → θ* ≈ 95,2°
```

### 2.3 El modelo opera en régimen holgado

Para ViT-B/16 con `d=768`:
- Cabezas de atención: `H=12`, prototipos: `K=9` (8 generadores + real)
- Ambos satisfacen `K << d+1`, todavía a 6 órdenes de magnitud del
  régimen saturado
- Separación óptima:
  - `θ*_cabezas = arccos(-1/11) ≈ 95,22°`
  - `θ*_protos = arccos(-1/8) ≈ 97,18°`
- Es por encima de 90° — los pares óptimos son **anti-correlacionados**

### 2.4 Algoritmo de descenso

[`src/codes/generate.py`](src/codes/generate.py) implementa
descenso de gradiente proyectado:

```
Entrada: d, K, s, T pasos, lr inicial η_0
1. Inicializar u_1..u_K aleatorias en S^{d-1}
2. Optimizador Adam con lr η_0
3. Planificador cosine annealing
4. Para t = 1..T:
   g_i        ← ∇_{u_i} E_s
   g_i^tan    ← g_i - <g_i, u_i> u_i      (proyección tangente)
   u_i        ← paso de Adam con g_i^tan
   u_i        ← u_i / ||u_i||             (reproyección a S^{d-1})
   actualizar lr según planificador
5. Devolver {u_1..u_K}
```

### 2.5 Validación geométrica

| `(d, K)` | Configuración canónica | θ_min recuperado | Notas |
|---|---|---|---|
| (2, 6) | hexágono | 60,00° ✓ | descenso aleatorio también lo recupera |
| (3, 12) | cuboctaedro | 63,43° ✓ | descenso *desliza al icosaedro* (Riesz prefiere icosaedro) |
| (4, 24) | D₄ (24-cell) | 60,00° ✓ | descenso aleatorio se atasca en 55,23° (cuenca estrecha) |
| (8, 240) | E₈ | 60,00° ✓ | descenso aleatorio sí converge |
| (24, 4096) | — | 57,22° | versión reducida del Leech (K original = 196560) |
| (768, 12) | ETF | 95,22° | régimen holgado |
| (768, 9)  | ETF | 97,18° | régimen holgado |

Los códigos se cachean en [`artifacts/codes/*.pt`](artifacts/codes/).

---

## 3. Arquitectura del modelo

```
┌──────────────┐  ┌──────────┐  ┌──────────────────┐  ┌──────┐
│ Imagen (B,3, │→ │  Patch   │→ │ Encoder ViT-B/16 │→ │ CLS  │
│  224, 224)   │  │ embed    │  │   L=12 bloques   │  │ z∈ℝ⁷⁶⁸│
└──────────────┘  └──────────┘  └────────┬─────────┘  └──┬───┘
                                          │              │
                          ┌────────────────────────┐    │
                          │ Contribución A         │    │
                          │ R_div sobre direcciones│    │
                          │ representativas de las │    │
                          │ H=12 cabezas por capa  │    │
                          └────────────────────────┘    │
                                                        ↓
                                              ┌─────────────────────┐
                                              │ Contribución B      │
                                              │ M+1 prototipos      │
                                              │ sobre código        │
                                              │ esférico (régimen   │
                                              │ holgado)            │
                                              └──────────┬──────────┘
                                                         ↓
                                              ┌─────────────────────┐
                                              │ cos s_k =           │
                                              │   <z/‖z‖, p_k>      │
                                              │ + margen angular    │
                                              │ + decisión real/    │
                                              │   sint / novel      │
                                              └─────────────────────┘
```

### 3.1 Backbone

ViT-B/16 desde [`timm`](https://github.com/huggingface/pytorch-image-models),
preentrenado en ImageNet-1k:

| Parámetro | Valor |
|---|---|
| Capas (L) | 12 |
| Cabezas por capa (H) | 12 |
| Dimensión embedding (d) | 768 |
| Dimensión por cabeza (d_h) | 64 |
| Parámetros | 86 M |
| Resolución de entrada | 224 px (Fase A) / 256 px (Fase B, vía `img_size` kwarg) |

Wrapper: [`src/models/vit_backbone.py:HeadProjections`](src/models/vit_backbone.py).

### 3.2 Dirección representativa de una cabeza

Para cada bloque `l` y cabeza `h`, la matriz `W_O^(l,h) ∈ ℝ^{d×d_h}`
es la proyección de salida de esa cabeza. La **dirección
representativa** es la media normalizada de sus columnas:

```
r_{l,h} = w̄_{l,h} / ‖w̄_{l,h}‖,    w̄_{l,h} = (1/d_h) Σ_c W_O^(l,h)[:,c]
```

Implementación: [`HeadProjections.head_directions()`](src/models/vit_backbone.py).
Devuelve un tensor `(L, H, d) = (12, 12, 768)`.

### 3.3 Captura de mapas de atención

Para medir entropía de atención y otras diagnósticas:
[`capture_attention`](src/models/vit_backbone.py) registra hooks
sobre `attn_drop`, conmuta el `fused_attn` durante el `with` y
restaura su estado al salir.

---

## 4. Contribución A: regulador angular sobre cabezas {#4-contribucion-a}

### 4.1 Pérdida `R_div`

```
R_div = Σ_l Σ_{h ≠ h'} max(0, <r_{l,h}, r_{l,h'}> - cos θ*)
```

donde `θ* = arccos(-1/(H-1)) ≈ 95,22°` para `H=12` (régimen
holgado). Penaliza pares de cabezas en la misma capa que estén a
ángulo menor del óptimo del símplex.

Implementación: [`src/losses/angular.py:r_div`](src/losses/angular.py).
**Suma sobre pares ordenados** (`h ≠ h'`, 132 por capa × 12 capas =
1584 términos), **coseno con signo** (no `|·|`).

### 4.2 Variante blanda vs dura

| Modo | Mecanismo | Config |
|---|---|---|
| **Blanda** | Añade `λ · R_div` a la pérdida CE | `hard_variant: false`, `lambda_div: 0.001` |
| **Dura** | Inicializa W_O para que `r_{l,h}` coincida con el código + **reproyecta tras cada `optimizer.step()`** | `hard_variant: true` |

[`AttnDiverseViT.reproject_to_code()`](src/models/attn_diverse.py)
es el hook de reproyección. La pérdida en modo duro es solo CE
(`λ_div=0.0`).

### 4.3 Función objetivo Fase A

```
L = L_CE + λ_A · R_div    (modo blando, λ_A = 0.001)
L = L_CE                  (modo duro, R_div se cumple por construcción)
L = L_CE                  (baseline, λ_A = 0)
```

`L_CE` con `label_smoothing=0.1`.

### 4.4 Tres configuraciones de Tabla 2

| Fila Tabla 2 | Config | λ_A | hard | log_dir |
|---|---|---|---|---|
| ViT-B base | [`attn_imagenet_base.yaml`](configs/attn_imagenet_base.yaml) | 0 | no | `artifacts/logs/attn_base` |
| + R_div (blanda) | [`attn_imagenet.yaml`](configs/attn_imagenet.yaml) | 0.001 | no | `artifacts/logs/attn` |
| + código (dura) | [`attn_imagenet_dura.yaml`](configs/attn_imagenet_dura.yaml) | 0 | sí | `artifacts/logs/attn_dura` |

---

## 5. Contribución B: prototipos en código esférico {#5-contribucion-b}

### 5.1 Cabeza de prototipos

El embedding `[CLS]` normalizado `ẑ = z/‖z‖` se compara con `M+1`
prototipos unitarios `{p_0=real, p_1=g_1, ..., p_M=g_M}`:

```
s_k = <ẑ, p_k>    ∈ [-1, 1]
```

Los `M+1` prototipos forman un código esférico en `ℝ⁷⁶⁸` (régimen
holgado, `M+1=9 << 768`), con separación óptima ≈ 97,18°.

Implementación: [`src/models/proto_head.py:PrototypeHead`](src/models/proto_head.py).
Modos: `fixed` (buffer) o `init_finetune` (parameter).

### 5.2 Pérdida con margen angular

```
L_proto(z, y) = CE( s · cos(θ_y + m), s · cos(θ_{k≠y}) ),    m=20°, s=30
```

ArcFace-like: añade un margen `m` al ángulo de la clase verdadera
para forzar separación. Implementación:
[`src/losses/angular.py:AngularMargin`](src/losses/angular.py).

### 5.3 Decisión en conjunto abierto

Para una imagen con scores `{s_0, ..., s_M}`:

```
real_vs_sintetico:  s_0  vs  max_{k>0} s_k     → si s_k > s_0: sintético
generador_novel:    max_k s_k < τ              → marcar como novel
```

`τ` es umbral configurable; el OSCR barre todos los umbrales para
generar la curva.

### 5.4 Protocolo LOGO (leave-one-generator-out)

Para cada generador `g` de los 8 de GenImage:
1. Entrena con todos los generadores **excepto** `g` (+ reales).
2. Evalúa sobre `g` (que el modelo no ha visto).
3. La media sobre los 8 cuantifica generalización a arquitecturas
   no observadas.

Implementación: [`src/data/genimage.py:build_logo_split`](src/data/genimage.py).

### 5.5 Cuatro configuraciones de Tabla 3

| Fila Tabla 3 | `head.type` | `proto_head.mode` | Notas |
|---|---|---|---|
| Cabeza lineal base | `linear` | — | nn.Linear + CE, sin código |
| + prototipos en código (fijo) | `proto` | `fixed` | buffer no entrenable |
| + prototipos init-y-afinar | `proto` | `init_finetune` | parameter, parte del código |
| + rechazo conjunto abierto | mismo proto | mismo | añade decisión novel/OSCR |
| + transferencia DRCT | reúsa cualquier checkpoint anterior | inferencia | sin entrenamiento |

Multi-semilla con `seeds: [42, 43, 44]` para reportar mean ± std.

---

## 6. Pipeline operativo (fases 0-5) {#6-pipeline-operativo}

```
Fase 0: andamiaje     Fase 1: códigos      Fase 2: ViT + métricas
seed determinista  →  Riesz descent     →  HeadProjections + 
load_config           gen_codes.py         capture_attention
                      validate.py
                            │
                            ↓ artifacts/codes/*.pt
Fase 3: Contribución A
train_attn (3 configs: blando, base, dura)
                            │
                            ↓ artifacts/logs/attn{,_base,_dura}/*.csv
Fase 4: Contribución B
train_detect (proto×2 + linear, ×3 semillas)
                            │
                            ↓ artifacts/logs/detect/*_seed*.csv
Fase 5: agregación
aggregate.py → tabularx + figs PNG
                            │
                            ↓ artifacts/tables/ + artifacts/figs/
```

### 6.1 Fases con script ejecutable

| Fase | Comando | Salida principal |
|---|---|---|
| 1 | `python scripts/gen_codes.py --config configs/codes.yaml` | `artifacts/codes/*.pt`, `artifacts/logs/codes.csv` |
| 3a | `python scripts/run_attn.py --config configs/attn_imagenet_base.yaml --run-name attnA_base` | `artifacts/logs/attn_base/` |
| 3b | `python scripts/run_attn.py --config configs/attn_imagenet.yaml --run-name attnA_blando` | `artifacts/logs/attn/` |
| 3c | `python scripts/run_attn.py --config configs/attn_imagenet_dura.yaml --run-name attnA_dura` | `artifacts/logs/attn_dura/` |
| 4 | `python scripts/run_detect.py --config configs/detect_genimage.yaml` | `artifacts/logs/detect/<gen>_seed<N>.csv` |
| 5 | `python scripts/aggregate.py` | `artifacts/tables/*.tex`, `artifacts/figs/*.png` |

### 6.2 Resume-safe

Fases 3 y 4 detectan checkpoint compatible al arrancar y reanudan
desde la siguiente época. Si el checkpoint es de formato antiguo
(sin `optimizer`), imprime aviso y arranca de cero. Tests en
[`tests/test_resume.py`](tests/test_resume.py).

Checkpoint extendido (Fase 3, [`train_attn.py`](src/train/train_attn.py)):

```
{
  "model":      state_dict,
  "optimizer":  state_dict,        ← nuevo (resume-safe)
  "scheduler":  state_dict | None, ← nuevo (resume-safe)
  "epoch":      N,
  "prev_dirs":  tensor cpu | None, ← para preservar dir_drift
  "val_top1":   float
}
```

Patrón análogo en Fase 4. Al arrancar:
1. Si `ckpt.exists()` y contiene `{model, optimizer, epoch}` →
   carga estado, `start_epoch = N`, CSVs en modo append.
2. Si falta alguna clave → imprime `[run] checkpoint incompatible
   (faltan claves: [...]); arrancando de cero`, trunca CSVs y
   empieza desde la época 1.

### 6.3 Selección de clases (ImageNet-100)

Por defecto se usa la **lista canónica de Contrastive Multiview
Coding** (Tian 2019), descargada de
[HobbitLong/CMC/imagenet100.txt](https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt)
y cacheada en
[`configs/imagenet100_cmc.txt`](configs/imagenet100_cmc.txt) —
100 wnids, todos presentes en `train_blurred` y `val_blurred`,
126 683 imágenes train.

El loader [`src/data/imagenet.py`](src/data/imagenet.py):
- Si `wnids_file` está en el YAML → filtra a esos synsets y
  **remapea etiquetas a [0..99]** según el orden del fichero.
- Si no → fallback a "primeras N synsets alfabéticas"
  (no comparable con literatura previa).

Implementación en `_RemappedSubset` (no es `Subset` estándar
porque hay que reasignar la etiqueta a la posición compacta).

### 6.4 Pruebas

```bash
pytest tests/ -q    # 50 tests verdes en ~2 min con GPU
```

Por suite:

| Suite | Tests | Cubre |
|---|---|---|
| `test_codes.py` | 13 | Fase 1 — Riesz, kissing, canónicas |
| `test_attention.py` | 8 | Fase 2 — ViT, head_directions, capture |
| `test_losses.py` | 8 | Fase 3 — R_div, AngularMargin, hard variant |
| `test_proto_and_openset.py` | 13 | Fase 4 — PrototypeHead, AUROC, OSCR |
| `test_baseline_and_seeds.py` | 5 | Fase 4 — LinearHead, multi-semilla |
| `test_data_and_aggregate.py` | 3 | Fase 4-5 — LOGO split, aggregate e2e |
| `test_resume.py` | 3 | Fases 3-4 — resume-safe |
| `test_cmc_wnids.py` | 4 | Fase 3 — wnids list, remap, missing |

---

## 7. Métricas y artefactos {#7-metricas-y-artefactos}

### 7.1 Fase 1 — códigos

| Artefacto | Descripción |
|---|---|
| `artifacts/codes/<name>.pt` | dict con `code` (K, d), `trace` (descenso), `stats`, `hist_cos_*` |
| `artifacts/logs/codes.csv` | resumen: theta_min, theta_max, energía Riesz, energía canónica |
| `artifacts/figs/hist_<name>.png` | histograma de cosenos por par |
| `artifacts/figs/descenso_<name>.png` | traza de energía y theta_min |

### 7.2 Fase 3 — atención

| CSV | Por época, contiene |
|---|---|
| `run.csv` | lr, train_loss, ce, div, train_top1/5, redundancy_mean/max, theta_min_mean/min/max, dir_drift, violating_pairs/total, val_top1/5, epoch_seconds, ips, grad_norm_last |
| `layerwise.csv` | por capa: redundancy_mean/q25/q75, theta_min |
| `entropy.csv` | por capa y cabeza: entropía media del mapa de atención en val |
| `head_norms.csv` | por capa y cabeza: media y desv. tip. de las normas de columnas de W_O |
| `directions/ep<NN>.pt` | tensor (L, H, D) de direcciones representativas |

### 7.3 Fase 4 — detección

| CSV | Contiene |
|---|---|
| `<gen>_seed<N>.csv` | por época: loss, ce, train_top1, train_acc_realfake, val_auroc_realfake, val_oscr, val_closed_top1, val_acc_realfake, precision, recall, f1, epoch_seconds, ips, grad_norm |
| `<gen>_seed<N>_per_class.csv` | por época y clase: n_samples, n_correct, accuracy |
| `<gen>_seed<N>_per_gen_auroc.csv` | por época y generador: AUROC binario real-vs-ese-generador |
| `predictions/<stem>/ep<NN>.pt` | cosenos, etiquetas, predicciones, scores, ROC — para reconstruir figuras |

### 7.4 Fase 5 — agregación

| Salida | Contenido |
|---|---|
| `tabla_codigos.tex` | Tabla 1 (validación geométrica) lista para `tabularx` |
| `tabla_atencion.tex` | Tabla 2 (diversidad de atención) — épocas seleccionadas |
| `tabla_logo.tex` | Tabla 3 (detección LOGO) — mean ± std sobre semillas |
| `figs/attn_layerwise.png` | redundancia y theta_min por capa en la última época |
| `figs/detect_roc.png` | ROC por generador excluido |

---

## 8. Reproducibilidad {#8-reproducibilidad}

- **Semilla global** en cada YAML (`seed: 42`).
  [`src/seed.py:set_seed()`](src/seed.py) fija `random`, `numpy`,
  `torch.cuda` y `cudnn.deterministic`.
- **Lista de clases canónica** (CMC) en
  [`configs/imagenet100_cmc.txt`](configs/imagenet100_cmc.txt) →
  el subset de ImageNet-100 es reproducible y comparable con
  literatura previa.
- **Códigos esféricos**: la misma semilla + canónica produce
  bit-a-bit el mismo `.pt`.
- **Entrenamientos**:
  - *Idempotencia funcional*: misma semilla → misma trayectoria.
  - *Idempotencia operativa*: re-correr no destruye progreso
    gracias a resume-safe (sección 6.2).
- **Tests**: 50 verdes (`pytest tests/ -q`).

---

## 9. Estado actual de la corrida {#9-estado-actual}

A día de hoy (2026-05-26):

| Item | Estado |
|---|---|
| Fase 1 códigos | ✓ generados y validados (artifacts/codes/ conservados) |
| Fase 3 corridas previas | ✗ borradas (logs + checkpoints + tablas + fig) tras decisión CMC |
| Lista CMC ImageNet-100 | ✓ descargada y wireada (`configs/imagenet100_cmc.txt`, 100 wnids, remap a [0..99]) |
| 3 configs Fase A con CMC | ✓ `attn_imagenet{,_base,_dura}.yaml` apuntan a `wnids_file: configs/imagenet100_cmc.txt` |
| Fase 3 con CMC | ⏳ pendiente relanzar las 3 corridas desde cero |
| GenImage descarga | ✓ 503 ficheros, 609 GB |
| GenImage extracción | ⏳ corriendo en background (~10h con contención HDD), destino WD2TB |
| Fase 4 LOGO | ⏳ pendiente de GenImage extraído |
| DRCT transferencia | ⏳ pendiente al final |

### 9.1 Hallazgo previo (corrida exploratoria con lista alfabética, ahora borrada)

La corrida exploratoria `attnA_blando` sobre lista alfabética
mostró que, con `λ_A = 0.001` y `θ* = 95,22°`, el ViT
**convergía al ETF en una época**:

```
ép.  ce    div    redundancy  theta_min  dir_drift  val_top1
 1   1.43 12.15   0.0908      95.16°     —          0.831
 5   1.23  0.23   0.0908      95.16°     0.0056     0.823
15   1.02  0.14   0.0909      95.18°     0.0020     0.839
30   0.89  0.003  0.0909      95.215°    0.000000   0.853
```

- `redundancy = 0.0908 ≈ |cos(95,22°)|` constante (el modelo
  acaba exactamente sobre el símplex regular).
- `dir_drift → 0` (las cabezas dejan de moverse).
- `val_top1` sube monotónico de 0.831 a 0.853 → R_div **no
  degrada** la exactitud.

**Estos números quedan como referencia preliminar**, no entran
en la Tabla 2 del paper. Los reproduciremos con la lista
canónica CMC en las nuevas corridas.

### 9.2 Limpieza ejecutada (2026-05-26)

Tras la decisión de adoptar la lista CMC, se borró todo lo
producido por Fase 3 con la lista alfabética para evitar
mezcla de protocolos en la Tabla 2 final:

```
✗ artifacts/logs/{attn,attn_base,attn_sanity*,attn_sanity_hard,
                  attn_sanity_mini,detect}/    (~18 MB)
✗ artifacts/checkpoints/{attn,attn_base,attn_sanity*}/  (~3,3 GB)
✗ artifacts/tables/*
✗ artifacts/figs/attn_layerwise.png

✓ artifacts/codes/*.pt                 conservados (7 códigos esféricos)
✓ artifacts/logs/codes.csv             conservado (Tabla 1 base)
✓ artifacts/figs/{hist,descenso}_*.png 14 figuras de Fase 1 conservadas
```

Los códigos esféricos no dependen del entrenamiento y siguen
valiendo. Las 14 figuras de Fase 1 (histogramas de cosenos por
par y trazas del descenso de Riesz) tampoco se ven afectadas.

### 9.3 Comando para relanzar Fase A con CMC

```bash
cd /media/manpla/Pruebas/Hiperesferas
conda activate pytorch28

python scripts/run_attn.py --config configs/attn_imagenet_base.yaml --run-name attnA_base && \
python scripts/run_attn.py --config configs/attn_imagenet.yaml --run-name attnA_blando && \
python scripts/run_attn.py --config configs/attn_imagenet_dura.yaml --run-name attnA_dura
```

Encadenado con `&&` (no `;`): si una falla, no arranca la
siguiente — útil para detectar problemas pronto. Tiempo total
estimado con cache caliente: ~10 h. Compite por HDD con la
extracción de GenImage durante la primera época cold-cache.

### 9.4 Sobre la extracción de GenImage

[`scripts/extract_genimage.py`](scripts/extract_genimage.py)
implementa un wrapper `ConcatFile` que presenta los 500
`.z000..z499` como un único stream seekable a `zipfile.ZipFile`,
sin generar el `.zip` combinado intermedio (que ocuparía 609 GB
extra). Destino: `/media/manpla/WD2TB/GenImage_extracted/`.
Tras verificar la integridad, se borran los `.zXXX` originales
para liberar 609 GB en ST2TB.

---

## 10. Auditoría de métricas — gaps conocidos {#10-auditoria-gaps}

Detalle completo en [`auditoria_metricas.md`](auditoria_metricas.md).
Resumen accionable:

### 10.1 Bug confirmado en `_violating_pairs` (no crítico)

[`src/train/train_attn.py:_violating_pairs`](src/train/train_attn.py)
usa `.abs()` sobre el coseno mientras compara con `cos θ* ≈ -0,09`
(negativo para `θ* = 95,22°`). Como `|cos| ≥ 0` siempre, la
condición evalúa True para **todos los pares**, dando
`viol = 792/792` constante. **R_div (la pérdida) está correcta**
y no usa `.abs()`. Solo el contador diagnóstico está mal. Una
línea de fix; sin urgencia porque no afecta entrenamiento.

### 10.2 Punto 4 del audit — dos resúmenes de cabeza

El paper §641-642 deja abierta la elección:

| Resumen | `red_mean` ép.0 | `θ_min_mean` ép.0 | Diagnóstico |
|---|---|---|---|
| Media de columnas (actual) | 0,035 | 82,6° | cabezas ≈ ortogonales |
| Primer vector singular | 0,208 | 55,8° | redundancia funcional alta |

→ El diagnóstico del régimen depende del resumen. **No es bug**,
es una decisión metodológica pendiente. Si se quiere reportar
ambos en la Tabla 2 hay que añadir el SVD como variante a
[`HeadProjections.head_directions()`](src/models/vit_backbone.py)
(~10 líneas).

### 10.3 Gaps de instrumentación pendientes

| # | Gap | Prioridad |
|---|---|---|
| ❸ | Multi-semilla en Fase 3 (Tabla 2 sin std mientras Tabla 3 sí) | alta |
| ❻ | AUROC closed-vs-novel no es columna del CSV de Fase 4 | media |
| ❼ | R_div opcional en Fase B (paper §739 lo promete) | media |
| ❽ | Ablación "código vs aleatorio" no expuesta en config | baja |
| ❿ | Script de transferencia DRCT (columna 3 de Tabla 3) | alta cuando termine Fase B |
| ⓫ | Figura única "θ_min(d) por dimensión" | media |
| ⓬ | Curvas OSCR (no ROC) por generador | media |

---

## Apéndice: mapa de ficheros

```
src/
├── seed.py                  # set_seed global
├── config.py                # load_config(yaml)
├── codes/                   # Fase 1
│   ├── riesz.py             # E_s y gradiente
│   ├── generate.py          # descenso proyectado Adam+tangente+cosine
│   ├── validate.py          # cota kissing, theta_min
│   └── canonical.py         # D4, E8, cuboctaedro, hexágono
├── data/                    # Fase 3 y 4
│   ├── imagenet.py          # loaders ImageFolder + sanity 90/10
│   └── genimage.py          # LOGO split
├── losses/
│   └── angular.py           # r_div (eq. 5 paper), AngularMargin
├── metrics/
│   ├── attention.py         # redundancy, theta_min, entropía
│   └── openset.py           # AUROC, ROC curve, OSCR
├── models/
│   ├── vit_backbone.py      # HeadProjections + capture_attention
│   ├── attn_diverse.py      # AttnDiverseViT con reproject_to_code
│   ├── proto_head.py        # PrototypeHead (fijo/entrenable)
│   └── linear_head.py       # baseline lineal Tabla 3
├── train/
│   ├── train_attn.py        # Fase 3, resume-safe
│   └── train_detect.py      # Fase 4, resume-safe
└── viz/
    └── sphere.py            # histograma cosenos + scatter 3d

scripts/
├── gen_codes.py             # Fase 1
├── run_attn.py              # Fase 3 (3 configs)
├── run_detect.py            # Fase 4 (LOGO sweep)
├── aggregate.py             # Fase 5
└── extract_genimage.py      # extracción zip multipart

configs/
├── codes.yaml               # generación de códigos
├── attn_imagenet.yaml       # Fase A blanda (con wnids_file CMC)
├── attn_imagenet_base.yaml  # Fase A baseline λ=0 (con wnids_file CMC)
├── attn_imagenet_dura.yaml  # Fase A variante dura (con wnids_file CMC)
├── attn_imagenet_sanity*.yaml  # smokes 1-3 épocas
├── detect_genimage.yaml     # Fase B LOGO
└── imagenet100_cmc.txt      # lista canónica CMC (100 wnids)

tests/                       # 50 tests
├── test_codes.py            # 13 — Fase 1, validación geométrica
├── test_attention.py        # 8  — Fase 2, ViT + métricas
├── test_losses.py           # 8  — Fase 3, R_div, AngularMargin, hard
├── test_proto_and_openset.py # 13 — Fase 4, PrototypeHead, AUROC, OSCR
├── test_baseline_and_seeds.py # 5 — Fase 4, LinearHead, multi-semilla
├── test_data_and_aggregate.py # 3 — Fase 4-5, LOGO split, aggregate
├── test_resume.py           # 3  — resume-safe (fresh/continue/incompat.)
└── test_cmc_wnids.py        # 4  — CMC list + remap a [0..99]
```
