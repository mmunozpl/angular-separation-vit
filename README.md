# Same function, different pruning

Reference implementation of the preprint *Same function, different
pruning: the dominant direction of $W_O$ under free, soft, and hard
intervention* (ES: *Dos modelos idénticos, dos podas distintas: la
dirección dominante de $W_O$ bajo intervención libre, suave y dura*).

The dominant direction of an attention head's output projection,
$v_1(W_O)$, is routinely used to prune redundant heads, interpret
their role, or route information between them. This work shows that
direction is not identifiable —a gauge freedom of the value-output
sector displaces it at will without touching the function—, that this
non-identifiability has a measurable consequence (the pruning decision
it produces changes under gauge in more than 90% of cases, both on a
vision-finetuned ViT and on a pretrained language transformer, without
any training), and that the coupling between that geometry and the
function is one-directional: a probe that angularly separates it up
to the simplex does not move the function, but imposing that same
separation by construction does damage it. As the correct instrument,
this work proposes the gauge-invariant response signature $v_1(C_h^P)$.

Code and measurement scripts are published here; the CSVs backing
every table and the reproduction checkpoints are published separately
on Hugging Face (see `## Data and weights` below).

## Requirements

- Python 3.11; PyTorch and [`timm`](https://github.com/huggingface/pytorch-image-models)
  for the ViT-B/16 and ViT-L/16 columns, [`transformers`](https://github.com/huggingface/transformers)
  for the language column (Pythia-410M) — exact versions locked in
  [`pyproject.toml`](pyproject.toml)/[`uv.lock`](uv.lock), frozen from
  the environment that produced every result in the paper.
- A single NVIDIA GPU (developed on an RTX 5090, 32 GB, CUDA 13.0);
  the language column is closed-form over weights and runs on CPU.
- Datasets (ImageNet-1k / ImageNet-100) are downloaded manually; paths
  are passed through the YAML configs and never fetched from code.

```bash
conda activate pytorch28
uv pip install -r <(uv export --no-hashes --no-dev)
```

(`uv sync` is deliberately not used: `pytorch28` is a conda
environment shared with other projects, and `uv sync` would remove
any package not declared in this repo's `pyproject.toml`. The command
above installs the exact locked versions additively instead.)

## Layout

```
configs/
  codes.yaml                     # spherical-code generation (Riesz descent)
  attn_vitb_{base,blanda,dura}_clean.yaml    # ViT-B column, 3 variants
  attn_vitl_{base,blanda,dura}_clean.yaml    # ViT-L column, 3 variants
  attn_dinov2_base_clean.yaml    # frozen DINOv2 column
  imagenet100_cmc.txt            # canonical 100-class synset list (CMC)
src/
  seed.py             # single set_seed for reproducibility
  config.py            # YAML config loader
  carga.py              # shared loaders: fine-tuned backbone + frozen probe set
  firma_funcional.py    # gauge-invariant response signature v1(C_h^P)
  gauge_flip.py         # value-output gauge transformation (Phase G)
  reg_funcional.py      # R_func: the angular probe, soft and hard
  codes/
    riesz.py       # Riesz energy and its gradient
    generate.py     # projected gradient descent on S^{d-1}
    canonical.py     # canonical inits for known kissing numbers
    validate.py      # theta_min vs. kissing-number lower bound
  models/
    vit_backbone.py   # ViT-B/L/DINOv2 backbone, attention hooks
    attn_diverse.py    # AttnDiverseViT: soft/hard angular probe
  losses/
    angular.py    # R_div (eq. 5) and angular margin
  data/
    imagenet.py    # ImageNet-100/1k loader
  metrics/
    attention.py    # head redundancy, entropy, representative direction
  train/
    train_attn.py    # training loop for the angular probe
  viz/
    sphere.py    # 2D/3D projections of spherical codes
scripts/   # one entry point per paper result — see Usage below
tests/     # pytest suite (attention, codes, losses, SVD fallback/GPU)
archivo/   # closed lines of work kept as process evidence, not the
           # live pipeline: the discarded detection contribution,
           # superseded paper drafts, orphaned src/scripts/configs
           # from earlier iterations, the SVD-crash postmortem, and
           # other closed exploratory probes and gates
```

## Usage

```bash
# Phase 1 — spherical codes: generate, cache, validate vs. kissing number
python scripts/gen_codes.py --config configs/codes.yaml

# anchor column ViT-B (n=5 seeds): base, soft and hard
bash scripts/cola_base_clean.sh && bash scripts/cola_finde.sh

# ViT-L column (n=3) and frozen DINOv2 column (n=1)
bash scripts/cola_vitl.sh
bash scripts/cola_dinov2.sh

# ablation table (base/soft/hard, val_top1/theta_min/redundancy) — tab:divattn
python scripts/ablation_table_A.py

# frozen 1000-image probe set shared by every diagnostic below
python scripts/build_probe_set.py

# Phase G — gauge-flip invariance: v1(W_O) drifts, the OV circuit doesn't — tab:gauge
python scripts/run_fase_G.py

# Phase 0 — computed-vs-static signature per layer — tab:residuo
python scripts/run_fase_0.py

# decision instability under gauge: vision (ViT-B) and language (Pythia-410M) — tab:decision
python scripts/decision_rota.py
python scripts/decision_rota_lm.py

# certified gauge orbit over the real weights — app:orbita
python scripts/orbita_gauge.py

# Q·K relocation with depth — tab:inversion, fig:inversion
python scripts/dissociation_D.py
python scripts/principal_angles.py
python scripts/fig_inversion.py

# pruning by criterion — weights vs. signature vs. random floor — tab:poda
python scripts/poda_criterio.py

# cold reads: soft-probe inertness and hard-variant cost (§6.2, §6.3)
python scripts/inertia_read.py
python scripts/dura_cost_read.py

# tests
pytest tests/ -q
```

## Data and weights

- Reproduction checkpoints (base/soft/hard pairs per seed and
  architecture): <https://huggingface.co/ManuelPla/angular-separation-vit-checkpoints>
  (DOI: [10.57967/hf/9742](https://doi.org/10.57967/hf/9742))
- CSVs backing every table and figure in the paper: <https://huggingface.co/datasets/ManuelPla/angular-separation-vit-results>
  (DOI: [10.57967/hf/9743](https://doi.org/10.57967/hf/9743))
- Code (this repository), archived on Zenodo: <https://github.com/mmunozpl/angular-separation-vit>
  (DOI: [10.5281/zenodo.21630535](https://doi.org/10.5281/zenodo.21630535))

## License

Apache-2.0. See [LICENSE](LICENSE). If you use this software, please
cite it as described in [CITATION.cff](CITATION.cff).
