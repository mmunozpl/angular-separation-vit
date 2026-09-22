# [Preprint] The dominant direction of $W_O$ is not identifiable: gauge orbit, zero certified radius, and consequences for pruning

🇬🇧 English · 🇪🇸 [Español](LEEME.md)

**Manuel Muñoz Plá** · [ORCID 0009-0000-5714-912X](https://orcid.org/0009-0000-5714-912X)

[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.21630534-009e73)](https://doi.org/10.5281/zenodo.21630534)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Dataset-ffd21e)](https://huggingface.co/datasets/ManPla/angular-separation-vit-results)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Model-ffd21e)](https://huggingface.co/ManPla/angular-separation-vit-checkpoints)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Space-ffd21e)](https://huggingface.co/spaces/ManPla/angular-separation-vit-demo)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0000--5714--912X-a6ce39)](https://orcid.org/0009-0000-5714-912X)
[![Web](https://img.shields.io/badge/Web-manpla.net-009e73)](https://manpla.net)
[![License](https://img.shields.io/badge/License-Apache--2.0-009e73)](LICENSE)
[![Cite](https://img.shields.io/badge/Cite-BibTeX-009e73)](#how-to-cite)

**Abstract:** The value-output factorisation of an attention head is not unique: for every $R\in GL(d_h)$, the substitution $(W_v,b_v,W_O)\to(W_vR,b_vR,R^{-1}W_O)$ leaves the function intact, and makes the dominant direction $v_1(W_O^{(h)})$ a tempting static proxy for pruning, interpretation or routing. We prove that this readout is not identifiable: its orbit under the gauge is the full unit sphere of the row space, and no admissible similarity threshold on it admits a positive functional certified radius. With a non-degenerate spectrum, only the conformal orthogonal class leaves it invariant for every $W_O$, and every invariant readout of $W_O$ alone factors through its row space. The consequence is measured: the pair that weight-based pruning declares most redundant changes under generic reparametrisation in more than $90$ % of cases on a ViT and —by the same closed form, without training— on a language transformer. In the gauge training leaves, the factorisation approaches balance without the decisions coinciding. An angular probe that separates the directions up to the simplex threshold produces no detectable functional cost; the hard imposition does. The response signature $v_1(C_h^P)$, probe-conditioned and gauge-invariant, measures functional diversity per head. Code, data and demo with DOIs.

Code and measurement scripts are published here, with the
pre-registration of every experiment under [`prereg/`](prereg/); the
CSVs backing every table and the reproduction checkpoints are
published separately on Hugging Face (see `## Data and weights`
below).

## Requirements

- Python 3.11; PyTorch and [`timm`](https://github.com/huggingface/pytorch-image-models)
  for the ViT-B/16 and ViT-L/16 columns, [`transformers`](https://github.com/huggingface/transformers)
  for the language column (Pythia-410M). Two files, two distinct
  roles: [`requirements_frozen.txt`](requirements_frozen.txt) is the
  `pip freeze` of the environment that produced every result in the
  paper ---certified 2026-08-11---, and is what to install for an
  exact reproduction; [`pyproject.toml`](pyproject.toml) and
  [`uv.lock`](uv.lock) declare a compatible resolution, useful for a
  fresh install but not identical to that environment.
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
prereg/    # pre-registration of every experiment, committed before running it
CHANGELOG.md
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

# E0 — d(R) of eq. (2) on every sampled gauge — tab:gauge (column d(R))
python scripts/gauge_dR.py

# E1 — random-pruning null, 100 draws per seed — tab:poda (percentile column)
python scripts/poda_nulo.py

# E2 — head identity across probe halves — tab:identidad
python scripts/identidad_mitades.py

# balanced point of the value-output sector (lemma checks) and signature radius
python scripts/punto_balanceado.py
python scripts/radio_firma.py

# B1/B2 — decision-instability curve vs. gauge strength; probe-size curve
python scripts/curva_inestabilidad.py --disp cpu
python scripts/build_probe_anidado.py && python scripts/curva_sonda.py

# tests
pytest tests/ -q
```

## Data and weights

- Reproduction checkpoints (base/soft/hard pairs per seed and
  architecture): <https://huggingface.co/ManPla/angular-separation-vit-checkpoints>
  (DOI: [10.57967/hf/9742](https://doi.org/10.57967/hf/9742))
- CSVs backing every table and figure in the paper: <https://huggingface.co/datasets/ManPla/angular-separation-vit-results>
  (DOI: [10.57967/hf/9743](https://doi.org/10.57967/hf/9743))
- Code (this repository), archived on Zenodo: <https://github.com/mmunozpl/angular-separation-vit>
  (DOI: [10.5281/zenodo.21630534](https://doi.org/10.5281/zenodo.21630534))

## How to cite

If you use or reference this work, please cite it as (GitHub also
offers a "Cite this repository" button, generated from
[CITATION.cff](CITATION.cff)):

```bibtex
@software{munozpla2026dominantdirection,
  author  = {Muñoz Plá, Manuel},
  title   = {The dominant direction of W_O is not identifiable: gauge
             orbit, zero certified radius, and consequences for pruning},
  year    = {2026},
  version = {v7.0},
  doi     = {10.5281/zenodo.21630534},
  url     = {https://github.com/mmunozpl/angular-separation-vit}
}
```

The manuscript itself will get its own entry once the submission is
assigned a public identifier.

## License

Apache-2.0. See [LICENSE](LICENSE).
