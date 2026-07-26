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

- Python ≥ 3.10, PyTorch, and [`timm`](https://github.com/huggingface/pytorch-image-models)
  for the ViT-B/16 and ViT-L/16 columns; [`transformers`](https://github.com/huggingface/transformers)
  for the language column (Pythia-410M).
- A single NVIDIA GPU (developed on an RTX 5090, 32 GB); the language
  column is closed-form over weights and runs on CPU.
- Datasets (ImageNet-1k / ImageNet-100) are downloaded manually; paths
  are passed through the YAML configs and never fetched from code.

```bash
conda activate pytorch28
```

## Layout

```
configs/   # YAML configs for the ViT-B, ViT-L and DINOv2 columns
src/
  codes/     # Riesz energy, projected descent, kissing-number checks
  models/    # ViT backbone, diverse-attention ViT (soft/hard probe)
  losses/    # angular margin and R_div
  data/      # ImageNet loader
  metrics/   # head diversity and functional similarity
  train/     # probe training loop
scripts/   # entry points (training queues, gauge probes, decision
           # rota, ablation, certified orbit)
tests/     # pytest suite
archivo/   # closed verification phases (de-risk, sanity, smoke),
           # kept as process evidence, not as the live pipeline
```

## Usage

```bash
# anchor column ViT-B (n=5 seeds): base, soft and hard
bash scripts/cola_base_clean.sh && bash scripts/cola_finde.sh

# ViT-L column (n=3) and frozen DINOv2 column (n=1)
bash scripts/cola_vitl.sh
bash scripts/cola_dinov2.sh

# decision instability under gauge: vision (ViT-B) and language (Pythia-410M)
python scripts/decision_rota.py
python scripts/decision_rota_lm.py

# certified gauge orbit over the real weights
python scripts/orbita_gauge.py

# tests
pytest tests/ -q
```

## Data and weights

- Reproduction checkpoints (base/soft/hard pairs per seed and
  architecture): <https://huggingface.co/ManuelPla/angular-separation-vit-checkpoints>
- CSVs backing every table and figure in the paper: <https://huggingface.co/datasets/ManuelPla/angular-separation-vit-results>

## License

Apache-2.0. See [LICENSE](LICENSE). If you use this software, please
cite it as described in [CITATION.cff](CITATION.cff).
