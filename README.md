# Maximum angular separation in Vision Transformers

Reference implementation of the preprint *Separación angular máxima en
Transformers de visión: códigos esféricos para una atención diversa y
una detección generalizable de medios sintéticos*.

The repository implements and validates three pieces:

1. **Spherical codes** — projected gradient descent on the Riesz
   energy over `S^{d-1}`, validated against the kissing number in
   `d ∈ {2, 3, 4, 8, 24}`.
2. **Contribution A** — an angular-diversity regularizer over the
   representative directions of a ViT's attention heads.
3. **Contribution B** — a per-generator prototype head anchored to a
   spherical code, with open-set rejection, for synthetic-image
   detection on GenImage under a leave-one-generator-out protocol.

## Requirements

- Python ≥ 3.10, PyTorch, [`timm`](https://github.com/huggingface/pytorch-image-models)
  for the ViT-B/16 backbone.
- A single NVIDIA GPU (developed on an RTX 5090, 32 GB).
- Datasets (ImageNet-1k / ImageNet-100, GenImage) are downloaded
  manually; paths are passed through the YAML configs and never
  fetched from code.

```bash
conda activate pytorch28
```

## Layout

```
configs/   # YAML configs for codes, contribution A and B
src/
  codes/   # Riesz energy, projected descent, kissing-number checks
  models/  # ViT backbone, diverse-attention ViT, prototype head
  losses/  # angular margin and R_div
  data/    # ImageNet and GenImage loaders (LOGO split)
  metrics/ # attention diversity, open-set (AUROC, OSCR)
  train/   # training loops for A and B
scripts/   # entry points (gen_codes, run_attn, run_detect, …)
tests/     # pytest suite
```

## Usage

```bash
# spherical codes
python scripts/gen_codes.py --config configs/codes.yaml

# contribution A — diverse attention
python scripts/run_attn.py --config configs/attn_imagenet.yaml

# contribution B — open-set detection (leave-one-generator-out)
python scripts/run_detect.py --config configs/detect_genimage.yaml

# tests
pytest tests/ -q
```

## License

Apache-2.0. See [LICENSE](LICENSE). If you use this software, please
cite it as described in [CITATION.cff](CITATION.cff).
