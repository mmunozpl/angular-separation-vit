"""Barre leave-one-generator-out sobre GenImage para la contribución B.

Soporta dos ejes ortogonales:

- ``head.type`` en el YAML: ``proto`` (cabeza con prototipos sobre
  código esférico + margen angular) o ``linear`` (baseline lineal
  + CE estándar).
- ``seeds`` en el YAML: lista de semillas a barrer; los logs y
  checkpoints se sufijean con ``_seed<N>`` para que la agregación
  pueda reducir por generador (mean ± std).
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.genimage import build_logo_split
from src.models.linear_head import LinearHead
from src.models.proto_head import PrototypeHead
from src.models.vit_backbone import HeadProjections
from src.seed import set_seed
from src.train.train_detect import train_detect


def _make_head(cfg: dict) -> tuple[torch.nn.Module, str]:
    """Construye la cabeza según ``head.type`` del config."""
    head_cfg = cfg.get("proto_head", {})
    head_type = str(cfg.get("head", {}).get("type", "proto"))
    if head_type == "linear":
        head = LinearHead(
            num_classes=int(head_cfg["num_protos"]),
            feature_dim=int(head_cfg["feature_dim"]),
        )
    else:
        blob = torch.load(
            head_cfg["code_path"], map_location="cpu",
            weights_only=False,
        )
        code = blob["code"]
        assert code.shape[0] == int(head_cfg["num_protos"]), (
            f"código con {code.shape[0]} filas, se esperan "
            f"{head_cfg['num_protos']}"
        )
        head = PrototypeHead(
            num_protos=int(head_cfg["num_protos"]),
            feature_dim=int(head_cfg["feature_dim"]),
            code=code,
            trainable=head_cfg["mode"] != "fixed",
        )
    return head, head_type


def main() -> None:
    """Entrena un detector por cada (semilla, generador excluido)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    gens = list(cfg["dataset"]["generators"])
    seeds = cfg.get("seeds", [int(cfg["seed"])])
    seeds = [int(s) for s in seeds]

    log_root = Path(cfg["log_dir"])
    ckpt_root = Path(cfg["checkpoint_dir"])
    log_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    for seed in tqdm(seeds, desc="seeds"):
        set_seed(seed)
        for g in tqdm(gens, desc=f"LOGO sweep (seed={seed})",
                      leave=False):
            train_loader, val_loader, lbl_map = build_logo_split(
                root=cfg["dataset"]["root"],
                generators=gens,
                held_out=g,
                image_size=int(cfg["dataset"]["image_size"]),
                batch_size=int(cfg["optimization"]["batch_size"]),
                num_workers=int(
                    cfg["dataset"].get("num_workers", 8)
                ),
                real_subdir=str(
                    cfg["dataset"].get("real_subdir", "nature")
                ),
                max_per_folder=cfg["dataset"].get("max_per_folder"),
            )
            backbone = HeadProjections(
                model_name=str(cfg["model"]["name"]),
                pretrained=bool(cfg["model"]["pretrained"]),
                num_classes=0,
                img_size=int(cfg["dataset"]["image_size"]),
            )
            head, head_type = _make_head(cfg)
            params = (
                list(head.parameters())
                + list(backbone.parameters())
            )
            optimizer = torch.optim.AdamW(
                params,
                lr=float(cfg["optimization"]["lr"]),
                weight_decay=float(cfg["optimization"]["weight_decay"]),
            )
            tag = f"{g}_seed{seed}"
            train_detect(
                backbone=backbone,
                head=head,
                train_loader=train_loader,
                val_loader=val_loader,
                held_out_label=lbl_map[g],
                epochs=int(cfg["optimization"]["epochs"]),
                optimizer=optimizer,
                margin_deg=float(
                    cfg.get("proto_head", {}).get("margin_deg", 20.0)
                ),
                scale=float(
                    cfg.get("proto_head", {}).get("scale", 30.0)
                ),
                log_path=str(log_root / f"{tag}.csv"),
                ckpt_path=str(ckpt_root / f"{tag}.pt"),
                device=str(cfg["device"]),
                head_type=head_type,
            )


if __name__ == "__main__":
    main()
