"""Bucle de la contribución A: entrena ViT con R_div sobre ImageNet."""

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.imagenet import (
    build_imagenet_loaders,
    build_imagenet_sanity_loaders,
)
from src.models.attn_diverse import AttnDiverseViT
from src.seed import set_seed
from src.train.train_attn import train_attn


def main() -> None:
    """Ejecuta una corrida de la contribución A."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--run-name", default="attnA",
        help="prefijo de logs y checkpoints",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    ds = cfg["dataset"]
    # carga opcional de wnids_file: lista canónica (p. ej. CMC-100)
    wnids: list[str] | None = None
    if "wnids_file" in ds:
        wnids_path = Path(ds["wnids_file"])
        wnids = [
            ln.strip() for ln in wnids_path.read_text().splitlines()
            if ln.strip()
        ]
        print(
            f"lista wnids cargada de {wnids_path}: "
            f"{len(wnids)} synsets",
            flush=True,
        )

    if str(ds.get("mode", "full")) == "sanity":
        # se reparte un único split en train/val (sin train_blurred)
        train_loader, val_loader = build_imagenet_sanity_loaders(
            root=ds["root"],
            val_subdir=str(ds.get("val_subdir", "val_blurred")),
            image_size=int(ds["image_size"]),
            batch_size=int(cfg["optimization"]["batch_size"]),
            num_workers=int(ds.get("num_workers", 8)),
            num_classes=int(ds["num_classes"]),
            val_fraction=float(ds.get("val_fraction", 0.1)),
            seed=int(cfg["seed"]),
            wnids=wnids,
        )
    else:
        max_per_class = ds.get("max_per_class")
        train_loader, val_loader = build_imagenet_loaders(
            root=ds["root"],
            image_size=int(ds["image_size"]),
            batch_size=int(cfg["optimization"]["batch_size"]),
            num_workers=int(ds.get("num_workers", 8)),
            num_classes=int(ds["num_classes"]),
            train_subdir=str(ds.get("train_subdir", "train")),
            val_subdir=str(ds.get("val_subdir", "val")),
            wnids=wnids,
            max_per_class=(
                int(max_per_class)
                if max_per_class is not None else None
            ),
        )

    # se carga el código solo si lo necesita la variante dura
    hard_code: torch.Tensor | None = None
    if cfg["loss"].get("hard_variant", False):
        code_blob = torch.load(
            cfg["code"]["path"], map_location="cpu",
            weights_only=False,
        )
        code = code_blob["code"]
        hard_code = code.unsqueeze(0).expand(
            int(cfg["model"]["num_layers"]), -1, -1,
        ).contiguous()

    # theta_target en régimen sin signo (paper §665 eq.6 + nota sobre
    # los dos regímenes). Las direcciones representativas son
    # vectores singulares (definidos salvo signo), así que la
    # separación con significado es arccos(|cos|), en [0°, 90°]. El
    # óptimo del símplex sin signo para K=H direcciones (H ≪ d) es
    #   theta* = arccos(1 / (H - 1))
    # Para H=12 son 84,7641°.
    H = int(cfg["model"]["num_heads"])
    theta_target_unsigned = math.degrees(math.acos(1.0 / (H - 1)))
    if "theta_target_deg" in cfg["loss"]:
        theta_target = float(cfg["loss"]["theta_target_deg"])
        if abs(theta_target - theta_target_unsigned) > 0.5:
            print(
                f"[aviso] override theta_target_deg="
                f"{theta_target:.3f}° (canónico sin signo para "
                f"H={H}: {theta_target_unsigned:.4f}°)",
                flush=True,
            )
    else:
        theta_target = theta_target_unsigned
        print(
            f"theta_target_deg (sin signo, H={H}) = "
            f"{theta_target:.4f}°",
            flush=True,
        )

    model = AttnDiverseViT(
        model_name=str(cfg["model"]["name"]),
        pretrained=bool(cfg["model"]["pretrained"]),
        num_classes=int(ds["num_classes"]),
        hard_code=hard_code,
        img_size=int(ds["image_size"]),
    )

    opt_cfg = cfg["optimization"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg["lr"]),
        weight_decay=float(opt_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(opt_cfg["epochs"]),
    )

    train_attn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(opt_cfg["epochs"]),
        optimizer=optimizer,
        scheduler=scheduler,
        lambda_div=float(cfg["loss"]["lambda_div"]),
        theta_target_deg=theta_target,
        label_smoothing=float(cfg["loss"]["ce_label_smoothing"]),
        log_dir=str(cfg["log_dir"]),
        ckpt_dir=str(cfg["checkpoint_dir"]),
        device=str(cfg["device"]),
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
