"""Barre leave-one-generator-out sobre GenImage para la contribución B.

Protocolo de evaluación (protocolo_eval_B.md):

- ``head.types`` en el YAML: lista de cabezas a entrenar por fold.
  ``proto`` (prototipos sobre código esférico + margen) y ``linear``
  (baseline CE estándar). Ambas comparten backbone, mismo train y
  mismas semillas para que la diferencia proto−lineal sea limpia
  (cambio 4). Se admite ``head.type`` singular por compatibilidad.
- ``seeds``: semillas de entrenamiento; barras de error por
  inicialización (no por datos: el subconjunto es fijo).
- ``dataset.n_train_per_gen``: submuestreo SOLO del train (cambio 2),
  con ``dataset.sample_seed`` fija e independiente de las semillas de
  entrenamiento. La val se evalúa completa (cambio 1).

Los logs y checkpoints se etiquetan ``<cabeza>_<gen>_seed<N>`` para que
``aggregate.py`` reduzca por (cabeza, generador) y separe familias.
"""

import argparse
import sys
import time
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


def _make_head(cfg: dict, head_type: str) -> torch.nn.Module:
    """Construye la cabeza del tipo indicado.

    Args:
        cfg: configuración cargada.
        head_type: ``proto`` o ``linear``.

    Returns:
        Módulo de cabeza listo para entrenar.
    """
    head_cfg = cfg.get("proto_head", {})
    if head_type == "linear":
        return LinearHead(
            num_classes=int(head_cfg["num_protos"]),
            feature_dim=int(head_cfg["feature_dim"]),
        )
    blob = torch.load(
        head_cfg["code_path"], map_location="cpu", weights_only=False,
    )
    code = blob["code"]
    assert code.shape[0] == int(head_cfg["num_protos"]), (
        f"código con {code.shape[0]} filas, se esperan "
        f"{head_cfg['num_protos']}"
    )
    return PrototypeHead(
        num_protos=int(head_cfg["num_protos"]),
        feature_dim=int(head_cfg["feature_dim"]),
        code=code,
        trainable=head_cfg["mode"] != "fixed",
    )


def _head_types(cfg: dict) -> list[str]:
    """Lista de cabezas a barrer; admite ``types`` o ``type``."""
    h = cfg.get("head", {})
    if "types" in h:
        return [str(t) for t in h["types"]]
    return [str(h.get("type", "proto"))]


def main() -> None:
    """Entrena un detector por (generador, semilla, cabeza)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]
    gens = list(ds["generators"])
    seeds = [int(s) for s in cfg.get("seeds", [int(cfg["seed"])])]
    head_types = _head_types(cfg)
    opt_cfg = cfg["optimization"]
    head_cfg = cfg.get("proto_head", {})

    log_root = Path(cfg["log_dir"])
    ckpt_root = Path(cfg["checkpoint_dir"])
    log_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    n_total = len(gens) * len(seeds) * len(head_types)
    print(
        f"barrido LOGO: {len(gens)} generadores × {len(seeds)} "
        f"semillas × {len(head_types)} cabezas = {n_total} corridas",
        flush=True,
    )
    done = 0
    first_dt: float | None = None

    for g in tqdm(gens, desc="LOGO sweep"):
        # loaders una vez por generador: el subconjunto de train es
        # fijo (sample_seed), idéntico para todas las semillas/cabezas
        train_loader, val_loader, lbl_map = build_logo_split(
            root=ds["root"],
            generators=gens,
            held_out=g,
            image_size=int(ds["image_size"]),
            batch_size=int(opt_cfg["batch_size"]),
            num_workers=int(ds.get("num_workers", 8)),
            real_subdir=str(ds.get("real_subdir", "nature")),
            n_train_per_gen=ds.get("n_train_per_gen"),
            n_val_per_gen=ds.get("n_val_per_gen"),
            sample_seed=int(ds.get("sample_seed", 1234)),
        )
        for seed in seeds:
            for head_type in head_types:
                # misma semilla -> misma init/orden para cada cabeza
                set_seed(seed)
                backbone = HeadProjections(
                    model_name=str(cfg["model"]["name"]),
                    pretrained=bool(cfg["model"]["pretrained"]),
                    num_classes=0,
                    img_size=int(ds["image_size"]),
                )
                head = _make_head(cfg, head_type)
                optimizer = torch.optim.AdamW(
                    list(head.parameters())
                    + list(backbone.parameters()),
                    lr=float(opt_cfg["lr"]),
                    weight_decay=float(opt_cfg["weight_decay"]),
                )
                tag = f"{head_type}_{g}_seed{seed}"
                t0 = time.perf_counter()
                train_detect(
                    backbone=backbone,
                    head=head,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    held_out_label=lbl_map[g],
                    epochs=int(opt_cfg["epochs"]),
                    optimizer=optimizer,
                    margin_deg=float(head_cfg.get("margin_deg", 20.0)),
                    scale=float(head_cfg.get("scale", 30.0)),
                    log_path=str(log_root / f"{tag}.csv"),
                    ckpt_path=str(ckpt_root / f"{tag}.pt"),
                    device=str(cfg["device"]),
                    head_type=head_type,
                )
                dt = time.perf_counter() - t0
                done += 1
                if first_dt is None:
                    # cronómetro de la primera corrida (cambio 5)
                    first_dt = dt
                    eta_h = first_dt * n_total / 3600.0
                    print(
                        f"\n[cronómetro] primera corrida ({tag}): "
                        f"{first_dt/60.0:.1f} min\n"
                        f"[proyección] {n_total} corridas "
                        f"≈ {eta_h:.1f} h de pared\n",
                        flush=True,
                    )
                print(
                    f"[progreso] {done}/{n_total}  ({tag})  "
                    f"{dt/60.0:.1f} min",
                    flush=True,
                )


if __name__ == "__main__":
    main()
