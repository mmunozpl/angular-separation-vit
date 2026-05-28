"""Bucle de entrenamiento para la contribución A."""

import csv
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses.angular import r_div
from src.metrics.attention import (
    attention_entropy_per_layer,
    min_pairwise_angle_unsigned_deg,
    pairwise_redundancy,
)
from src.models.vit_backbone import capture_attention


def _global_grad_norm(params) -> float:
    """Norma L2 global de los gradientes acumulados."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return total ** 0.5


def _layerwise_cos_quantiles(
    dirs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cuantiles q25 y q75 de |cos| por par y capa.

    Args:
        dirs: tensor (L, H, D) de direcciones unitarias.

    Returns:
        Par (q25, q75) con tensores (L,).
    """
    l, h, _ = dirs.shape
    q25 = dirs.new_zeros(l)
    q75 = dirs.new_zeros(l)
    if h < 2:
        return q25, q75
    idx_i, idx_j = torch.triu_indices(
        h, h, offset=1, device=dirs.device,
    )
    for li in range(l):
        pair = (dirs[li] @ dirs[li].t())[idx_i, idx_j].abs()
        qs = torch.quantile(
            pair.float(),
            torch.tensor([0.25, 0.75], device=dirs.device),
        )
        q25[li] = qs[0]
        q75[li] = qs[1]
    return q25, q75


def _violating_pairs(
    dirs: torch.Tensor, theta_target_deg: float,
) -> tuple[int, int]:
    """Cuenta pares con |cos| > cos(theta_target) y total de pares.

    Coherente con ``R_div`` (paper eq. 6) bajo el resumen SVD: las
    direcciones representativas están definidas salvo signo, por
    lo que la comparación usa el valor absoluto del coseno.

    Con ``theta* = 95,216°`` el umbral ``cos(theta*) ≈ -0,091`` es
    negativo y ``|cos| ≥ 0`` siempre, así que el contador es siempre
    máximo: el regulador penaliza por igual a todos los pares hasta
    llevarlos a ortogonalidad. Esta es la forma esperada cuando el
    régimen holgado permite orthogonalidad estricta.
    """
    l, h, _ = dirs.shape
    if h < 2:
        return 0, 0
    cos_t = math.cos(theta_target_deg * math.pi / 180.0)
    idx_i, idx_j = torch.triu_indices(
        h, h, offset=1, device=dirs.device,
    )
    n_viol = 0
    n_total = 0
    for li in range(l):
        pair = (dirs[li] @ dirs[li].t())[idx_i, idx_j].abs()
        n_viol += int((pair > cos_t).sum().item())
        n_total += int(pair.numel())
    return n_viol, n_total


def _head_norms(model: nn.Module) -> torch.Tensor:
    """Norma media y desv. tip. de columnas de W_O por cabeza-capa.

    Returns:
        Tensor (L, H, 2) con [mean, std] de las normas de las columnas
        que cada cabeza inyecta en la proyección de salida. Si el
        modelo no expone ``embed_dim``/``num_heads``/``head_dim`` o
        no tiene bloques, devuelve un tensor vacío (0, 0, 2).
    """
    out: list[torch.Tensor] = []
    backbone = model.backbone if hasattr(model, "backbone") else model
    if not all(
        hasattr(backbone, a) for a in ("embed_dim", "num_heads", "head_dim")
    ):
        return torch.zeros(0, 0, 2)
    m = backbone.model
    d = backbone.embed_dim
    h = backbone.num_heads
    hd = backbone.head_dim
    blocks = getattr(m, "blocks", [])
    for blk in blocks:
        w_o = blk.attn.proj.weight  # (D, D)
        wo = w_o.view(d, h, hd)
        # norma por columna: norma sobre la dim D para cada (head, col)
        cols = wo.norm(dim=0)  # (H, hd)
        out.append(torch.stack(
            [cols.mean(dim=1), cols.std(dim=1)], dim=-1,
        ))
    if not out:
        return torch.zeros(0, 0, 2)
    return torch.stack(out, dim=0).detach()


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    capture_every: int = 20,
) -> dict:
    """Evalúa top-1, top-5 y métricas de atención sobre validación."""
    model.eval()
    n_correct1 = 0
    n_correct5 = 0
    n_total = 0
    entropy_acc: torch.Tensor | None = None
    n_ent_batches = 0
    for bi, (imgs, lbls) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)
        if bi % capture_every == 0:
            with capture_attention(
                model.backbone.model
                if hasattr(model, "backbone")
                else model.model
            ) as maps:
                out = model(imgs)
            # solo se acumula entropía si el modelo expuso bloques
            if maps:
                ent = attention_entropy_per_layer(maps).detach()
                entropy_acc = (
                    ent if entropy_acc is None
                    else entropy_acc + ent
                )
                n_ent_batches += 1
        else:
            out = model(imgs)
        _, top5 = out.topk(5, dim=1)
        match5 = top5.eq(lbls.unsqueeze(1))
        n_correct5 += match5.any(dim=1).sum().item()
        n_correct1 += match5[:, 0].sum().item()
        n_total += lbls.numel()
    top1 = n_correct1 / max(n_total, 1)
    top5 = n_correct5 / max(n_total, 1)
    entropy_mean = (
        (entropy_acc / max(n_ent_batches, 1)).cpu()
        if entropy_acc is not None
        else None
    )
    return {
        "val_top1": top1,
        "val_top5": top5,
        "val_entropy_layer_head": entropy_mean,
    }


def train_attn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    lambda_div: float,
    theta_target_deg: float,
    label_smoothing: float,
    log_dir: str,
    ckpt_dir: str,
    device: str = "cuda",
    run_name: str = "run",
) -> None:
    """Entrena el ViT con R_div y guarda métricas amplias por época.

    Vuelca en ``log_dir``:

    - ``run.csv`` con, por época: lr, train_loss, ce, div,
      redundancy_mean/max, theta_min mean/min/max, dir_drift,
      violating_pairs, train_top1/5, val_top1/5, epoch_seconds,
      img_per_sec, grad_norm_last.
    - ``layerwise.csv`` con redundancy mean/q25/q75 y theta_min por
      capa.
    - ``entropy.csv`` con entropía por capa y cabeza (val).
    - ``head_norms.csv`` con norma media y desv. tip. de las columnas
      de W_O por capa-cabeza.
    - ``directions/ep<NN>.pt`` con tensores (L, H, D).
    """
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    log_p = Path(log_dir)
    log_p.mkdir(parents=True, exist_ok=True)
    ck_p = Path(ckpt_dir)
    ck_p.mkdir(parents=True, exist_ok=True)
    dir_p = log_p / "directions"
    dir_p.mkdir(exist_ok=True)

    main_csv = log_p / "run.csv"
    layer_csv = log_p / "layerwise.csv"
    ent_csv = log_p / "entropy.csv"
    head_norm_csv = log_p / "head_norms.csv"

    csv_specs = [
        (main_csv, [
            "epoch", "lr",
            "train_loss", "ce", "div",
            "train_top1", "train_top5",
            "redundancy_mean", "redundancy_max",
            "theta_min_mean_deg", "theta_min_min_deg",
            "theta_min_max_deg", "dir_drift",
            "violating_pairs", "total_pairs",
            "val_top1", "val_top5",
            "epoch_seconds", "img_per_sec",
            "grad_norm_last",
        ]),
        (layer_csv, [
            "epoch", "layer",
            "redundancy_mean", "redundancy_q25", "redundancy_q75",
            "theta_min_deg",
        ]),
        (ent_csv, ["epoch", "layer", "head", "entropy"]),
        (head_norm_csv, [
            "epoch", "layer", "head",
            "w_o_col_norm_mean", "w_o_col_norm_std",
        ]),
    ]

    model.to(device)
    hard = bool(getattr(model, "hard_code_applied", False))
    print(
        f"[{run_name}] arranca: epochs={epochs}  "
        f"lambda_div={lambda_div}  "
        f"theta_target_deg={theta_target_deg:.4f}  "
        f"hard_variant={hard}  "
        f"label_smoothing={label_smoothing}",
        flush=True,
    )
    prev_dirs: torch.Tensor | None = None
    start_epoch = 0

    # resume: si existe checkpoint con optimizer/scheduler, se carga
    ckpt_file = ck_p / f"{run_name}_last.pt"
    if ckpt_file.exists():
        try:
            blob = torch.load(
                ckpt_file, map_location=device, weights_only=False,
            )
            needed = {"model", "optimizer", "epoch"}
            if needed.issubset(blob.keys()):
                model.load_state_dict(blob["model"])
                optimizer.load_state_dict(blob["optimizer"])
                if scheduler is not None and blob.get("scheduler"):
                    scheduler.load_state_dict(blob["scheduler"])
                start_epoch = int(blob["epoch"])
                prev_dirs = blob.get("prev_dirs")
                print(
                    f"[{run_name}] reanudando desde época "
                    f"{start_epoch}/{epochs}",
                    flush=True,
                )
            else:
                missing = needed - blob.keys()
                print(
                    f"[{run_name}] checkpoint incompatible "
                    f"(faltan claves: {sorted(missing)}); "
                    f"arrancando de cero",
                    flush=True,
                )
        except Exception as exc:
            print(
                f"[{run_name}] no se pudo cargar checkpoint "
                f"({exc}); arrancando de cero",
                flush=True,
            )
            start_epoch = 0
            prev_dirs = None

    # cabeceras: solo si arrancamos de cero (truncate); si resumimos
    # se mantienen las filas existentes y se añade en modo append
    if start_epoch == 0:
        for path, header in csv_specs:
            with path.open("w", newline="") as fh:
                csv.writer(fh).writerow(header)

    for ep in range(start_epoch, epochs):
        model.train()
        ep_start = time.perf_counter()
        ce_sum = 0.0
        div_sum = 0.0
        loss_sum = 0.0
        n_batches = 0
        n_correct1 = 0
        n_correct5 = 0
        n_imgs = 0
        last_grad_norm = 0.0
        pbar = tqdm(
            train_loader, desc=f"ep {ep+1}/{epochs}", leave=False,
        )
        for imgs, lbls in pbar:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss_ce = ce(logits, lbls)
            dirs = model.head_directions()
            loss_div = r_div(dirs, theta_target_deg=theta_target_deg)
            loss = loss_ce + lambda_div * loss_div
            loss.backward()
            last_grad_norm = _global_grad_norm(model.parameters())
            optimizer.step()
            # reproyección al código en modo duro (paper §486)
            if hasattr(model, "reproject_to_code"):
                model.reproject_to_code()
            ce_sum += loss_ce.item()
            div_sum += loss_div.item()
            loss_sum += loss.item()
            n_batches += 1
            with torch.no_grad():
                _, top5 = logits.detach().topk(5, dim=1)
                match5 = top5.eq(lbls.unsqueeze(1))
                n_correct5 += int(match5.any(dim=1).sum().item())
                n_correct1 += int(match5[:, 0].sum().item())
                n_imgs += int(lbls.numel())
            pbar.set_postfix(
                ce=f"{loss_ce.item():.3f}",
                div=f"{loss_div.item():.3f}",
            )
        if scheduler is not None:
            scheduler.step()

        ep_seconds = time.perf_counter() - ep_start
        ips = n_imgs / max(ep_seconds, 1.0e-9)
        train_top1 = n_correct1 / max(n_imgs, 1)
        train_top5 = n_correct5 / max(n_imgs, 1)

        with torch.no_grad():
            dirs = model.head_directions().detach()
        red = pairwise_redundancy(dirs).detach()
        # theta_min sin signo (paper §665): apropiado para vectores
        # singulares; rango [0°, 90°]; óptimo = arccos(1/(H-1))
        thmin = min_pairwise_angle_unsigned_deg(dirs).detach()
        q25, q75 = _layerwise_cos_quantiles(dirs)
        n_viol, n_pairs_total = _violating_pairs(
            dirs, theta_target_deg,
        )

        if prev_dirs is not None and prev_dirs.shape == dirs.shape:
            # se mide drift como 1 - cos por cabeza, luego promedia
            sim = (dirs * prev_dirs.to(dirs.device)).sum(
                dim=-1
            ).clamp(-1.0, 1.0)
            drift = float((1.0 - sim).mean().item())
        else:
            drift = float("nan")
        prev_dirs = dirs.detach().clone().cpu()

        head_n = _head_norms(model)  # (L, H, 2)
        torch.save(dirs.cpu(), dir_p / f"ep{ep+1:03d}.pt")

        ev = _evaluate(model, val_loader, device)
        lr_now = float(optimizer.param_groups[0]["lr"])

        with main_csv.open("a", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow([
                ep + 1, f"{lr_now:.6g}",
                f"{loss_sum/n_batches:.4f}",
                f"{ce_sum/n_batches:.4f}",
                f"{div_sum/n_batches:.4f}",
                f"{train_top1:.4f}", f"{train_top5:.4f}",
                f"{red.mean().item():.4f}",
                f"{red.max().item():.4f}",
                f"{thmin.mean().item():.4f}",
                f"{thmin.min().item():.4f}",
                f"{thmin.max().item():.4f}",
                f"{drift:.6f}" if drift == drift else "",
                n_viol, n_pairs_total,
                f"{ev['val_top1']:.4f}",
                f"{ev['val_top5']:.4f}",
                f"{ep_seconds:.3f}",
                f"{ips:.1f}",
                f"{last_grad_norm:.4f}",
            ])
        # rastro persistente en el terminal (tqdm con leave=False)
        print(
            f"[{run_name}] ep {ep+1:>3d}/{epochs}  "
            f"ce={ce_sum/n_batches:.3f}  "
            f"div={div_sum/n_batches:.4f}  "
            f"train_top1={train_top1:.3f}  "
            f"val_top1={ev['val_top1']:.3f}  "
            f"red={red.mean().item():.3f}  "
            f"theta_min={thmin.mean().item():.1f}°  "
            f"viol={n_viol}/{n_pairs_total}  "
            f"t={ep_seconds:.1f}s  ips={ips:.1f}  "
            f"|g|={last_grad_norm:.2f}",
            flush=True,
        )
        with layer_csv.open("a", newline="") as fh:
            wr = csv.writer(fh)
            for li in range(red.numel()):
                wr.writerow([
                    ep + 1, li,
                    f"{red[li].item():.6f}",
                    f"{q25[li].item():.6f}",
                    f"{q75[li].item():.6f}",
                    f"{thmin[li].item():.4f}",
                ])
        if ev["val_entropy_layer_head"] is not None:
            ent = ev["val_entropy_layer_head"]
            with ent_csv.open("a", newline="") as fh:
                wr = csv.writer(fh)
                for li in range(ent.shape[0]):
                    for hi in range(ent.shape[1]):
                        wr.writerow([
                            ep + 1, li, hi,
                            f"{ent[li, hi].item():.6f}",
                        ])
        with head_norm_csv.open("a", newline="") as fh:
            wr = csv.writer(fh)
            for li in range(head_n.shape[0]):
                for hi in range(head_n.shape[1]):
                    wr.writerow([
                        ep + 1, li, hi,
                        f"{head_n[li, hi, 0].item():.6f}",
                        f"{head_n[li, hi, 1].item():.6f}",
                    ])
        # checkpoint resume-safe: incluye estado completo del optimizador,
        # planificador y prev_dirs para reanudar bit-a-bit en caso
        # de interrupción.
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": (
                    scheduler.state_dict() if scheduler is not None
                    else None
                ),
                "epoch": ep + 1,
                "prev_dirs": (
                    prev_dirs.cpu()
                    if prev_dirs is not None else None
                ),
                "val_top1": ev["val_top1"],
            },
            ckpt_file,
        )

    with (log_p / "summary.json").open("w") as fh:
        json.dump({"epochs": epochs, "run_name": run_name}, fh)
    print(
        f"\ncorrida '{run_name}' completada: {epochs} épocas. "
        f"logs en {log_p}/  checkpoints en {ck_p}/",
        flush=True,
    )
