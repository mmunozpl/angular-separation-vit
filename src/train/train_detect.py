"""Bucle de entrenamiento para la contribución B."""

import csv
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses.angular import AngularMargin
from src.metrics.openset import auroc, oscr, roc_curve


def _cls_features(backbone: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    """Extrae el token CLS (o la pool) del backbone."""
    feats = backbone.forward_features(imgs)
    if feats.dim() == 3:
        return feats[:, 0]
    return feats


def _precision_recall_f1(
    pred_pos: torch.Tensor, y_pos: torch.Tensor,
) -> tuple[float, float, float]:
    """Precisión, recall y F1 sobre etiqueta binaria positiva."""
    tp = float(((pred_pos == 1) & (y_pos == 1)).sum().item())
    fp = float(((pred_pos == 1) & (y_pos == 0)).sum().item())
    fn = float(((pred_pos == 0) & (y_pos == 1)).sum().item())
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def train_detect(
    backbone: nn.Module,
    head: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    held_out_label: int,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    margin_deg: float,
    scale: float,
    log_path: str,
    ckpt_path: str,
    device: str = "cuda",
    dump_predictions: bool = True,
    head_type: str = "proto",
) -> None:
    """Entrena la cabeza sobre los generadores no excluidos y
    evalúa OOD sobre el excluido.

    Soporta dos tipos de cabeza vía ``head_type``:

    - ``proto``: PrototypeHead con margen angular (ArcFace-like) y
      scoring por cosenos; usada con prototipos sobre código esférico.
    - ``linear``: LinearHead estándar con CE sobre logits, sin
      margen. Es la fila 1 "cabeza lineal base" de Tabla 3.

    Métricas registradas por época en ``log_path`` (csv principal):

    - ``loss`` y ``ce`` (CE sobre logits con o sin margen),
    - ``train_top1`` y ``train_acc_realfake``,
    - AUROC binaria real vs sintético (global) + P/R/F1,
    - OSCR contra el excluido y top-1 cerrado,
    - tiempo de época y throughput,
    - norma gradiente al final.

    Adicionalmente:

    - ``<stem>_per_class.csv``: accuracy por clase por época,
    - ``<stem>_per_gen_auroc.csv``: AUROC real vs cada generador,
    - ``predictions/<stem>/ep<NN>.pt`` con cosenos/probabilidades,
      etiquetas, predicciones, scores y curva ROC.
    """
    assert head_type in {"proto", "linear"}, (
        f"head_type debe ser 'proto' o 'linear', no {head_type!r}"
    )
    margin = AngularMargin(
        margin_deg=margin_deg, scale=scale,
    ).to(device)
    ce = nn.CrossEntropyLoss()

    log_p = Path(log_path)
    log_p.parent.mkdir(parents=True, exist_ok=True)
    ck_p = Path(ckpt_path)
    ck_p.parent.mkdir(parents=True, exist_ok=True)
    pred_dir = ck_p.parent / "predictions" / log_p.stem
    if dump_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    per_class_csv = log_p.with_name(f"{log_p.stem}_per_class.csv")
    per_gen_csv = log_p.with_name(f"{log_p.stem}_per_gen_auroc.csv")

    csv_specs = [
        (log_p, [
            "epoch", "loss", "ce",
            "train_top1", "train_acc_realfake",
            "val_auroc_realfake", "val_oscr",
            "val_closed_top1", "val_acc_realfake",
            "val_precision", "val_recall", "val_f1",
            "epoch_seconds", "img_per_sec",
            "grad_norm_last",
        ]),
        (per_class_csv, [
            "epoch", "class_idx", "n_samples",
            "n_correct", "accuracy",
        ]),
        (per_gen_csv, [
            "epoch", "gen_label", "auroc",
            "n_real", "n_gen",
        ]),
    ]

    backbone.to(device)
    head.to(device)

    # resume: si existe el checkpoint con optimizer/scheduler, cargar
    start_epoch = 0
    if ck_p.exists():
        try:
            blob = torch.load(
                ck_p, map_location=device, weights_only=False,
            )
            needed = {"backbone", "head", "optimizer", "epoch"}
            if needed.issubset(blob.keys()):
                backbone.load_state_dict(blob["backbone"])
                head.load_state_dict(blob["head"])
                optimizer.load_state_dict(blob["optimizer"])
                start_epoch = int(blob["epoch"])
                print(
                    f"[{log_p.stem}] reanudando desde época "
                    f"{start_epoch}/{epochs}",
                    flush=True,
                )
            else:
                missing = needed - blob.keys()
                print(
                    f"[{log_p.stem}] checkpoint incompatible "
                    f"(faltan claves: {sorted(missing)}); "
                    f"arrancando de cero",
                    flush=True,
                )
        except Exception as exc:
            print(
                f"[{log_p.stem}] no se pudo cargar checkpoint "
                f"({exc}); arrancando de cero",
                flush=True,
            )
            start_epoch = 0

    # cabeceras solo si arrancamos de cero
    if start_epoch == 0:
        for path, header in csv_specs:
            with path.open("w", newline="") as fh:
                csv.writer(fh).writerow(header)

    for ep in range(start_epoch, epochs):
        backbone.train()
        head.train()
        ep_start = time.perf_counter()
        loss_sum = 0.0
        ce_sum = 0.0
        n_batches = 0
        n_imgs = 0
        n_correct = 0
        n_correct_rf = 0
        last_grad_norm = 0.0
        pbar = tqdm(
            train_loader, desc=f"ep {ep+1}/{epochs}", leave=False,
        )
        for imgs, lbls in pbar:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            mask = lbls != held_out_label
            if not mask.any():
                continue
            imgs = imgs[mask]
            lbls = lbls[mask]
            optimizer.zero_grad(set_to_none=True)
            feats = _cls_features(backbone, imgs)
            out = head(feats)
            # margen angular solo en modo proto; linear usa logits crudos
            if head_type == "proto":
                logits = margin(out, lbls)
            else:
                logits = out
            loss = ce(logits, lbls)
            loss.backward()
            last_grad_norm = 0.0
            for p in list(backbone.parameters()) + list(head.parameters()):
                if p.grad is not None:
                    last_grad_norm += float(
                        p.grad.detach().pow(2).sum().item()
                    )
            last_grad_norm = last_grad_norm ** 0.5
            optimizer.step()
            loss_sum += loss.item()
            ce_sum += loss.item()
            n_batches += 1
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                n_correct += int((pred == lbls).sum().item())
                pred_rf = (pred != 0).to(torch.long)
                y_rf = (lbls != 0).to(torch.long)
                n_correct_rf += int(
                    (pred_rf == y_rf).sum().item()
                )
                n_imgs += int(lbls.numel())
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        ep_seconds = time.perf_counter() - ep_start
        ips = n_imgs / max(ep_seconds, 1.0e-9)
        train_top1 = n_correct / max(n_imgs, 1)
        train_rf = n_correct_rf / max(n_imgs, 1)

        ev = _evaluate(
            backbone, head, val_loader, held_out_label, device,
            head_type=head_type,
        )
        with log_p.open("a", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow([
                ep + 1,
                f"{loss_sum/max(n_batches,1):.4f}",
                f"{ce_sum/max(n_batches,1):.4f}",
                f"{train_top1:.4f}", f"{train_rf:.4f}",
                f"{ev['auroc']:.4f}",
                f"{ev['oscr']:.4f}",
                f"{ev['closed_top1']:.4f}",
                f"{ev['acc_realfake']:.4f}",
                f"{ev['precision']:.4f}",
                f"{ev['recall']:.4f}",
                f"{ev['f1']:.4f}",
                f"{ep_seconds:.3f}",
                f"{ips:.1f}",
                f"{last_grad_norm:.4f}",
            ])
        print(
            f"ep {ep+1:>3d}/{epochs}  ({log_p.stem})  "
            f"loss={loss_sum/max(n_batches,1):.3f}  "
            f"train_top1={train_top1:.3f}  "
            f"AUROC={ev['auroc']:.3f}  OSCR={ev['oscr']:.3f}  "
            f"F1={ev['f1']:.3f}  "
            f"t={ep_seconds:.1f}s  ips={ips:.1f}",
            flush=True,
        )
        with per_class_csv.open("a", newline="") as fh:
            wr = csv.writer(fh)
            for c, (ns, nc) in ev["per_class"].items():
                acc = nc / max(ns, 1)
                wr.writerow([ep + 1, c, ns, nc, f"{acc:.6f}"])
        with per_gen_csv.open("a", newline="") as fh:
            wr = csv.writer(fh)
            for c, info in ev["per_gen_auroc"].items():
                wr.writerow([
                    ep + 1, c,
                    f"{info['auroc']:.6f}",
                    info["n_real"], info["n_gen"],
                ])
        if dump_predictions:
            torch.save(ev["dump"], pred_dir / f"ep{ep+1:03d}.pt")
        # checkpoint resume-safe: incluye optimizer y epoch
        torch.save(
            {
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": ep + 1,
                "auroc": ev["auroc"],
                "oscr": ev["oscr"],
            },
            ck_p,
        )

    with (log_p.parent / f"{log_p.stem}_summary.json").open("w") as fh:
        json.dump({"epochs": epochs, "held_out": held_out_label}, fh)
    print(
        f"\ncorrida detect ({log_p.stem}) completada: "
        f"{epochs} épocas. logs en {log_p.parent}/  "
        f"checkpoint en {ck_p}",
        flush=True,
    )


@torch.no_grad()
def _evaluate(
    backbone: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    held_out_label: int,
    device: str,
    head_type: str = "proto",
) -> dict:
    """Calcula AUROC global y por generador, OSCR, per-class accuracy y
    volcado completo.

    Para ``head_type='proto'`` los scores son cosenos en [-1, 1];
    para ``head_type='linear'`` se normalizan los logits con softmax
    para que los scores sean comparables (probabilidades en [0, 1]).
    """
    backbone.eval()
    head.eval()
    all_cos: list[torch.Tensor] = []
    all_lbl: list[torch.Tensor] = []
    all_pred: list[torch.Tensor] = []
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)
        feats = _cls_features(backbone, imgs)
        out = head(feats)
        if head_type == "linear":
            out = out.softmax(dim=-1)
        all_cos.append(out.cpu())
        all_lbl.append(lbls.cpu())
        all_pred.append(out.argmax(dim=1).cpu())
    cos = torch.cat(all_cos)
    lbl = torch.cat(all_lbl)
    pred = torch.cat(all_pred)

    # AUROC global: positivo = sintético; score = max cos sobre
    # prototipos sintéticos (mayor → más sintético)
    score_synth = cos[:, 1:].max(dim=1).values
    is_synth = (lbl != 0).to(torch.long)
    au = auroc(score_synth, is_synth)
    roc = roc_curve(score_synth, is_synth, n_points=101)

    pred_synth = (cos.argmax(dim=1) != 0).to(torch.long)
    acc_rf = float((pred_synth == is_synth).float().mean().item())
    prec, rec, f1 = _precision_recall_f1(pred_synth, is_synth)

    closed_mask = (lbl != held_out_label) & (lbl != 0)
    novel_mask = lbl == held_out_label
    score_conf = cos.max(dim=1).values
    closed_scores = score_conf[closed_mask]
    novel_scores = score_conf[novel_mask]
    closed_correct = (pred[closed_mask] == lbl[closed_mask])
    os_ = oscr(closed_correct, closed_scores, novel_scores)

    closed_top1 = (
        float(closed_correct.float().mean().item())
        if closed_mask.any() else float("nan")
    )

    # per-class accuracy
    per_class: dict[int, tuple[int, int]] = {}
    for c in lbl.unique().tolist():
        m = lbl == c
        ns = int(m.sum().item())
        nc = int((pred[m] == c).sum().item())
        per_class[int(c)] = (ns, nc)

    # AUROC real vs cada generador concreto (k in {1..M})
    per_gen: dict[int, dict] = {}
    for k in lbl.unique().tolist():
        k = int(k)
        if k == 0:
            continue
        mask = (lbl == 0) | (lbl == k)
        if mask.sum() == 0:
            continue
        y_k = (lbl[mask] == k).to(torch.long)
        s_k = cos[mask, k]
        per_gen[k] = {
            "auroc": auroc(s_k, y_k),
            "n_real": int((lbl == 0).sum().item()),
            "n_gen": int((lbl == k).sum().item()),
        }

    return {
        "auroc": au,
        "oscr": os_,
        "closed_top1": closed_top1,
        "acc_realfake": acc_rf,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "per_class": per_class,
        "per_gen_auroc": per_gen,
        "dump": {
            "cos": cos, "lbl": lbl, "pred": pred,
            "score_synth": score_synth,
            "score_conf": score_conf,
            "roc": roc,
            "held_out_label": held_out_label,
        },
    }
