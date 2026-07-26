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
from src.firma_funcional import (CapturaContexto,
                                 geometria_circuito_exacta)
from src.reg_funcional import reg_funcional_localizado
from src.metrics.attention import (
    attention_entropy_per_layer,
    min_pairwise_angle_unsigned_deg,
    pairwise_redundancy,
)
from src.models.vit_backbone import capture_attention


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
    """Evalúa top-1, top-5 y métricas de atención sobre validación.

    Acumula además la similitud funcional entre cabezas por capa
    (cos entre mapas de atención promediados sobre las capturas), que
    se vuelca a ``run.csv`` y ``layerwise.csv`` por época. Permite
    seguir la evolución del desacople sin coste extra: reusa los
    mismos mapas que ya captura para la entropía.
    """
    from src.metrics.attention import (
        functional_similarity_matrix,
        mean_upper_offdiag,
    )

    model.eval()
    n_correct1 = 0
    n_correct5 = 0
    n_total = 0
    entropy_acc: torch.Tensor | None = None
    n_ent_batches = 0
    attn_sum: list[torch.Tensor] | None = None
    n_imgs_captured = 0
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
            if maps:
                ent = attention_entropy_per_layer(maps).detach()
                entropy_acc = (
                    ent if entropy_acc is None
                    else entropy_acc + ent
                )
                n_ent_batches += 1
                # acumular suma de mapas por capa para s_func al final
                if attn_sum is None:
                    attn_sum = [
                        m.detach().sum(dim=0) for m in maps
                    ]
                else:
                    for li, m in enumerate(maps):
                        attn_sum[li] += m.detach().sum(dim=0)
                n_imgs_captured += int(maps[0].shape[0])
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
    # similitud funcional por capa: media |pares no ordenados| del
    # mapa de atención promediado en el subconjunto capturado.
    s_func_layer: torch.Tensor | None = None
    if attn_sum is not None and n_imgs_captured > 0:
        s_layer = []
        for li in range(len(attn_sum)):
            avg_li = (attn_sum[li] / n_imgs_captured).cpu().float()
            mat = functional_similarity_matrix(avg_li)
            s_layer.append(mean_upper_offdiag(mat))
        s_func_layer = torch.stack(s_layer)
    return {
        "val_top1": top1,
        "val_top5": top5,
        "val_entropy_layer_head": entropy_mean,
        "val_s_func_layer": s_func_layer,
    }


def _captura_nan(ruta: str, ep: int, imgs, lbls, model, causa: str) -> None:
    """serializa el par (batch tóxico, pesos pre-step) en el primer skip.

    el NaN es propiedad del par (imagen, pesos), no de la imagen sola: una
    imagen puede saturar solo bajo la configuración de pesos del crash. el
    forward-scan sobre pesos representativos es ciego a esas; capturar el
    estado exacto aquí ---pesos antes del step, las imágenes del batch---
    cierra el par sin aproximación.

    args:
        ruta: fichero .pt de salida.
        ep: índice de época (0-based).
        imgs: tensor del batch tóxico.
        lbls: etiquetas del batch.
        model: el modelo en el estado pre-step (pesos del crash).
        causa: 'loss' o 'grad' (qué check disparó).
    """
    torch.save({
        "epoch": ep + 1,
        "causa": causa,
        "imgs": imgs.detach().cpu(),
        "lbls": lbls.detach().cpu(),
        "state_dict": {k: v.detach().cpu()
                       for k, v in model.state_dict().items()},
    }, ruta)
    print(f"[NaN-captura] par (batch,pesos) -> {ruta} "
          f"(ep{ep+1}, causa={causa})", flush=True)


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
    portador: str = "wo",
    mascara: list[int] | None = None,
    save_every: int = 0,
) -> None:
    """Entrena el ViT con R_div y guarda métricas amplias por época.

    El portador 'wo' es el camino de la v3 ---``head_directions`` +
    ``r_div`` + ``reproject_to_code``, intacto y literal, el brazo
    inerte---; 'ov'/'qk'/'parche'/'cls' separan el circuito o la
    contribución identificable vía ``reg_funcional``, sin reproyección.
    ``mascara`` localiza el regularizador por profundidad (None = todas).

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
    # log_dir actúa como raíz; los CSV y direcciones por época se
    # aíslan bajo log_dir/<run_name>/ para que un sweep con varias
    # corridas no sobrescriba los registros entre sí.
    log_p = Path(log_dir) / run_name
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
            "val_s_func_mean",
            "epoch_seconds", "img_per_sec",
            "grad_norm_last",
        ]),
        (layer_csv, [
            "epoch", "layer",
            "redundancy_mean", "redundancy_q25", "redundancy_q75",
            "theta_min_deg", "s_func",
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

    # selección de portador: 'wo' usa el camino de la v3; los demás leen
    # el vit interno y, para parche/cls, capturan el contexto a_h v_h.
    es_wo = portador == "wo"
    vit_int = None if es_wo else model.backbone.model
    nh = model.backbone.num_heads
    hd = model.backbone.head_dim
    # la máscara por defecto se deriva de la profundidad real del
    # modelo, no de una constante: una lista fija de 12 capas dejaría
    # sin regularizar la mitad profunda de un modelo de 24
    n_capas = len(model.backbone.model.blocks)
    mask = list(range(n_capas)) if mascara is None else mascara
    captura = None
    if portador in ("parche", "cls"):
        captura = CapturaContexto(vit_int, nh, hd)
    print(
        f"[{run_name}] portador={portador}  "
        f"mascara={'todas' if mascara is None else mascara}",
        flush=True,
    )
    # readout exacto de la geometría del circuito (ov/qk) por época: svd
    # exacta, no la iteración de potencia del loss, para juzgar la
    # separación sin ruido de init; theta_min y redundancia media juntos.
    es_circuito = portador in ("ov", "qk")
    carrier_csv = log_p / "carrier_geom.csv"
    if es_circuito and start_epoch == 0:
        with carrier_csv.open("w", newline="") as fh:
            csv.writer(fh).writerow(
                ["epoch", "theta_min_carrier_deg", "redund_carrier"])

    # captura del par (batch, pesos) en el PRIMER skip de NaN: el estado
    # exacto del crash, que el forward-scan sobre pesos representativos no
    # recupera (el NaN es del par imagen-pesos, no de la imagen sola)
    nan_captured = False
    nan_ruta = str(ck_p / f"{run_name}_nan_capture.pt")

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
        n_skip = 0  # batches tóxicos saltados (loss/grad no finito)
        n_guard = 0  # reintentos ce-solo por nan de la svd del término
        div_con_grafo = True  # se suspende al primer nan de la época
        pbar = tqdm(
            train_loader, desc=f"ep {ep+1}/{epochs}", leave=False,
        )
        for imgs, lbls in pbar:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if es_wo:
                logits = model(imgs)
                loss_ce = ce(logits, lbls)
                # con lambda 0 (base y dura) el término no aporta
                # gradiente por definición, pero construir su grafo
                # arrastra el backward de la svd de head_directions,
                # que en cuda produce nan cuando el par singular
                # dominante degenera (sigma1~sigma2, gap ~1e-4) y
                # 0*nan = nan envenena todo el gradiente (bloqueo de
                # dura44/45, 11-07). se computa sin grafo; la métrica
                # se registra igual. la blanda (lambda>0) puede
                # producir el mismo nan por un empate interior exacto
                # del espectro (sigma_i == sigma_j a precisión de
                # máquina, con el par dominante sano): no hay proxy
                # previo fiable, así que se detecta el nan exacto en
                # gnorm y se suspende el grafo del término el resto
                # de la época (se re-arma cada época).
                if lambda_div == 0.0 or not div_con_grafo:
                    with torch.no_grad():
                        dirs = model.head_directions()
                        loss_div = r_div(
                            dirs[mask],
                            theta_target_deg=theta_target_deg)
                else:
                    dirs = model.head_directions()
                    # se localiza r_div a las capas de la máscara;
                    # con la máscara completa queda la v3 literal
                    loss_div = r_div(
                        dirs[mask],
                        theta_target_deg=theta_target_deg)
            else:
                if captura is not None:
                    captura.limpiar()
                logits = model(imgs)
                loss_ce = ce(logits, lbls)
                loss_div = reg_funcional_localizado(
                    captura, vit_int, mask, portador,
                    n_cabezas=nh, dim_cabeza=hd)
            loss = loss_ce + lambda_div * loss_div
            # robustez: un batch tóxico (p. ej. imagen corrupta) puede dar
            # loss no finita -> NaN en los pesos. se salta sin backward ni
            # step. el grad-clip de umbral alto (10) es transparente en
            # pasos sanos (grad ~3-6) y solo capa explosiones. es red de
            # seguridad; la causa (imagen corrupta) se retira aparte.
            if not torch.isfinite(loss):
                if not nan_captured:
                    _captura_nan(nan_ruta, ep, imgs, lbls, model, "loss")
                    nan_captured = True
                optimizer.zero_grad(set_to_none=True)
                n_skip += 1
                continue
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=10.0)
            if not torch.isfinite(gnorm):
                if not nan_captured:
                    _captura_nan(nan_ruta, ep, imgs, lbls, model, "grad")
                    nan_captured = True
                optimizer.zero_grad(set_to_none=True)
                if es_wo and lambda_div > 0.0 and div_con_grafo:
                    # nan del backward de la svd del término:
                    # reintento ce-solo del batch y suspensión del
                    # grafo esta época; si aun así no es finito, es
                    # un batch tóxico real y se salta
                    div_con_grafo = False
                    n_guard += 1
                    logits = model(imgs)
                    loss_ce = ce(logits, lbls)
                    with torch.no_grad():
                        dirs = model.head_directions()
                        loss_div = r_div(
                            dirs[mask],
                            theta_target_deg=theta_target_deg)
                    loss = loss_ce + lambda_div * loss_div
                    loss.backward()
                    gnorm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=10.0)
                    if not torch.isfinite(gnorm):
                        optimizer.zero_grad(set_to_none=True)
                        n_skip += 1
                        continue
                else:
                    n_skip += 1
                    continue
            last_grad_norm = float(gnorm)
            optimizer.step()
            # reproyección al código solo en la dura del portador de pesos
            if es_wo and hasattr(model, "reproject_to_code"):
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
        # una época enteramente saltada es un bloqueo auto-sostenido:
        # sin step no hay cambio de pesos y el mismo nan recurre en el
        # siguiente batch, de modo que la recuperación es imposible por
        # construcción y continuar falsifica el conteo de épocas (30
        # filas sin 30 épocas). se aborta con marca grepeable y SIN
        # escribir la fila: el run.csv termina en la última época real,
        # con ce > 0. el salto de un batch suelto (más arriba) se
        # conserva: un batch tóxico aislado se salta y se sigue.
        if n_batches == 0 and n_skip > 0:
            print(f"[ABORT] epoca {ep + 1} enteramente saltada: "
                  f"bloqueo por skip, corrida FALLIDA "
                  f"({run_name}, skip={n_skip})", flush=True)
            raise SystemExit(3)
        # nb evita división por cero en el caso degenerado de un
        # loader vacío (sin batches y sin skips)
        nb = max(n_batches, 1)

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
        s_func_layer = ev.get("val_s_func_layer")
        s_func_mean = (
            float(s_func_layer.mean().item())
            if s_func_layer is not None else float("nan")
        )

        with main_csv.open("a", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow([
                ep + 1, f"{lr_now:.6g}",
                f"{loss_sum/nb:.4f}",
                f"{ce_sum/nb:.4f}",
                f"{div_sum/nb:.4f}",
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
                f"{s_func_mean:.6f}" if s_func_mean == s_func_mean
                else "",
                f"{ep_seconds:.3f}",
                f"{ips:.1f}",
                f"{last_grad_norm:.4f}",
            ])
        # rastro persistente en el terminal (tqdm con leave=False)
        print(
            f"[{run_name}] ep {ep+1:>3d}/{epochs}  "
            f"ce={ce_sum/nb:.3f}  "
            f"div={div_sum/nb:.4f}  "
            f"train_top1={train_top1:.3f}  "
            f"val_top1={ev['val_top1']:.3f}  "
            f"red={red.mean().item():.3f}  "
            f"theta_min={thmin.mean().item():.1f}°  "
            f"viol={n_viol}/{n_pairs_total}  "
            f"t={ep_seconds:.1f}s  ips={ips:.1f}  "
            f"|g|={last_grad_norm:.2f}"
            + (f"  skip={n_skip}" if n_skip else "")
            + (f"  guard={n_guard}" if n_guard else ""),
            flush=True,
        )
        # lectura exacta del circuito (svd) para juzgar la separación
        if es_circuito:
            th_c, red_c = geometria_circuito_exacta(
                vit_int, portador, nh, hd)
            with carrier_csv.open("a", newline="") as fh:
                csv.writer(fh).writerow(
                    [ep + 1, f"{th_c:.4f}", f"{red_c:.6f}"])
            print(
                f"[{run_name}] ep {ep+1} circuito {portador} (svd "
                f"exacta): theta_min={th_c:.1f}°  redund={red_c:.4f}",
                flush=True,
            )
        with layer_csv.open("a", newline="") as fh:
            wr = csv.writer(fh)
            for li in range(red.numel()):
                s_func_li = (
                    float(s_func_layer[li].item())
                    if s_func_layer is not None else float("nan")
                )
                wr.writerow([
                    ep + 1, li,
                    f"{red[li].item():.6f}",
                    f"{q25[li].item():.6f}",
                    f"{q75[li].item():.6f}",
                    f"{thmin[li].item():.4f}",
                    f"{s_func_li:.6f}" if s_func_li == s_func_li
                    else "",
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
        # traza por época para la curva pago-vs-exactitud (post-hoc):
        # solo el modelo, ligero y purgable tras adjudicar por época.
        if save_every and (ep + 1) % save_every == 0:
            torch.save(
                {"model": model.state_dict(), "epoch": ep + 1,
                 "val_top1": ev["val_top1"]},
                ck_p / f"{run_name}_ep{ep + 1:02d}.pt",
            )

    if captura is not None:
        captura.quitar()

    with (log_p / "summary.json").open("w") as fh:
        json.dump({"epochs": epochs, "run_name": run_name}, fh)
    print(
        f"\ncorrida '{run_name}' completada: {epochs} épocas. "
        f"logs en {log_p}/  checkpoints en {ck_p}/",
        flush=True,
    )
