"""Métricas en régimen de conjunto abierto."""

import torch


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUROC binario calculado por ranks (Mann-Whitney U).

    Args:
        scores: tensor (N,) con un score donde mayor → clase positiva.
        labels: tensor (N,) binario, 1 = positivo, 0 = negativo.

    Returns:
        AUROC en [0, 1]; NaN si una de las clases está vacía.
    """
    s = scores.detach().cpu().to(torch.float64)
    y = labels.detach().cpu().to(torch.long)
    n_pos = int((y == 1).sum().item())
    n_neg = int((y == 0).sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = s.argsort()
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(
        1, s.numel() + 1, dtype=torch.float64,
    )
    sum_ranks_pos = float(ranks[y == 1].sum().item())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def roc_curve(
    scores: torch.Tensor,
    labels: torch.Tensor,
    n_points: int = 101,
) -> dict:
    """Curva ROC muestreada en ``n_points`` umbrales.

    Args:
        scores: tensor (N,) con un score donde mayor → positivo.
        labels: tensor (N,) binario.
        n_points: número de umbrales a barrer.

    Returns:
        Diccionario con listas 'thresholds', 'tpr' y 'fpr'.
    """
    s = scores.detach().cpu().to(torch.float64)
    y = labels.detach().cpu().to(torch.long)
    smin, smax = float(s.min().item()), float(s.max().item())
    thr = torch.linspace(smax, smin, n_points)
    n_pos = max(int((y == 1).sum().item()), 1)
    n_neg = max(int((y == 0).sum().item()), 1)
    tpr = []
    fpr = []
    for t in thr.tolist():
        pred = (s >= t).to(torch.long)
        tp = int(((pred == 1) & (y == 1)).sum().item())
        fp = int(((pred == 1) & (y == 0)).sum().item())
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)
    return {
        "thresholds": thr.tolist(),
        "tpr": tpr, "fpr": fpr,
    }


def oscr(
    closed_correct: torch.Tensor,
    closed_scores: torch.Tensor,
    novel_scores: torch.Tensor,
) -> float:
    """OSCR — área bajo la curva CCR (correcta) vs FPR (novedad).

    Args:
        closed_correct: máscara bool (N_c,), True si la predicción
            cerrada acierta.
        closed_scores: tensor (N_c,) con el score de confianza
            (mayor = más in-distribution).
        novel_scores: tensor (N_o,) con el score sobre OOD.

    Returns:
        OSCR escalar en [0, 1].
    """
    cs = closed_scores.detach().cpu().to(torch.float64)
    ns = novel_scores.detach().cpu().to(torch.float64)
    cc = closed_correct.detach().cpu().to(torch.bool)
    n_c = max(cs.numel(), 1)
    n_o = max(ns.numel(), 1)
    thr = torch.cat([cs, ns]).sort(descending=True).values
    pts: list[tuple[float, float]] = []
    for t in thr.tolist():
        ccr = float(
            ((cs >= t) & cc).sum().item()
        ) / n_c
        fpr = float((ns >= t).sum().item()) / n_o
        pts.append((fpr, ccr))
    pts.sort()
    area = 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        area += (x1 - x0) * 0.5 * (y0 + y1)
    return float(area)
