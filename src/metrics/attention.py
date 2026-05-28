"""Métricas de diversidad y entropía de cabezas de atención."""

import torch


def pairwise_redundancy(dirs: torch.Tensor) -> torch.Tensor:
    """Redundancia (|cos|) media entre pares de cabezas por capa.

    Args:
        dirs: tensor (L, H, D) de direcciones unitarias.

    Returns:
        Tensor (L,) con la redundancia media por capa, en [0, 1].
    """
    l, h, _ = dirs.shape
    if h < 2:
        return dirs.new_zeros(l)
    idx_i, idx_j = torch.triu_indices(
        h, h, offset=1, device=dirs.device,
    )
    out = dirs.new_zeros(l)
    for li in range(l):
        g = dirs[li] @ dirs[li].t()
        pair = g[idx_i, idx_j].abs()
        out[li] = pair.mean()
    return out


def min_pairwise_angle_deg(dirs: torch.Tensor) -> torch.Tensor:
    """Ángulo mínimo entre pares de cabezas, por capa (con signo).

    Usa el coseno con signo: rango [0°, 180°]. Apropiada para
    direcciones con orientación definida ---p. ej. la validación
    geométrica de Fase 1, donde los códigos esféricos son vectores
    canónicos del símplex con cosenos pares negativos---.

    Para direcciones invariantes al signo (vectores singulares de
    la contribución A) usar ``min_pairwise_angle_unsigned_deg``.

    Args:
        dirs: tensor (L, H, D) de direcciones unitarias.

    Returns:
        Tensor (L,) con theta_min en grados, rango [0°, 180°].
    """
    l, h, _ = dirs.shape
    out = dirs.new_zeros(l)
    if h < 2:
        return out
    idx_i, idx_j = torch.triu_indices(
        h, h, offset=1, device=dirs.device,
    )
    for li in range(l):
        g = (dirs[li] @ dirs[li].t())[idx_i, idx_j]
        cos_max = g.max().clamp(-1.0, 1.0)
        out[li] = cos_max.acos() * (180.0 / torch.pi)
    return out


def min_pairwise_angle_unsigned_deg(dirs: torch.Tensor) -> torch.Tensor:
    """Ángulo mínimo sin signo entre pares de cabezas, por capa.

    Para vectores definidos salvo signo (p. ej. primer vector
    singular izquierdo de W_O), el ángulo con significado es
    ``arccos(|cos|)``, rango ``[0°, 90°]``. El óptimo del símplex
    sin signo para K=H direcciones es ``arccos(1/(H-1))``, que da
    84,76° para H=12 (no 95,22°, que era el complemento en el
    régimen con signo).

    Args:
        dirs: tensor (L, H, D) de direcciones unitarias.

    Returns:
        Tensor (L,) con theta_min sin signo, rango [0°, 90°].
    """
    l, h, _ = dirs.shape
    out = dirs.new_zeros(l)
    if h < 2:
        return out
    idx_i, idx_j = torch.triu_indices(
        h, h, offset=1, device=dirs.device,
    )
    for li in range(l):
        g = (dirs[li] @ dirs[li].t())[idx_i, idx_j].abs()
        cos_max = g.max().clamp(0.0, 1.0)
        out[li] = cos_max.acos() * (180.0 / torch.pi)
    return out


def functional_similarity_matrix(
    attn_avg: torch.Tensor,
) -> torch.Tensor:
    """Similitud coseno entre mapas de atención de cada par de cabezas.

    Mide diversidad *funcional* (comportamiento observado), no
    geométrica (direcciones de ``W_O``). Cada cabeza aporta su mapa de
    atención promediado sobre el lote; se aplana a un vector y se toma
    el coseno entre pares. Complementa a ``pairwise_redundancy``, que
    opera sobre las direcciones representativas.

    Args:
        attn_avg: tensor (H, T, T) con el mapa de atención promediado
            sobre el lote para una capa.

    Returns:
        Tensor (H, H) con la similitud coseno entre mapas de cabezas.
    """
    h = attn_avg.shape[0]
    flat = attn_avg.reshape(h, -1)
    flat = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    return flat @ flat.t()


def mean_upper_offdiag(mat: torch.Tensor) -> torch.Tensor:
    """Media de los elementos sobre la diagonal de una matriz cuadrada.

    Args:
        mat: tensor (H, H) simétrico; se promedia su triángulo superior
            estricto (pares no ordenados, sin la diagonal).

    Returns:
        Escalar con la media de los pares; 0 si ``H < 2``.
    """
    h = mat.shape[0]
    if h < 2:
        return mat.new_zeros(())
    iu = torch.triu_indices(h, h, offset=1, device=mat.device)
    return mat[iu[0], iu[1]].mean()


def attention_entropy(attn: torch.Tensor) -> torch.Tensor:
    """Entropía media de un mapa de atención por cabeza.

    Args:
        attn: tensor (B, H, T, T) con filas que suman 1.

    Returns:
        Tensor (H,) con la entropía media por cabeza.
    """
    eps = 1.0e-12
    # entropía por token y muestra: -sum_j p log p sobre la fila
    h = -(attn * (attn + eps).log()).sum(dim=-1)  # (B, H, T)
    return h.mean(dim=(0, 2))


def attention_entropy_per_layer(
    attn_maps: list[torch.Tensor],
) -> torch.Tensor:
    """Entropía media por capa y por cabeza.

    Args:
        attn_maps: lista de tensores (B, H, T, T), una por capa.

    Returns:
        Tensor (L, H) con la entropía media. Si la lista está vacía
        (p. ej. modelo sin bloques capturables) devuelve un tensor
        vacío de forma (0, 0).
    """
    if not attn_maps:
        return torch.zeros(0, 0)
    rows = [attention_entropy(a) for a in attn_maps]
    return torch.stack(rows, dim=0)
