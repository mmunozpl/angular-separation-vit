"""Energía de Riesz y su gradiente analítico en la esfera unidad."""

import torch


def riesz_energy(x: torch.Tensor, s: float = 1.0) -> torch.Tensor:
    """Calcula la energía de Riesz E_s sobre direcciones unitarias.

    Para K puntos en S^{d-1}, la energía es la suma sobre pares
    distintos de ``1 / ||x_i - x_j||^s``. El parámetro ``s`` regula la
    intensidad de la repulsión: ``s = 1`` recupera la electrostática y
    ``s -> 0`` la energía logarítmica de Whyte.

    Args:
        x: tensor (K, d) con norma unidad por fila.
        s: exponente de la energía, ``s > 0``.

    Returns:
        Tensor escalar con la energía total.
    """
    # diferencias por pares: (K, K, d)
    diff = x.unsqueeze(0) - x.unsqueeze(1)
    sq = (diff * diff).sum(-1)  # ||x_i - x_j||^2
    k = x.shape[0]
    # se enmascara la diagonal para evitar 1/0
    mask = ~torch.eye(k, dtype=torch.bool, device=x.device)
    e_pairs = sq[mask].clamp_min(1.0e-12).pow(-s / 2.0)
    return e_pairs.sum() / 2.0


def riesz_gradient(x: torch.Tensor, s: float = 1.0) -> torch.Tensor:
    """Gradiente analítico de la energía de Riesz, fila a fila.

    Sirve como referencia frente al gradiente vía autograd.

    Args:
        x: tensor (K, d) con norma unidad por fila.
        s: exponente de la energía.

    Returns:
        Tensor (K, d) con el gradiente por punto.
    """
    diff = x.unsqueeze(1) - x.unsqueeze(0)  # x_i - x_j, (K, K, d)
    sq = (diff * diff).sum(-1)  # (K, K)
    k = x.shape[0]
    eye = torch.eye(k, dtype=torch.bool, device=x.device)
    # se reemplaza la diagonal por un valor neutro para evitar nan
    sq = sq.masked_fill(eye, 1.0)
    coeff = sq.pow(-(s + 2.0) / 2.0)
    coeff = coeff.masked_fill(eye, 0.0)
    return -s * (coeff.unsqueeze(-1) * diff).sum(dim=1)
