"""Pérdidas angulares: regulador R_div y margen angular."""

import math

import torch
from torch import nn


def r_div(
    dirs: torch.Tensor,
    theta_target_deg: float = 60.0,
) -> torch.Tensor:
    """Regulador de diversidad angular (paper §665, eq. 6).

    Con direcciones representativas dadas por el primer vector
    singular derecho ``v_1(W_O^(h))`` ---la dirección dominante de
    escritura en el residuo, en R^768, definida salvo signo---, la
    pérdida usa **el valor absoluto** del producto escalar:

    ``R_div = sum_{h != h'} max(0, |<r_h, r_{h'}>| - cos theta*)``

    Sumado sobre pares ordenados y sobre las capas. ``theta*`` es la
    separación angular del símplex regular en el régimen holgado
    (95,216° para H=12 en d=768).

    Args:
        dirs: tensor (L, H, D) o (H, D) de direcciones unitarias.
        theta_target_deg: ángulo objetivo en grados.

    Returns:
        Tensor escalar con la pérdida sumada.
    """
    if dirs.dim() == 2:
        dirs = dirs.unsqueeze(0)
    l, h, _ = dirs.shape
    cos_target = math.cos(theta_target_deg * math.pi / 180.0)
    cos_target_t = dirs.new_tensor(cos_target)
    off_diag = (
        ~torch.eye(h, dtype=torch.bool, device=dirs.device)
    ).to(dirs.dtype)
    loss = dirs.new_zeros(())
    for li in range(l):
        g = dirs[li] @ dirs[li].t()
        # |cos|: invariante al signo, que es arbitrario en SVD
        excess = (g.abs() - cos_target_t).clamp_min(0.0) * off_diag
        loss = loss + excess.sum()
    return loss


class AngularMargin(nn.Module):
    """Margen angular sobre logits de prototipos.

    Implementa la modificación habitual de ArcFace: se suma un margen
    ``m`` al ángulo de la clase verdadera y se escala el resultado.
    """

    def __init__(
        self, margin_deg: float = 20.0, scale: float = 30.0
    ) -> None:
        """Inicializa el módulo.

        Args:
            margin_deg: margen angular en grados.
            scale: factor de escala sobre los cosenos.
        """
        super().__init__()
        self.m = float(margin_deg * math.pi / 180.0)
        self.s = float(scale)

    def forward(
        self, cos_sim: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Devuelve los logits modificados por el margen angular.

        Args:
            cos_sim: tensor (B, C) con cosenos a cada prototipo.
            target: tensor (B,) con el índice verdadero.

        Returns:
            Tensor (B, C) listo para ``cross_entropy``.
        """
        eps = 1.0e-7
        theta = cos_sim.clamp(-1.0 + eps, 1.0 - eps).acos()
        one_hot = torch.zeros_like(cos_sim).scatter_(
            1, target.unsqueeze(1), 1.0
        )
        theta_m = theta + one_hot * self.m
        return self.s * theta_m.cos()
