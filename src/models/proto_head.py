"""Cabeza de prototipos por generador anclada a un código esférico."""

import torch
from torch import nn


class PrototypeHead(nn.Module):
    """Capa lineal cuyos pesos son prototipos en la esfera unidad.

    La salida es la similitud coseno de la entrada con cada
    prototipo. Si ``trainable`` es True, los prototipos son parámetros
    libres; si es False quedan como buffer fijo cargado del código.
    """

    def __init__(
        self,
        num_protos: int,
        feature_dim: int,
        code: torch.Tensor | None = None,
        trainable: bool = False,
    ) -> None:
        """Inicializa los prototipos.

        Args:
            num_protos: número de prototipos M+1.
            feature_dim: dimensión del espacio de features.
            code: tensor (M+1, D) opcional para inicializar.
            trainable: si True, los prototipos son ``nn.Parameter``.
        """
        super().__init__()
        if code is None:
            w = torch.randn(num_protos, feature_dim)
        else:
            assert code.shape == (num_protos, feature_dim), (
                f"código {tuple(code.shape)} != "
                f"({num_protos}, {feature_dim})"
            )
            w = code.clone()
        w = w / w.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
        if trainable:
            self.protos = nn.Parameter(w)
        else:
            self.register_buffer("protos", w)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Devuelve los cosenos contra cada prototipo.

        Args:
            feats: tensor (B, D).

        Returns:
            Tensor (B, M+1) de similitudes coseno.
        """
        f = feats / feats.norm(
            dim=1, keepdim=True
        ).clamp_min(1.0e-12)
        p = self.protos
        if isinstance(p, nn.Parameter):
            p = p / p.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
        return f @ p.t()
