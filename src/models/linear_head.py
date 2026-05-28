"""Cabeza lineal clásica (baseline de la Tabla 3 del paper).

Sirve como fila 1 ("cabeza lineal base") de la Tabla 3: clasifica
las M+1 categorías con ``nn.Linear`` y CE, sin prototipos y sin
margen angular. La intención es aislar el efecto del código esférico
y del rechazo en conjunto abierto frente a una arquitectura
estándar.
"""

import torch
from torch import nn


class LinearHead(nn.Module):
    """Capa lineal estándar para clasificación de M+1 categorías."""

    def __init__(
        self, num_classes: int, feature_dim: int,
    ) -> None:
        """Inicializa la capa.

        Args:
            num_classes: número de clases (M+1).
            feature_dim: dimensión del embedding de entrada.
        """
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Devuelve los logits.

        Args:
            feats: tensor (B, D).

        Returns:
            Tensor (B, M+1) de logits sin normalizar.
        """
        return self.fc(feats)
