"""Proyecciones de códigos esféricos a 2D/3D para inspección."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def scatter_3d(
    code: torch.Tensor,
    title: str = "",
    path: str | None = None,
) -> None:
    """Proyecta sobre las tres primeras coordenadas y dibuja.

    Args:
        code: tensor (K, d) con filas en la esfera.
        title: título del gráfico.
        path: si se da, guarda la figura ahí (y no llama a show).
    """
    x = code[:, :3].cpu().numpy()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=12)
    ax.set_title(title)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def cos_histogram(
    edges: list[float], counts: list[int],
    title: str = "", path: str | None = None,
) -> None:
    """Dibuja el histograma de cosenos entre pares.

    Args:
        edges: bordes del histograma (N+1 valores).
        counts: contadores (N valores).
        title: título.
        path: si se da, guarda en disco.
    """
    e = np.asarray(edges, dtype=np.float64)
    c = np.asarray(counts, dtype=np.float64)
    centers = 0.5 * (e[:-1] + e[1:])
    width = e[1] - e[0]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(centers, c, width=width, color="#3a5fcd")
    ax.set_xlabel("cos(theta_ij)")
    ax.set_ylabel("pares")
    ax.set_title(title)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
