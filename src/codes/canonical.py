"""Configuraciones canónicas para los kissing numbers conocidos.

Se utilizan como inicialización del descenso de Riesz para los pares
(d, K) en {(2,6), (3,12), (4,24), (8,240)}. Para (24, 196560) se
omite por coste; ahí se cae al inicio aleatorio.
"""

import itertools
import math

import torch


def hexagon_d2(k: int = 6) -> torch.Tensor:
    """6 vértices del hexágono regular en S^1, theta_min = 60°."""
    assert k == 6, "el hexágono solo admite K=6"
    ang = torch.arange(6, dtype=torch.float32) * (math.pi / 3.0)
    return torch.stack([ang.cos(), ang.sin()], dim=1)


def cuboctahedron_d3(k: int = 12) -> torch.Tensor:
    """12 vértices del cuboctaedro: (±1,±1,0), (±1,0,±1), (0,±1,±1).

    Es la configuración de besamiento (kissing) en d=3:
    theta_min = 60°.
    """
    assert k == 12, "el cuboctaedro solo admite K=12"
    rows: list[list[float]] = []
    for s1, s2 in itertools.product((1.0, -1.0), repeat=2):
        rows.append([s1, s2, 0.0])
        rows.append([s1, 0.0, s2])
        rows.append([0.0, s1, s2])
    x = torch.tensor(rows, dtype=torch.float32)
    return x / x.norm(dim=1, keepdim=True)


def d4_root_system(k: int = 24) -> torch.Tensor:
    """24 raíces del sistema D4: ±e_i ± e_j, i<j, en R^4.

    Son los vértices del 24-cell normalizados. theta_min = 60°.
    """
    assert k == 24, "D4 solo admite K=24"
    rows: list[list[float]] = []
    for i, j in itertools.combinations(range(4), 2):
        for si, sj in itertools.product((1.0, -1.0), repeat=2):
            v = [0.0] * 4
            v[i] = si
            v[j] = sj
            rows.append(v)
    x = torch.tensor(rows, dtype=torch.float32)
    return x / x.norm(dim=1, keepdim=True)


def e8_root_system(k: int = 240) -> torch.Tensor:
    """240 raíces del sistema E8 normalizadas (theta_min = 60°).

    La construcción combina:
    - 112 vectores de tipo ±e_i ± e_j, i<j en {1..8},
    - 128 vectores 1/2·(±1,...,±1) con producto de signos +1.
    """
    assert k == 240, "E8 solo admite K=240"
    rows: list[list[float]] = []
    for i, j in itertools.combinations(range(8), 2):
        for si, sj in itertools.product((1.0, -1.0), repeat=2):
            v = [0.0] * 8
            v[i] = si
            v[j] = sj
            rows.append(v)
    # 128 vectores 1/2·(s_1,...,s_8) con producto de signos = +1
    for bits in itertools.product((1.0, -1.0), repeat=8):
        if math.prod(bits) > 0:
            rows.append([b * 0.5 for b in bits])
    x = torch.tensor(rows, dtype=torch.float32)
    return x / x.norm(dim=1, keepdim=True)


def canonical_kissing(d: int, k: int) -> torch.Tensor | None:
    """Devuelve la configuración canónica si (d, k) es kissing exacto.

    Args:
        d: dimensión ambiente.
        k: número de puntos.

    Returns:
        Tensor (k, d) con la configuración canónica o None si no se
        dispone de construcción cerrada para ese par.
    """
    table = {
        (2, 6): hexagon_d2,
        (3, 12): cuboctahedron_d3,
        (4, 24): d4_root_system,
        (8, 240): e8_root_system,
    }
    fn = table.get((d, k))
    if fn is None:
        return None
    return fn(k)
