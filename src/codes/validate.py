"""Validación geométrica de códigos esféricos."""

import math

import torch


# kissing numbers conocidos (referencia para la validación)
KISSING_NUMBER: dict[int, int] = {
    2: 6,
    3: 12,
    4: 24,
    8: 240,
    24: 196560,
}


def min_angle_deg(x: torch.Tensor) -> float:
    """Ángulo mínimo (en grados) entre pares de filas de un código.

    Args:
        x: tensor (K, d) con filas en la esfera unidad.

    Returns:
        Ángulo mínimo entre cualesquiera dos filas distintas, en
        grados.
    """
    g = x @ x.t()
    k = x.shape[0]
    # se descarta la diagonal restándole un valor lo bastante grande
    g = g - 2.0 * torch.eye(k, device=x.device)
    cos_max = g.max().clamp(-1.0, 1.0).item()
    return math.degrees(math.acos(cos_max))


def validate_kissing(
    x: torch.Tensor,
    d: int,
    k: int,
    expected_deg: float = 60.0,
    tol_deg: float = 1.5,
) -> tuple[bool, float]:
    """Comprueba que se alcanza la cota inferior del kissing number.

    El kissing number garantiza que existe una configuración con
    theta_min ≥ 60°; el descenso de Riesz puede recalar en
    configuraciones más expandidas (p. ej. icosaedro en d=3, K=12, a
    63,43°). Por eso se valida la **cota inferior** y no la igualdad.

    Args:
        x: tensor (K, d) generado.
        d: dimensión esperada del código.
        k: número de puntos esperado.
        expected_deg: ángulo de referencia en grados (60° por defecto).
        tol_deg: tolerancia en grados sobre la cota inferior.

    Returns:
        Par (válido, theta_min_grados).
    """
    assert x.shape == (k, d), (
        f"forma inesperada {tuple(x.shape)}, se esperaba ({k}, {d})"
    )
    th = min_angle_deg(x)
    return th >= expected_deg - tol_deg, th
