"""Tests de validación geométrica de códigos esféricos."""

import math
import sys
from pathlib import Path

import pytest
import torch

# se permite importar src/ al ejecutar pytest desde la raíz
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.codes.canonical import canonical_kissing
from src.codes.generate import generate_code
from src.codes.riesz import riesz_energy, riesz_gradient
from src.codes.validate import min_angle_deg, validate_kissing
from src.seed import set_seed


def _device() -> str:
    """Selecciona cuda si está disponible."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize(
    "d,k",
    [(2, 6), (3, 12), (4, 24), (8, 240)],
)
def test_canonical_kissing_is_stable(d: int, k: int) -> None:
    """Descender desde la configuración canónica preserva 60°.

    Es el criterio práctico de validación: la configuración canónica
    es estable bajo el descenso de Riesz para s=1.
    """
    set_seed(0)
    init = canonical_kissing(d, k)
    assert init is not None
    code, _ = generate_code(
        k=k,
        d=d,
        s=1.0,
        steps=2000,
        lr=0.005,
        log_every=2000,
        device=_device(),
        init=init,
    )
    ok, th = validate_kissing(code, d=d, k=k, tol_deg=1.5)
    assert ok, f"d={d} K={k}: theta_min={th:.2f}° < 60°-1.5°"


@pytest.mark.parametrize(
    "d,k,steps",
    [
        (2, 6, 2000),
        (3, 12, 3000),
        (8, 240, 10000),
    ],
)
def test_random_init_recovers_kissing(
    d: int, k: int, steps: int
) -> None:
    """Desde inicio aleatorio se recupera la cota de kissing.

    El caso (4, 24) se excluye porque presenta un mínimo local muy
    estable a 55,23° desde inicialización gaussiana; ese par se
    valida con la inicialización canónica.
    """
    set_seed(0)
    code, _ = generate_code(
        k=k,
        d=d,
        s=1.0,
        steps=steps,
        lr=0.05,
        log_every=steps,
        device=_device(),
        seed=0,
    )
    ok, th = validate_kissing(code, d=d, k=k, tol_deg=1.5)
    assert ok, f"d={d} K={k}: theta_min={th:.2f}° < 60°-1.5°"


def test_min_angle_orthonormal() -> None:
    """Una base ortonormal tiene theta_min = 90°."""
    x = torch.eye(4)
    assert math.isclose(min_angle_deg(x), 90.0, abs_tol=1.0e-6)


def test_riesz_gradient_matches_autograd() -> None:
    """El gradiente analítico coincide con el de autograd."""
    set_seed(1)
    x = torch.randn(8, 5, dtype=torch.float64)
    x = x / x.norm(dim=1, keepdim=True)
    x_a = x.clone().requires_grad_(True)
    e = riesz_energy(x_a, s=1.0)
    e.backward()
    g_auto = x_a.grad
    g_ana = riesz_gradient(x, s=1.0)
    diff = (g_auto - g_ana).abs().max().item()
    assert diff < 1.0e-8, f"gradientes difieren en {diff:.2e}"


@pytest.mark.parametrize("d,k", [(2, 6), (3, 12), (4, 24), (8, 240)])
def test_canonical_construction(d: int, k: int) -> None:
    """Las construcciones canónicas tienen norma unidad y 60°."""
    x = canonical_kissing(d, k)
    assert x is not None
    assert x.shape == (k, d)
    norms = x.norm(dim=1)
    assert torch.allclose(norms, torch.ones(k), atol=1.0e-5)
    th = min_angle_deg(x)
    assert th >= 60.0 - 1.0e-3, f"d={d} K={k}: theta_min={th:.3f}°"
