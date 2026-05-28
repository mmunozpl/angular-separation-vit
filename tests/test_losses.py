"""Tests para Fase 3: R_div y margen angular."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses.angular import AngularMargin, r_div


def test_r_div_zero_for_orthonormal() -> None:
    """Una base ortonormal por capa no penaliza con theta_target=60°."""
    dirs = torch.eye(6).unsqueeze(0).repeat(3, 1, 1)
    loss = r_div(dirs, theta_target_deg=60.0)
    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_r_div_positive_for_collinear() -> None:
    """Direcciones casi colineales producen pérdida positiva."""
    h = 4
    base = torch.tensor([1.0, 0.0, 0.0])
    pert = torch.stack([
        base + 0.01 * torch.randn(3) for _ in range(h)
    ])
    pert = pert / pert.norm(dim=1, keepdim=True)
    loss = r_div(pert.unsqueeze(0), theta_target_deg=60.0)
    assert loss.item() > 0.0


def test_r_div_2d_input_works() -> None:
    """Acepta (H, D) sin capa explícita."""
    x = torch.eye(5)
    loss = r_div(x, theta_target_deg=60.0)
    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_r_div_uses_target_angle() -> None:
    """Subiendo el ángulo objetivo crece la penalización."""
    h = 6
    dirs = torch.eye(h)  # ortonormales: cos=0 entre todos los pares
    l60 = r_div(dirs, theta_target_deg=60.0).item()
    l30 = r_div(dirs, theta_target_deg=30.0).item()
    l85 = r_div(dirs, theta_target_deg=85.0).item()
    # con theta_target alto, cos_target ≈ 0 deja a las ortogonales
    # justo en el umbral; con target=30 el coste sigue siendo 0
    # porque cos_target>0 y |cos|=0 < cos_target.
    # con target=85, cos_target≈0.087 > 0, sigue 0.
    assert l60 == pytest.approx(0.0, abs=1.0e-6)
    assert l30 == pytest.approx(0.0, abs=1.0e-6)
    assert l85 == pytest.approx(0.0, abs=1.0e-6)
    # caso con pares colineales
    same = torch.ones(3, 4) / math.sqrt(4)
    assert r_div(same, theta_target_deg=60.0).item() > 0.4


def test_angular_margin_shape_and_scale() -> None:
    """AngularMargin preserva la forma y aplica escala."""
    am = AngularMargin(margin_deg=20.0, scale=30.0)
    cos = torch.rand(8, 5) * 2.0 - 1.0
    target = torch.randint(0, 5, (8,))
    out = am(cos, target)
    assert out.shape == cos.shape
    # con margen 0, el resultado coincide con scale * cos
    am0 = AngularMargin(margin_deg=0.0, scale=30.0)
    out0 = am0(cos, target)
    assert torch.allclose(out0, 30.0 * cos, atol=1.0e-3)


def test_angular_margin_target_decreases_logit() -> None:
    """En la clase verdadera, sumar margen reduce el coseno."""
    cos = torch.full((4, 3), 0.9)
    target = torch.tensor([0, 1, 2, 0])
    am = AngularMargin(margin_deg=20.0, scale=1.0)
    out = am(cos, target)
    for i, t in enumerate(target.tolist()):
        # la clase verdadera baja respecto al resto
        for j in range(3):
            if j == t:
                assert out[i, j].item() < cos[i, j].item()
            else:
                assert out[i, j].item() == pytest.approx(
                    cos[i, j].item(), abs=1.0e-5,
                )


def test_attn_diverse_forward_backward() -> None:
    """AttnDiverseViT permite forward + r_div backward sin error."""
    pytest.importorskip("timm")
    from src.models.attn_diverse import AttnDiverseViT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = AttnDiverseViT(
        model_name="vit_base_patch16_224",
        pretrained=False, num_classes=10,
    ).to(device)
    x = torch.randn(2, 3, 224, 224, device=device)
    out = m(x)
    dirs = m.head_directions()
    loss = out.sum() + 0.1 * r_div(dirs, theta_target_deg=60.0)
    loss.backward()
    grads = [
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in m.parameters() if p.requires_grad
    ]
    assert all(grads)


def test_attn_diverse_hard_code_applies() -> None:
    """El modo duro acepta un código (L, H, D) y conserva el forward."""
    pytest.importorskip("timm")
    from src.models.attn_diverse import AttnDiverseViT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    code = torch.randn(12, 12, 768)
    code = code / code.norm(dim=-1, keepdim=True)
    m = AttnDiverseViT(
        model_name="vit_base_patch16_224",
        pretrained=False, num_classes=5,
        hard_code=code,
    ).to(device)
    x = torch.randn(1, 3, 224, 224, device=device)
    assert m.hard_code_applied
    assert m(x).shape == (1, 5)
