"""Tests del resumen SVD y la reproyección dura por rotación."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses.angular import r_div
from src.models.attn_diverse import _rotate_align_columns
from src.models.vit_backbone import HeadProjections


def _device() -> str:
    """Selecciona cuda si está disponible."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_rotate_align_columns_basic() -> None:
    """La rotación lleva u1 a c y preserva las normas de columnas."""
    torch.manual_seed(0)
    d, hd = 32, 8
    w = torch.randn(d, hd)
    u, _, _ = torch.linalg.svd(w, full_matrices=False)
    u1 = u[:, 0]
    c = torch.randn(d)
    c = c / c.norm()
    if (u1 * c).sum() < 0:
        c = -c
    new_w = _rotate_align_columns(w, u1, c)
    # nuevo primer vector singular ≈ c
    nu, _, _ = torch.linalg.svd(new_w, full_matrices=False)
    new_u1 = nu[:, 0]
    sim = (new_u1 * c).sum().abs().item()
    assert sim > 0.9999, f"alineación falla: |<u1', c>|={sim:.6f}"


def test_rotate_align_preserves_singular_values() -> None:
    """La rotación ortogonal no cambia los valores singulares."""
    torch.manual_seed(1)
    d, hd = 32, 8
    w = torch.randn(d, hd)
    u, s_old, _ = torch.linalg.svd(w, full_matrices=False)
    u1 = u[:, 0]
    c = torch.randn(d)
    c = c / c.norm()
    if (u1 * c).sum() < 0:
        c = -c
    new_w = _rotate_align_columns(w, u1, c)
    _, s_new, _ = torch.linalg.svd(new_w, full_matrices=False)
    diff = (s_old - s_new).abs().max().item()
    assert diff < 1.0e-5, (
        f"valores singulares cambiaron en {diff:.2e}"
    )


def test_head_directions_svd_unit_norm() -> None:
    """head_directions devuelve vectores unitarios."""
    pytest.importorskip("timm")
    m = HeadProjections(pretrained=False, num_classes=10).to(_device())
    dirs = m.head_directions()
    assert dirs.shape == (12, 12, 768)
    norms = dirs.norm(dim=-1).flatten()
    assert torch.allclose(
        norms, torch.ones_like(norms), atol=1.0e-4,
    )


def test_hard_variant_aligns_first_singular() -> None:
    """Tras hard_variant, el primer SV de W_O^(h) coincide con c_h."""
    pytest.importorskip("timm")
    from src.models.attn_diverse import AttnDiverseViT

    # código aleatorio unitario (L, H, D)
    code = torch.randn(12, 12, 768)
    code = code / code.norm(dim=-1, keepdim=True)
    m = AttnDiverseViT(
        pretrained=False, num_classes=5, hard_code=code,
    ).to(_device())
    dirs = m.head_directions().detach().cpu()
    # |cos| entre dirs y code debe ser ≈ 1 en todos los pares (capa, h)
    sims = (dirs * code).sum(dim=-1).abs()
    min_sim = sims.min().item()
    assert min_sim > 0.999, (
        f"reproyección no aliena: min |cos| = {min_sim:.6f}"
    )


def test_unsigned_angle_metric() -> None:
    """min_pairwise_angle_unsigned_deg está en [0, 90] y trata
    vectores antiparalelos como paralelos."""
    from src.metrics.attention import min_pairwise_angle_unsigned_deg
    # 3 vectores ortogonales: angulo sin signo = 90°
    v = torch.eye(3).unsqueeze(0)
    th = min_pairwise_angle_unsigned_deg(v)
    assert torch.allclose(th, torch.tensor([90.0]), atol=1.0e-4)
    # un vector y su opuesto: angulo sin signo = 0°
    v = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])
    th = min_pairwise_angle_unsigned_deg(v)
    assert th.item() < 0.01


def test_unsigned_angle_simplex_target() -> None:
    """Para el símplex sin signo de K=12, el óptimo es arccos(1/11)."""
    import math
    from src.metrics.attention import min_pairwise_angle_unsigned_deg
    # construyo direcciones canonizadas (max-|comp| positivo) desde
    # un código simplex regular; canonizar voltea algunas filas,
    # rompiendo la estructura signed pero preservando |cos|=1/(K-1)
    K = 12
    blob = torch.load(
        "artifacts/codes/heads_vitb_layer.pt",
        map_location="cpu", weights_only=False,
    )
    code = blob["code"]  # (K, D)
    # canoniza el código
    argmax_idx = code.abs().argmax(dim=-1, keepdim=True)
    pivot = torch.gather(code, dim=-1, index=argmax_idx)
    sign = torch.where(
        pivot >= 0, torch.ones_like(pivot), -torch.ones_like(pivot),
    )
    code_c = code * sign
    th = min_pairwise_angle_unsigned_deg(code_c.unsqueeze(0))
    expected = math.degrees(math.acos(1.0 / (K - 1)))
    assert abs(th.item() - expected) < 0.5, (
        f"theta_min sin signo = {th.item():.3f}°, esperado "
        f"{expected:.3f}°"
    )


def test_rotate_align_columns_batched() -> None:
    """La versión batched produce el mismo resultado que el bucle."""
    from src.models.attn_diverse import (
        _rotate_align_columns,
        _rotate_align_columns_batched,
    )
    torch.manual_seed(7)
    H, d, hd = 5, 32, 8
    w = torch.randn(H, d, hd)
    # u1 = primer SVD por cabeza
    u, _, _ = torch.linalg.svd(w, full_matrices=False)
    u1 = u[:, :, 0]  # (H, d)
    # objetivos aleatorios unitarios, signo ya elegido
    c = torch.randn(H, d)
    c = c / c.norm(dim=-1, keepdim=True)
    dot = (u1 * c).sum(dim=-1, keepdim=True)
    c = torch.where(dot < 0, -c, c)
    # batched
    new_b = _rotate_align_columns_batched(w, u1, c)
    # loop
    new_l = torch.stack([
        _rotate_align_columns(w[h], u1[h], c[h]) for h in range(H)
    ])
    diff = (new_b - new_l).abs().max().item()
    assert diff < 1.0e-5, f"batched != loop, diff={diff:.2e}"


def test_r_div_uses_abs() -> None:
    """R_div penaliza |cos|, no cos: dos vectores antiparalelos
    deben dar misma pérdida que dos paralelos."""
    d = 16
    v = torch.randn(d)
    v = v / v.norm()
    pos = torch.stack([v, v + 0.01 * torch.randn(d)])
    pos[1] = pos[1] / pos[1].norm()
    neg = torch.stack([v, -v + 0.01 * torch.randn(d)])
    neg[1] = neg[1] / neg[1].norm()
    loss_pos = r_div(pos, theta_target_deg=60.0).item()
    loss_neg = r_div(neg, theta_target_deg=60.0).item()
    # ambos casi colineales (en signo o antisigno), pérdida similar
    assert abs(loss_pos - loss_neg) / max(loss_pos, 1.0e-6) < 0.1, (
        f"r_div no invariante al signo: {loss_pos:.4f} vs {loss_neg:.4f}"
    )
