"""Tests para Fase 2: ViT-B/16 + métricas de atención."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.attention import (
    attention_entropy,
    attention_entropy_per_layer,
    min_pairwise_angle_deg,
    pairwise_redundancy,
)
from src.models.vit_backbone import HeadProjections, capture_attention


def _device() -> str:
    """Selecciona cuda si está disponible."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def vit_model() -> HeadProjections:
    """ViT-B/16 sin pesos preentrenados, instanciado una vez."""
    m = HeadProjections(
        model_name="vit_base_patch16_224",
        pretrained=False,
        num_classes=10,
    ).to(_device()).eval()
    return m


def test_vit_metadata(vit_model: HeadProjections) -> None:
    """La columna ViT-B/16 expone L=12, H=12, D=768."""
    assert vit_model.num_layers == 12
    assert vit_model.num_heads == 12
    assert vit_model.embed_dim == 768
    assert vit_model.head_dim == 64


def test_head_directions_shape_and_norm(
    vit_model: HeadProjections,
) -> None:
    """head_directions devuelve (L, H, D) con normas ~1."""
    dirs = vit_model.head_directions()
    assert dirs.shape == (12, 12, 768)
    norms = dirs.norm(dim=-1).flatten()
    assert torch.allclose(
        norms, torch.ones_like(norms), atol=1.0e-5,
    )


def test_capture_attention_shape_and_softmax(
    vit_model: HeadProjections,
) -> None:
    """capture_attention captura L mapas (B, H, T, T) con filas≈1."""
    x = torch.randn(2, 3, 224, 224, device=_device())
    with capture_attention(vit_model.model) as maps:
        _ = vit_model(x)
    assert len(maps) == 12
    for m in maps:
        assert m.shape[:2] == (2, 12)
        assert m.shape[2] == m.shape[3]
        row_sums = m.sum(dim=-1)
        assert torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=1.0e-4,
        )


def test_capture_attention_restores_fused(
    vit_model: HeadProjections,
) -> None:
    """Tras salir del with, fused_attn vuelve a su valor original."""
    pre = [
        bool(blk.attn.fused_attn) for blk in vit_model.model.blocks
    ]
    x = torch.randn(1, 3, 224, 224, device=_device())
    with capture_attention(vit_model.model):
        _ = vit_model(x)
    post = [
        bool(blk.attn.fused_attn) for blk in vit_model.model.blocks
    ]
    assert pre == post


def test_pairwise_redundancy_range() -> None:
    """La redundancia cae en [0, 1] y es 0 para una base ortonormal."""
    dirs = torch.eye(8).unsqueeze(0)  # (1, 8, 8)
    r = pairwise_redundancy(dirs)
    assert r.shape == (1,)
    assert torch.allclose(r, torch.zeros(1), atol=1.0e-6)
    # caso degenerado: todas iguales → redundancia 1
    same = torch.ones(1, 4, 5)
    same = same / same.norm(dim=-1, keepdim=True)
    assert pairwise_redundancy(same).item() == pytest.approx(1.0)


def test_min_pairwise_angle_deg() -> None:
    """En una base ortonormal el ángulo mínimo es 90°."""
    dirs = torch.eye(6).unsqueeze(0)
    th = min_pairwise_angle_deg(dirs)
    assert th.item() == pytest.approx(90.0, abs=1.0e-3)


def test_attention_entropy_positive_and_max() -> None:
    """Entropía no negativa; máxima para distribución uniforme."""
    # uniforme: H = log(T)
    t = 7
    attn = torch.full((2, 4, t, t), 1.0 / t)
    h = attention_entropy(attn)
    import math
    assert torch.allclose(
        h, torch.full((4,), math.log(t)), atol=1.0e-5,
    )
    # delta: H = 0
    attn = torch.zeros(1, 1, 3, 3)
    attn[..., 0] = 1.0
    h = attention_entropy(attn)
    assert h.item() == pytest.approx(0.0, abs=1.0e-5)


def test_attention_entropy_per_layer_shape() -> None:
    """attention_entropy_per_layer devuelve (L, H)."""
    maps = [torch.softmax(torch.randn(2, 5, 9, 9), dim=-1)
            for _ in range(3)]
    H = attention_entropy_per_layer(maps)
    assert H.shape == (3, 5)
    assert (H >= 0).all()
