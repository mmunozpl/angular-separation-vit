"""Tests para Fase 4: cabeza de prototipos y métricas de openset."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.openset import auroc, oscr, roc_curve
from src.models.proto_head import PrototypeHead


def test_proto_head_fixed_is_buffer() -> None:
    """En modo fijo, los prototipos son buffer (no entrenable)."""
    code = torch.randn(5, 16)
    head = PrototypeHead(5, 16, code=code, trainable=False)
    # no debe haber parámetros entrenables
    n_params = sum(p.numel() for p in head.parameters())
    assert n_params == 0
    # el buffer existe y tiene norma 1
    assert torch.allclose(
        head.protos.norm(dim=1), torch.ones(5), atol=1.0e-5,
    )


def test_proto_head_trainable_is_parameter() -> None:
    """En modo entrenable, los prototipos son nn.Parameter."""
    head = PrototypeHead(4, 8, code=None, trainable=True)
    params = list(head.parameters())
    assert len(params) == 1
    assert params[0].shape == (4, 8)


def test_proto_head_forward_cosine_range() -> None:
    """Las salidas son cosenos, acotados en [-1, 1]."""
    code = torch.randn(3, 12)
    head = PrototypeHead(3, 12, code=code, trainable=False)
    feats = torch.randn(7, 12)
    out = head(feats)
    assert out.shape == (7, 3)
    assert (out >= -1.0 - 1.0e-5).all()
    assert (out <= 1.0 + 1.0e-5).all()


def test_proto_head_alignment() -> None:
    """Si el feature coincide con un prototipo, ese coseno = 1."""
    code = torch.eye(4)
    head = PrototypeHead(4, 4, code=code, trainable=False)
    feats = code  # cada fila = un prototipo
    out = head(feats)
    diag = torch.diagonal(out)
    assert torch.allclose(diag, torch.ones(4), atol=1.0e-5)


def test_auroc_perfectly_separable() -> None:
    """Si los positivos siempre tienen score mayor, AUROC = 1."""
    scores = torch.tensor([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    assert auroc(scores, labels) == pytest.approx(1.0)


def test_auroc_inverse_separable() -> None:
    """Orden invertido: AUROC = 0."""
    scores = torch.tensor([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    assert auroc(scores, labels) == pytest.approx(0.0)


def test_auroc_empty_class_is_nan() -> None:
    """Una clase vacía da NaN."""
    scores = torch.tensor([0.1, 0.2, 0.3])
    labels = torch.tensor([0, 0, 0])
    import math
    assert math.isnan(auroc(scores, labels))


def test_roc_curve_endpoints_and_shape() -> None:
    """La curva ROC tiene los puntos esperados en los extremos."""
    scores = torch.rand(200)
    labels = (torch.rand(200) > 0.5).long()
    roc = roc_curve(scores, labels, n_points=51)
    assert len(roc["fpr"]) == 51
    assert len(roc["tpr"]) == 51
    assert len(roc["thresholds"]) == 51
    # umbral más alto (primer punto): FPR y TPR cerca de 0
    assert roc["fpr"][0] <= 0.05
    assert roc["tpr"][0] <= 0.05
    # umbral más bajo (último): FPR y TPR cerca de 1
    assert roc["fpr"][-1] >= 0.95
    assert roc["tpr"][-1] >= 0.95


def test_oscr_bounds() -> None:
    """OSCR cae en [0, 1] para entradas arbitrarias."""
    cs = torch.randn(80) + 1.0
    ns = torch.randn(20)
    cc = (torch.rand(80) > 0.3).bool()
    val = oscr(cc, cs, ns)
    assert 0.0 <= val <= 1.0


def test_oscr_perfect_case() -> None:
    """Closed siempre acierta y supera a OOD → OSCR cerca de 1."""
    cs = torch.full((100,), 5.0)
    ns = torch.full((50,), 0.0)
    cc = torch.ones(100, dtype=torch.bool)
    val = oscr(cc, cs, ns)
    assert val > 0.98


def test_proto_head_loads_real_code(tmp_path: Path) -> None:
    """Carga el código real generado en artifacts/codes."""
    cands = list(
        (Path(__file__).resolve().parents[1] / "artifacts" / "codes")
        .glob("protos_genimage.pt")
    )
    if not cands:
        pytest.skip("código de prototipos no generado")
    blob = torch.load(cands[0], map_location="cpu",
                      weights_only=False)
    code = blob["code"]
    head = PrototypeHead(
        num_protos=code.shape[0],
        feature_dim=code.shape[1],
        code=code,
        trainable=False,
    )
    f = torch.randn(3, code.shape[1])
    out = head(f)
    assert out.shape == (3, code.shape[0])
