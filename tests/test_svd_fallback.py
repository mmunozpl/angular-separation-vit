"""Test sintético del fallback SVD de head_directions().

Construye matrices W_O patológicas y verifica que la cascada
isfinite → gesdd → gesvd → CPU float64 las maneja devolviendo
finito y reproducible.

Motivación: crash de blanda seed 43 ep 18, cabeza 69 (layer 5,
head 9), torch._C._LinAlgError de gesdd batched. El ckpt ep18
no se conservó como caso de test; este sintético valida la
lógica de la cascada en ausencia del fallo real.

CPU-only: la GPU está ocupada con blanda 43 corriendo. La rama
gesvd CUDA usa el mismo torch.linalg.svd con kwarg driver='gesvd'
(cubierto upstream); aquí se valida el isfinite check, el except
chain y la rama CPU float64.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.vit_backbone import HeadProjections


def _stress_matrix(d: int, hd: int, kind: str) -> torch.Tensor:
    """Construye una matriz (d, hd) deliberadamente patológica.

    Args:
        d: dimensión de embedding.
        hd: dimensión de cabeza.
        kind: 'rank1', 'equal_singulars' o 'near_zero'.

    Returns:
        Matriz (d, hd) que típicamente rompe gesdd.
    """
    if kind == "rank1":
        u = torch.randn(d, 1)
        v = torch.randn(1, hd)
        return u @ v
    if kind == "equal_singulars":
        u_q, _ = torch.linalg.qr(torch.randn(d, hd))
        v_q, _ = torch.linalg.qr(torch.randn(hd, hd))
        return u_q @ torch.eye(hd) @ v_q.T
    if kind == "near_zero":
        u_q, _ = torch.linalg.qr(torch.randn(d, hd))
        s = torch.ones(hd) * 1e-30
        v_q, _ = torch.linalg.qr(torch.randn(hd, hd))
        return u_q @ torch.diag(s) @ v_q.T
    raise ValueError(f"kind desconocido: {kind}")


def test_cascade_handles_pathologies() -> None:
    """Inyecta patologías reales y verifica salida finita."""
    torch.manual_seed(42)
    model = HeadProjections(
        model_name="vit_base_patch16_224",
        pretrained=False, num_classes=100, img_size=224,
    ).cpu().eval()
    d = model.embed_dim
    h = model.num_heads
    hd = model.head_dim

    # inyectar tres tipos de patología en cabezas distintas
    with torch.no_grad():
        # layer 5, head 9 — réplica del crash real (batch elem 69)
        w = model.model.blocks[5].attn.proj.weight.view(d, h, hd)
        w[:, 9, :] = _stress_matrix(d, hd, "rank1")
        # layer 11, head 9 — reincidente del paper, singulares iguales
        w = model.model.blocks[11].attn.proj.weight.view(d, h, hd)
        w[:, 9, :] = _stress_matrix(d, hd, "equal_singulars")
        # layer 8, head 6 — singulares casi nulos
        w = model.model.blocks[8].attn.proj.weight.view(d, h, hd)
        w[:, 6, :] = _stress_matrix(d, hd, "near_zero")

    dirs1 = model.head_directions()
    assert dirs1.shape == (12, h, d), dirs1.shape
    assert torch.isfinite(dirs1).all(), (
        "head_directions devolvió no finitos sobre matrices patológicas"
    )
    norms = dirs1.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), (
        f"norms fuera de unidad: min={norms.min()} max={norms.max()}"
    )

    # reproducibilidad: dos llamadas consecutivas idénticas
    dirs2 = model.head_directions()
    assert torch.allclose(dirs1, dirs2, atol=1e-6), (
        "head_directions no es reproducible entre llamadas"
    )

    print("[OK] cascada SVD maneja rank1, equal_singulars, near_zero")
    print(
        f"     shape={tuple(dirs1.shape)}  "
        f"norms∈[{norms.min():.6f},{norms.max():.6f}]  "
        f"reproducible ✓"
    )


def test_isfinite_check_raises_on_nan() -> None:
    """NaN explícito en W_O debe disparar el RuntimeError limpio."""
    torch.manual_seed(0)
    model = HeadProjections(
        model_name="vit_base_patch16_224",
        pretrained=False, num_classes=100, img_size=224,
    ).cpu().eval()

    with torch.no_grad():
        model.model.blocks[7].attn.proj.weight[0, 0] = float("nan")

    try:
        _ = model.head_directions()
    except RuntimeError as e:
        assert "contiene NaN/Inf" in str(e), (
            f"mensaje inesperado: {e}"
        )
        print("[OK] isfinite check detectó NaN y abortó con mensaje claro")
        return
    raise AssertionError("head_directions debió detectar el NaN")


def test_isfinite_check_raises_on_inf() -> None:
    """Inf en W_O también debe disparar el RuntimeError."""
    torch.manual_seed(0)
    model = HeadProjections(
        model_name="vit_base_patch16_224",
        pretrained=False, num_classes=100, img_size=224,
    ).cpu().eval()

    with torch.no_grad():
        model.model.blocks[3].attn.proj.weight[5, 7] = float("inf")

    try:
        _ = model.head_directions()
    except RuntimeError as e:
        assert "contiene NaN/Inf" in str(e)
        print("[OK] isfinite check detectó Inf y abortó con mensaje claro")
        return
    raise AssertionError("head_directions debió detectar el Inf")


if __name__ == "__main__":
    test_cascade_handles_pathologies()
    test_isfinite_check_raises_on_nan()
    test_isfinite_check_raises_on_inf()
    print("\nTodos los tests del parche SVD pasados.")
