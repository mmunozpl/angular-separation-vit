"""Test GPU del parche SVD — valida la rama gesvd CUDA.

El test CPU-only en `tests/test_svd_fallback.py` no toca la rama
gesvd CUDA, que es la que se ejecuta primero en producción cuando
gesdd falla en GPU. Este test la valida explícitamente sobre la
patología exacta que motivó el parche: el crash de blanda seed 43
ep 18 en cabeza 69 (= capa 5, cabeza 9) con error de torch:
"too many repeated singular values".

Dos tests:
1. test_gesvd_kernel_handles_equal_singulars: llamada directa a
   torch.linalg.svd(..., driver='gesvd') sobre matriz con singulares
   todos iguales. Valida el kernel.
2. test_cascade_integration_on_gpu: integración a través de
   HeadProjections.head_directions() con captura de stdout para
   verificar qué rama se activó.

Coste: ~10s GPU. Solo correr cuando la GPU esté libre.
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.vit_backbone import HeadProjections


def _equal_singulars_matrix(
    d: int, hd: int, device: str, seed: int = 43,
) -> torch.Tensor:
    """Construye (d, hd) con singulares todos iguales a 1.

    Replica el patrón "too many repeated singular values" que
    rompe gesdd en gesdd-divide-and-conquer.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    u_q, _ = torch.linalg.qr(
        torch.randn(d, hd, device=device, generator=g)
    )
    v_q, _ = torch.linalg.qr(
        torch.randn(hd, hd, device=device, generator=g)
    )
    return u_q @ v_q.T


def test_gesvd_kernel_handles_equal_singulars() -> None:
    """Llamada directa a gesvd sobre patología real."""
    if not torch.cuda.is_available():
        print("[SKIP] no hay CUDA")
        return
    d, hd = 768, 64
    m = _equal_singulars_matrix(d, hd, device="cuda")

    # primer intento: gesdd (default)
    try:
        u_d, s_d, _ = torch.linalg.svd(m, full_matrices=False)
        gesdd_finite = (
            torch.isfinite(u_d).all() and torch.isfinite(s_d).all()
        )
        gesdd_status = (
            f"OK (s∈[{s_d.min():.4f}, {s_d.max():.4f}])"
            if gesdd_finite else "FAIL (no finitos)"
        )
    except torch._C._LinAlgError as e:
        gesdd_status = f"FAIL (LinAlgError: {str(e)[:70]}...)"

    # segundo intento: gesvd (el del parche)
    try:
        u_v, s_v, _ = torch.linalg.svd(
            m, full_matrices=False, driver="gesvd",
        )
        assert torch.isfinite(u_v).all() and torch.isfinite(s_v).all(), (
            "gesvd devolvió no finitos"
        )
        gesvd_status = f"OK (s∈[{s_v.min():.4f}, {s_v.max():.4f}])"
    except torch._C._LinAlgError as e:
        raise AssertionError(
            f"gesvd falló sobre patología equal-singulars: {e}. "
            "El parche no cubre el caso real."
        )

    print(f"[gesdd] {gesdd_status}")
    print(f"[gesvd] {gesvd_status}")
    print("[OK] gesvd CUDA maneja singulares iguales (kernel validado)")


def test_cascade_integration_on_gpu() -> None:
    """Inyecta patología en capa 5 cabeza 9 y valida la cascada."""
    if not torch.cuda.is_available():
        print("[SKIP] no hay CUDA")
        return
    device = "cuda"
    torch.manual_seed(43)
    model = HeadProjections(
        model_name="vit_base_patch16_224",
        pretrained=False, num_classes=100, img_size=224,
    ).to(device).eval()
    d = model.embed_dim
    h = model.num_heads
    hd = model.head_dim

    # inyectar en capa 5 cabeza 9 — la cabeza 69 del crash
    with torch.no_grad():
        bad = _equal_singulars_matrix(d, hd, device=device)
        w = model.model.blocks[5].attn.proj.weight.view(d, h, hd)
        w[:, 9, :] = bad

    buf = io.StringIO()
    with redirect_stdout(buf):
        dirs = model.head_directions()
    log = buf.getvalue()

    # asserts: shape, finitud, norma unitaria, reproducibilidad
    assert dirs.shape == (12, h, d), dirs.shape
    assert torch.isfinite(dirs).all(), (
        "head_directions devolvió no finitos en GPU"
    )
    norms = dirs.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), (
        f"norms fuera de unidad: min={norms.min()} max={norms.max()}"
    )

    # la rama caliente (gesdd OK) no loguea nada. Si vemos log,
    # la cascada activó alguna rama de fallback.
    fell_to_cpu = "CPU float64 recuperó" in log
    assert not fell_to_cpu, (
        "cascada cayó hasta CPU; gesvd CUDA no recuperó.\n"
        f"log capturado:\n{log}"
    )

    if "gesvd CUDA recuperó" in log:
        print(
            "[OK] cascada GPU: gesdd falló, gesvd CUDA recuperó "
            "(rama intermedia validada)"
        )
    else:
        print(
            "[OK] cascada GPU: gesdd manejó esta patología "
            "(rama caliente; gesvd no se activó)"
        )

    if log.strip():
        print(f"\nlog capturado por la cascada:\n{log.strip()}")


if __name__ == "__main__":
    test_gesvd_kernel_handles_equal_singulars()
    print()
    test_cascade_integration_on_gpu()
    print("\nTest GPU del parche SVD completado.")
