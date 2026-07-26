"""Sonda D — similitud entre cabezas por ángulos principales sobre W_O.

Cierra la objeción del rango 1: la dirección representativa
$\\bm{u}_1(\\bm{W}_O^{(h)})$ ignora 63 vectores singulares del
subespacio columna completo $768 \\times 64$. Esta sonda mide la
similitud entre cabezas por **ángulos principales entre subespacios
columna**, no por el primer vector singular, y recomputa
$r(s_{\\mathrm{func}}, s_{\\mathrm{subespacio}})$. Si sigue $\\approx 0$,
la objeción queda cerrada sin gastar GPU.

Definición usada: dado $A \\in \\mathbb{R}^{D \\times k}$ y
$B \\in \\mathbb{R}^{D \\times k}$ con bases ortonormales, los $k$
ángulos principales $\\theta_i \\in [0, \\pi/2]$ se obtienen de los
valores singulares de $Q_A^\\top Q_B$, donde $Q_A, Q_B$ son las
bases ortonormales por QR. La similitud agregada es:

$$s(A, B) = \\frac{1}{k} \\sum_{i=1}^k \\cos(\\theta_i)$$

(Grassmann mean). $s=1$ si los subespacios son idénticos, $s=0$ si
son ortogonales.

Verificación analítica antes de tocar W_O real:
1. dos subespacios idénticos → $s = 1$ (tol 1e-6)
2. dos subespacios ortogonales → $s = 0$ (tol 1e-6)
3. rotación 45° en un plano del subespacio → valor conocido

Solo tras pasar las tres, se procesa el ckpt real.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# =====================================================================
# core: similitud de subespacios por ángulos principales
# =====================================================================

def subspace_similarity(
    A: torch.Tensor, B: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    """Similitud Grassmann mean entre dos subespacios.

    Args:
        A: tensor (D, k_A) con columnas que generan el subespacio.
        B: tensor (D, k_B) idem.

    Returns:
        Tupla (s, theta_deg):
            s         media de cos(theta_i), en [0, 1].
            theta_deg ángulos principales en grados (shape min(k_A,k_B)).
    """
    if A.dim() != 2 or B.dim() != 2:
        raise ValueError("A y B deben ser 2D")
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            f"dim ambient discrepa: A.D={A.shape[0]} B.D={B.shape[0]}"
        )
    # bases ortonormales por QR
    Q_A, _ = torch.linalg.qr(A.float(), mode="reduced")
    Q_B, _ = torch.linalg.qr(B.float(), mode="reduced")
    # valores singulares de Q_A^T Q_B son los cos(theta_i)
    sv = torch.linalg.svdvals(Q_A.t() @ Q_B)
    cos_theta = sv.clamp(0.0, 1.0)
    theta_deg = torch.acos(cos_theta) * (180.0 / math.pi)
    s = float(cos_theta.mean())
    return s, theta_deg


# =====================================================================
# verificación analítica (ground truth, NO ckpt real)
# =====================================================================

def _run_analytic_checks(D: int = 768, k: int = 64, tol: float = 1e-5) -> dict:
    """Tres tests contra subespacios sintéticos con resultado conocido.

    Args:
        D: dimensión ambiente (768 para ViT-B/16).
        k: dimensión de subespacio (64 para head_dim).
        tol: tolerancia.

    Returns:
        dict con per-test pass/fail y valores observados.
    """
    torch.manual_seed(0)
    results: dict = {}

    # test 1: subespacios idénticos → s = 1
    A = torch.randn(D, k)
    s_ident, theta = subspace_similarity(A, A)
    results["identical"] = {
        "expected": 1.0, "observed": s_ident,
        "pass": abs(s_ident - 1.0) < tol,
        "max_theta_deg": float(theta.max()),
    }

    # test 2: subespacios ortogonales → s = 0
    # construir B en el complemento ortogonal de A
    Q_A, _ = torch.linalg.qr(A, mode="reduced")        # (D, k)
    # base de R^D y proyectar fuera de Q_A para obtener complemento
    full = torch.randn(D, D)
    Q_full, _ = torch.linalg.qr(full, mode="reduced")
    # tomar los últimos D-k vectores: ortogonales a Q_A por construcción
    # (después de re-ortogonalizar contra Q_A)
    B = Q_full[:, k:k + k]
    # proyectar fuera del span de Q_A
    proj = Q_A @ (Q_A.t() @ B)
    B_perp = B - proj
    Q_B, _ = torch.linalg.qr(B_perp, mode="reduced")
    s_ortho, theta_ortho = subspace_similarity(Q_A, Q_B)
    results["orthogonal"] = {
        "expected": 0.0, "observed": s_ortho,
        "pass": abs(s_ortho - 0.0) < tol,
        "min_theta_deg": float(theta_ortho.min()),
    }

    # test 3: rotación 45° en un solo plano del subespacio
    # B = A salvo que el primer vector se rota 45° hacia el segundo
    # (los otros k-1 vectores quedan idénticos)
    # En el espacio span(a1, a2), el subespacio rotado es el mismo (¡el
    # plano sigue siendo el mismo!). Solo cambia si rotamos hacia algo
    # FUERA de span(A). Construcción mejor: B = A salvo que reemplazamos
    # a1 por cos(45°)·a1 + sin(45°)·v_fuera, donde v_fuera está en el
    # complemento ortogonal. Esto crea un nuevo subespacio cuyo primer
    # ángulo principal con A es 45°.
    angle_rad = math.pi / 4.0  # 45°
    v_outside = Q_full[:, k]   # vector fuera de span(A)
    # asegurar que v_outside está realmente fuera
    v_outside = v_outside - Q_A @ (Q_A.t() @ v_outside.unsqueeze(1)).squeeze(1)
    v_outside = v_outside / v_outside.norm()
    a1_rot = (math.cos(angle_rad) * Q_A[:, 0]
              + math.sin(angle_rad) * v_outside)
    B_rot = Q_A.clone()
    B_rot[:, 0] = a1_rot
    s_rot, theta_rot = subspace_similarity(Q_A, B_rot)
    # los k ángulos principales son: 45° (uno) y 0° (k-1)
    # s = (cos45° + (k-1)·cos0°) / k = (0.7071 + 63) / 64 ≈ 0.9954
    expected_rot = (math.cos(angle_rad) + (k - 1)) / k
    results["rotation_45deg"] = {
        "expected": expected_rot, "observed": s_rot,
        "pass": abs(s_rot - expected_rot) < tol,
        "max_theta_deg": float(theta_rot.max()),
        "min_theta_deg": float(theta_rot.min()),
        "note": (
            f"esperado: 1 ángulo a 45° + {k-1} a 0°. "
            f"max_theta debería ser ≈45°"
        ),
    }

    results["all_pass"] = all(
        v["pass"] for k_, v in results.items() if isinstance(v, dict)
    )
    return results


# =====================================================================
# extracción de subespacios W_O por (capa, cabeza) y matriz por capa
# =====================================================================

def _wo_subspaces(model) -> list[list[torch.Tensor]]:
    """Subespacios columna completos de W_O por (capa, cabeza).

    Returns:
        lista de L listas de H tensores (D, d_h) cada uno.
    """
    base = model.model
    H = model.num_heads
    d_h = model.head_dim
    D = model.embed_dim
    out: list[list[torch.Tensor]] = []
    for blk in base.blocks:
        wo = blk.attn.proj.weight.detach().cpu()       # (D, D)
        # cada cabeza ocupa columnas [h*d_h, (h+1)*d_h] en W_O
        # (W_O = concat over heads, propagates back to residual)
        per_layer = [
            wo[:, h * d_h:(h + 1) * d_h].clone()        # (D, d_h)
            for h in range(H)
        ]
        out.append(per_layer)
    return out


def _subspace_matrix_layer(
    subspaces: list[torch.Tensor],
) -> np.ndarray:
    """Matriz $H \\times H$ de similitudes Grassmann entre cabezas."""
    h = len(subspaces)
    s = np.zeros((h, h), dtype=np.float64)
    for i in range(h):
        for j in range(i + 1, h):
            sim, _ = subspace_similarity(subspaces[i], subspaces[j])
            s[i, j] = s[j, i] = sim
        s[i, i] = 1.0
    return s


# =====================================================================
# smoke contra ckpt real
# =====================================================================

def _smoke_against_ckpt(ckpt_path: Path) -> dict:
    """Mide la matriz de similitud Grassmann sobre un ckpt y resume.

    Returns:
        dict con per-capa: similitud media (off-diagonal), min, max;
        y comparativa contra el primer-vector-singular (head_directions).
    """
    from src.models.vit_backbone import HeadProjections
    model = HeadProjections(
        model_name="vit_base_patch16_224",
        pretrained=False, num_classes=100, img_size=224,
    ).cpu().eval()
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = blob.get("model", blob)
    sd = {k.replace("backbone.", "", 1): v for k, v in sd.items()}
    ms = model.state_dict()
    sd = {k: v for k, v in sd.items()
          if not (k in ms and ms[k].shape != v.shape)}
    model.load_state_dict(sd, strict=False)

    subs = _wo_subspaces(model)
    L = len(subs)
    H = len(subs[0])

    # similitud por subespacios (sonda D)
    s_subspace_mean = []
    s_subspace_min = []
    s_subspace_max = []
    for li, layer in enumerate(subs):
        s_mat = _subspace_matrix_layer(layer)
        iu = np.triu_indices(H, k=1)
        off = s_mat[iu]
        s_subspace_mean.append(float(off.mean()))
        s_subspace_min.append(float(off.min()))
        s_subspace_max.append(float(off.max()))

    # similitud por primer vector singular (head_directions, rango 1)
    dirs = model.head_directions().detach().cpu()       # (L, H, D)
    s_u1_mean = []
    for li in range(L):
        layer = dirs[li]
        cos = (layer @ layer.t()).clamp(-1.0, 1.0)
        iu_pt = torch.triu_indices(H, H, offset=1)
        u1_pairs = cos[iu_pt[0], iu_pt[1]].abs()
        s_u1_mean.append(float(u1_pairs.mean()))

    return {
        "n_layers": L,
        "n_heads": H,
        "s_subspace_mean_per_layer": s_subspace_mean,
        "s_subspace_min_per_layer": s_subspace_min,
        "s_subspace_max_per_layer": s_subspace_max,
        "s_u1_abs_mean_per_layer": s_u1_mean,
        "global_subspace_mean": float(np.mean(s_subspace_mean)),
        "global_u1_abs_mean": float(np.mean(s_u1_mean)),
    }


# =====================================================================
# main
# =====================================================================

def main() -> None:
    """Punto de entrada CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        default="artifacts/checkpoints/A/attnA_base_seed42_last.pt",
    )
    parser.add_argument("--D", type=int, default=768)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument(
        "--skip-real", action="store_true",
        help="solo tests analíticos, sin tocar el ckpt",
    )
    args = parser.parse_args()

    # 1. tests analíticos contra ground truth
    print("=" * 60)
    print("VERIFICACIÓN ANALÍTICA (ground truth)")
    print("=" * 60)
    checks = _run_analytic_checks(D=args.D, k=args.k)
    for name, val in checks.items():
        if name == "all_pass":
            continue
        marca = "OK" if val["pass"] else "FALLO"
        print(f"\n[{name}] {marca}")
        for k_, v in val.items():
            if k_ != "pass":
                print(f"    {k_}: {v}")
    print(f"\nall_pass = {checks['all_pass']}")
    if not checks["all_pass"]:
        print("\n[BLOQUEO] verificación analítica falló. "
              "No procesar ckpt real.", file=sys.stderr)
        sys.exit(1)

    if args.skip_real:
        print("\n--skip-real: no se procesa ckpt.")
        return

    # 2. smoke contra ckpt real
    print("\n" + "=" * 60)
    print(f"SMOKE CONTRA CKPT REAL: {args.ckpt}")
    print("=" * 60)
    if not Path(args.ckpt).exists():
        print(f"[ERROR] ckpt no existe: {args.ckpt}", file=sys.stderr)
        sys.exit(1)
    res = _smoke_against_ckpt(Path(args.ckpt))
    print(f"\nL={res['n_layers']}  H={res['n_heads']}")
    print(f"\n{'capa':>4} {'s_subsp_mean':>14} {'s_subsp_min':>13} "
          f"{'s_subsp_max':>13} {'s_u1_abs_mean':>15}")
    for li in range(res["n_layers"]):
        print(f"{li:>4} {res['s_subspace_mean_per_layer'][li]:>14.4f} "
              f"{res['s_subspace_min_per_layer'][li]:>13.4f} "
              f"{res['s_subspace_max_per_layer'][li]:>13.4f} "
              f"{res['s_u1_abs_mean_per_layer'][li]:>15.4f}")
    print(f"\nglobal s_subspace_mean = "
          f"{res['global_subspace_mean']:.4f}")
    print(f"global s_u1_abs_mean   = {res['global_u1_abs_mean']:.4f}")


if __name__ == "__main__":
    main()
