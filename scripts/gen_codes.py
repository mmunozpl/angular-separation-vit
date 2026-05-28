"""Genera y cachea los códigos esféricos requeridos por el proyecto.

Para cada código se guardan, además del tensor, las métricas
necesarias para reconstruir Tabla 1 y las figuras del descenso:

- traza (paso, energía, theta_min) durante el descenso;
- estadísticos del histograma de cosenos por par;
- energía de Riesz final y, cuando se conoce, energía canónica de
  referencia;
- theta_min, theta_max y percentiles del ángulo entre pares.

Estos artefactos viven en ``artifacts/codes/<name>.pt`` y la tabla
resumen se vuelca en ``artifacts/logs/codes.csv``.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import torch

# se permite ejecutar el script desde la raíz del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.codes.canonical import canonical_kissing
from src.codes.generate import generate_code
from src.codes.riesz import riesz_energy
from src.codes.validate import (
    KISSING_NUMBER,
    min_angle_deg,
    validate_kissing,
)
from src.config import load_config
from src.seed import set_seed


def _pair_cosines(x: torch.Tensor) -> torch.Tensor:
    """Devuelve los cosenos de todos los pares (i<j) de x."""
    k = x.shape[0]
    g = x @ x.t()
    idx_i, idx_j = torch.triu_indices(k, k, offset=1, device=g.device)
    return g[idx_i, idx_j]


def _angle_stats(x: torch.Tensor) -> dict:
    """Estadísticos de ángulos entre pares (en grados)."""
    cos = _pair_cosines(x).clamp(-1.0, 1.0)
    ang = torch.acos(cos) * (180.0 / math.pi)
    q = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=ang.device)
    qs = torch.quantile(ang.float(), q)
    return {
        "n_pairs": int(ang.numel()),
        "theta_min": float(qs[0].item()),
        "theta_q25": float(qs[1].item()),
        "theta_median": float(qs[2].item()),
        "theta_q75": float(qs[3].item()),
        "theta_max": float(qs[4].item()),
        "theta_mean": float(ang.mean().item()),
        "cos_min": float(cos.min().item()),
        "cos_max": float(cos.max().item()),
        "cos_mean": float(cos.mean().item()),
    }


def _cos_histogram(
    x: torch.Tensor, n_bins: int = 200
) -> tuple[list[float], list[int]]:
    """Histograma de cosenos entre pares en [-1, 1]."""
    cos = _pair_cosines(x).clamp(-1.0, 1.0).cpu()
    hist = torch.histc(cos, bins=n_bins, min=-1.0, max=1.0)
    edges = torch.linspace(-1.0, 1.0, n_bins + 1).tolist()
    return edges, hist.long().tolist()


def main() -> None:
    """Punto de entrada del script de generación de códigos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("artifacts/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    s = float(cfg["riesz"]["exponent_s"])
    opt = cfg["optimization"]
    val = cfg["validation"]

    csv_path = log_dir / "codes.csv"
    with csv_path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow([
            "name", "d", "K", "theta_min_deg", "theta_max_deg",
            "theta_median_deg", "theta_mean_deg",
            "energy_riesz_s1", "energy_canonical",
            "kissing_ok", "n_pairs",
        ])

    summary: list[tuple[str, int, int, float, str]] = []
    for entry in cfg["dimensions"]:
        d = int(entry["d"])
        k = int(entry["k"])
        name = str(entry["name"])
        init = canonical_kissing(d, k)
        code, trace = generate_code(
            k=k,
            d=d,
            s=s,
            steps=int(opt["steps"]),
            lr=float(opt["lr"]),
            log_every=int(opt["log_every"]),
            device=str(opt["device"]),
            seed=int(cfg["seed"]),
            init=init,
        )
        stats = _angle_stats(code)
        hist_edges, hist_counts = _cos_histogram(code, n_bins=200)
        energy_final = float(
            riesz_energy(code, s=1.0).item()
        )
        energy_canonical: float | None = None
        if init is not None:
            energy_canonical = float(
                riesz_energy(init.to(code.device), s=1.0).item()
            )

        # validación frente a kissing number conocido
        if d in KISSING_NUMBER and k == KISSING_NUMBER[d]:
            ok, _ = validate_kissing(
                code, d=d, k=k,
                expected_deg=float(val["expected_theta_deg"]),
                tol_deg=float(val["tolerance_deg"]),
            )
            tag = "ok" if ok else "FALLA"
        else:
            ok = None
            tag = "sin-ref"

        path = out_dir / f"{name}.pt"
        torch.save(
            {
                "code": code.cpu(),
                "d": d,
                "k": k,
                "s": s,
                "trace": trace,
                "stats": stats,
                "hist_cos_edges": hist_edges,
                "hist_cos_counts": hist_counts,
                "energy_final": energy_final,
                "energy_canonical": energy_canonical,
                "kissing_ok": ok,
                "init_from_canonical": init is not None,
            },
            path,
        )
        with csv_path.open("a", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow([
                name, d, k,
                f"{stats['theta_min']:.6f}",
                f"{stats['theta_max']:.6f}",
                f"{stats['theta_median']:.6f}",
                f"{stats['theta_mean']:.6f}",
                f"{energy_final:.6f}",
                (
                    f"{energy_canonical:.6f}"
                    if energy_canonical is not None else ""
                ),
                "" if ok is None else ("1" if ok else "0"),
                stats["n_pairs"],
            ])

        summary.append((name, d, k, stats["theta_min"], tag))

        # 15 direcciones aleatorias del código guardado
        n_show = min(15, k)
        idx = torch.randperm(k)[:n_show]
        n_dims_show = min(8, d)
        print(f"\n[{name}] guardado en {path}")
        print(
            f"  d={d}  K={k}  theta_min={stats['theta_min']:.3f}°  "
            f"E_final={energy_final:.3f}  ({tag})"
        )
        if energy_canonical is not None:
            print(
                f"  energia canonica de referencia = "
                f"{energy_canonical:.3f}"
            )
        print(
            f"  ang: median={stats['theta_median']:.2f}°  "
            f"mean={stats['theta_mean']:.2f}°  "
            f"max={stats['theta_max']:.2f}°"
        )
        print(f"  15 direcciones (primeras {n_dims_show} dims):")
        for j in idx.tolist():
            row = code[j].cpu().tolist()[:n_dims_show]
            shown = ", ".join(f"{v:+.3f}" for v in row)
            ell = "..." if d > n_dims_show else ""
            print(f"    i={j:6d}  [{shown}{ell}]")

    print("\nresumen:")
    print(
        f"  {'nombre':28s} {'d':>5s} {'K':>8s} "
        f"{'theta_min':>11s}  {'estado':>7s}"
    )
    for name, d, k, th, tag in summary:
        print(
            f"  {name:28s} {d:>5d} {k:>8d} {th:>10.3f}°  "
            f"{tag:>7s}"
        )
    print(f"\nresumen tabular en {csv_path}")


if __name__ == "__main__":
    main()
