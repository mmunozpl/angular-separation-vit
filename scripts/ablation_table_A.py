"""Tabla de ablación de la contribución 1: base vs blanda vs dura.

Lee los run.csv finales de las 15 corridas de A (3 variantes × 5 semillas)
y produce la tabla del paper: media±std de val_top1, θ_min alcanzado y
redundancia entre cabezas, por variante. La lectura es el contraste
central: ambas variantes reguladas alcanzan el óptimo geométrico (θ*≈
84,78°), pero la blanda sin coste de exactitud y la dura pagando ~4,7 pp.

Salida: `artifacts/tables/ablation_A.csv`.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

CSV_OUT = Path("artifacts/tables/ablation_A.csv")
VARIANTS = ["base", "blanda", "dura"]
SEEDS = [42, 43, 44, 45, 46]
# columnas de run.csv (índice 0-based)
COL = {"val_top1": 15, "theta_min_mean": 9, "redundancy_mean": 7}


def _final_row(
    log_root: Path, variant: str, seed: int,
) -> list[str] | None:
    """Última fila de run.csv de una corrida, o None si no existe."""
    f = log_root / f"attnA_{variant}_seed{seed}" / "run.csv"
    if not f.exists():
        return None
    rows = f.read_text().splitlines()
    if len(rows) < 2:
        return None
    return rows[-1].split(",")


def main() -> None:
    """Agrega las corridas de una arquitectura y vuelca/merge la tabla."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="vitb",
                    help="etiqueta de arquitectura (vitb/vitl/dinov2); "
                         "lee de artifacts/logs/<arch>_clean")
    arch = ap.parse_args().arch
    log_root = Path(f"artifacts/logs/{arch}_clean")

    table = {}
    for v in VARIANTS:
        vals = {k: [] for k in COL}
        n_done = 0
        for s in SEEDS:
            row = _final_row(log_root, v, s)
            if row is None:
                continue
            n_done += 1
            for k, idx in COL.items():
                vals[k].append(float(row[idx]))
        if n_done > 0:
            table[v] = {
                # ddof=1: std muestral entre semillas, coherente con el
                # resto de agregadores
                k: (float(np.mean(arr)), float(np.std(arr, ddof=1)))
                for k, arr in vals.items()
            }
        else:
            table[v] = {k: (float("nan"), float("nan")) for k in COL}
        table[v]["n"] = n_done

    base_val = table["base"]["val_top1"][0]
    header = [
        "arch", "variante", "n", "val_top1_mean", "val_top1_std",
        "coste_vs_base_pp", "theta_min_mean", "theta_min_std",
        "redundancy_mean", "redundancy_std",
    ]
    nuevas = []
    for v in VARIANTS:
        t = table[v]
        coste = (base_val - t["val_top1"][0]) * 100.0
        nuevas.append([
            arch, v, t["n"],
            f"{t['val_top1'][0]:.4f}", f"{t['val_top1'][1]:.4f}",
            f"{coste:.2f}",
            f"{t['theta_min_mean'][0]:.3f}",
            f"{t['theta_min_mean'][1]:.3f}",
            f"{t['redundancy_mean'][0]:.4f}",
            f"{t['redundancy_mean'][1]:.4f}",
        ])

    # merge-por-arch: conserva las filas de otras arquitecturas y
    # reemplaza las de esta; el csv canónico acumula todas con su columna
    # arch, de modo que el agregador del .tex lee un solo fichero
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    previas = []
    if CSV_OUT.exists():
        with CSV_OUT.open(newline="") as fh:
            for r in csv.reader(fh):
                if r and r[0] not in ("arch", arch):
                    previas.append(r)
    with CSV_OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(previas)
        w.writerows(nuevas)

    # impresión legible
    print(f"[guardado] {CSV_OUT}  (arch={arch})\n")
    print(f"{'variante':>8} {'n':>2} {'val_top1':>16} {'coste_pp':>9} "
          f"{'θ_min':>16} {'redundancia':>16}")
    for v in VARIANTS:
        t = table[v]
        coste = (base_val - t["val_top1"][0]) * 100.0
        print(
            f"{v:>8} {t['n']:>2} "
            f"{t['val_top1'][0]:.4f}±{t['val_top1'][1]:.4f} "
            f"{coste:>8.2f} "
            f"{t['theta_min_mean'][0]:.3f}±{t['theta_min_mean'][1]:.3f} "
            f"{t['redundancy_mean'][0]:.4f}±{t['redundancy_mean'][1]:.4f}"
        )
    print("\nlectura: ambas reguladas alcanzan θ*≈84,78°; la blanda sin "
          "coste, la dura pagando el coste de imponerlo por construcción.")


if __name__ == "__main__":
    main()
