"""derivación y verificación de los datos de la fig:inversion del paper.

la figura del paper (fig:inversion) es nativa en pgfplots dentro del
.tex; este script NO la alimenta. su papel es la TRAZA: computa, por
capa y como media±std entre las 5 semillas, la correlación de spearman
de s_func con tres candidatas geométricas (subespacio de W_O a k=64,
dirección dominante u1 a k=1, e interacción Q·K) más la Q·K blanda, que
son los números incrustados a mano en el pgfplots; y deja una versión
matplotlib en artifacts/figs como vista rápida. así los valores del tex
son reproducibles desde el csv. fuente: artifacts/tables/
dissociation_D/by_layer.csv (el mismo que tab:inversion).

uso:
    python scripts/fig_inversion.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.seed import set_seed

CSV = "artifacts/tables/dissociation_D/by_layer.csv"
SALIDA = "artifacts/figs/inversion_premise"
ARCH = "vitb"  # arquitectura a leer del csv (filtra por columna arch)


def media_std_por_capa(
    rows: list[dict], variante: str, col: str, k: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """media±std entre semillas, por capa, de una columna del csv.

    args:
        rows: filas del csv como dicts.
        variante: 'base' o 'blanda'.
        col: columna a agregar (r_WO, r_QK, ...).
        k: filtro de k; None no filtra (para columnas sin k, como r_QK
            que es constante en k, se fija uno cualquiera fuera).

    returns:
        tupla (capas, media, std) como arrays ordenados por capa.
    """
    por_capa: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        # filtra por arquitectura: con 2+ archs en el csv, no filtrar
        # promediaría capas a través de arquitecturas (fallo silencioso)
        if r.get("arch", ARCH) != ARCH:
            continue
        if r["variant"] != variante:
            continue
        if k is not None and int(r["k"]) != k:
            continue
        por_capa[int(r["layer"])].append(float(r[col]))
    capas = np.array(sorted(por_capa))
    media = np.array([np.mean(por_capa[c]) for c in capas])
    std = np.array([np.std(por_capa[c], ddof=1) for c in capas])
    return capas, media, std


def main() -> None:
    """construye y guarda la figura de la inversión hacia Q·K."""
    set_seed(0)
    rows = list(csv.DictReader(open(CSV)))

    # tres candidatas en base + Q·K blanda para la independencia.
    # r_QK no depende de k: se fija k=64 para tomar una fila por
    # (semilla, capa)
    cx, wo_m, wo_s = media_std_por_capa(rows, "base", "r_WO", 64)
    _, u1_m, u1_s = media_std_por_capa(rows, "base", "r_WO", 1)
    _, qk_m, qk_s = media_std_por_capa(rows, "base", "r_QK", 64)
    _, qkb_m, qkb_s = media_std_por_capa(rows, "blanda", "r_QK", 64)

    fig, ax = plt.subplots(figsize=(6.2, 3.9))

    def banda(x, m, s, color, label, **kw):
        # curva con su banda de ±1 std entre semillas
        ax.plot(x, m, color=color, label=label, **kw)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.15)

    banda(cx, wo_m, wo_s, "#1f77b4",
          r"$r_{W_O}$ subesp. ($k{=}64$)", marker="o", ms=4)
    banda(cx, u1_m, u1_s, "#2ca02c",
          r"$r_{W_O}$ dir. $u_1$ ($k{=}1$)", marker="s", ms=3,
          ls="--")
    banda(cx, qk_m, qk_s, "#d62728",
          r"$r_{Q\cdot K}$ (base)", marker="^", ms=4, lw=2)
    # blanda superpuesta, tenue: el cruce no depende del regularizador
    ax.plot(cx, qkb_m, color="#d62728", alpha=0.5, lw=1.2, ls=":",
            marker="^", ms=3, label=r"$r_{Q\cdot K}$ (blanda)")

    # la costura de la capa 5 y los baselines aleatorios
    ax.axvline(5, color="0.4", lw=1, ls="-.", alpha=0.7)
    ax.text(5.1, ax.get_ylim()[1] * 0.92, "costura (capa 5)",
            fontsize=8, color="0.3")
    ax.axhline(0.0, color="0.7", lw=0.8, ls=":")
    ax.axhline(0.25, color="#1f77b4", lw=0.8, ls=":", alpha=0.5)

    ax.set_xlabel("capa")
    ax.set_ylabel(r"Spearman $r$ con $s_{\mathrm{func}}$")
    ax.set_xticks(cx)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    out = Path(SALIDA)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=150)
    plt.close(fig)
    print(f"figura guardada: {out}.pdf y {out}.png", flush=True)

    # 15 observaciones aleatorias del dato que sostiene la figura
    rng = np.random.default_rng(0)
    base_rows = [r for r in rows
                 if r.get("arch", ARCH) == ARCH
                 and r["variant"] == "base" and int(r["k"]) in (1, 64)]
    idx = rng.choice(len(base_rows), size=15, replace=False)
    print("\n=== 15 observaciones aleatorias (by_layer.csv, base) ===")
    for i in idx:
        r = base_rows[i]
        print(f"  seed {r['seed']} capa {int(r['layer']):2d} "
              f"k={int(r['k']):2d}  r_WO={float(r['r_WO']):+.3f}  "
              f"r_QK={float(r['r_QK']):+.3f}", flush=True)


if __name__ == "__main__":
    main()
