"""e1 — nulo aleatorio de la poda a presupuesto fijo (tab:poda).

ejecuta el prerregistro `Paper_X/prereg_E1_nulo_poda.md` (respaldo
c7773e4, 21-09-2026): cien sorteos por semilla del criterio
aleatorio, con el mismo presupuesto y el mismo mecanismo que
`poda_criterio.py`, del que se reutilizan el cargador, la selección y
la poda. gate de reproducción antes de leer nada más: la media de los
sorteos 42/43/44 coincide con la celda «aleatorio» de tab:poda a
1e-3 en top-1 (0,1 pp).

uso:
    python scripts/poda_nulo.py [--seeds 42 ...] [--n-sorteos 100]
"""

import argparse
import csv
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from scripts.poda_criterio import (K_PODA, SORTEOS, podar,  # noqa: E402
                                   restaurar, seleccion_aleatoria,
                                   top1, val_loader_completo)
from scripts.poda_criterio import SALIDA as CSV_CRITERIO  # noqa: E402
from scripts.run_fase_G import carga_modelo  # noqa: E402

SALIDA = "artifacts/logs/poda_criterio/poda_nulo_100.csv"
RESUMEN = "artifacts/logs/poda_criterio/poda_nulo_resumen.csv"
GATE_ALEATORIO_PP = 3.25     # celda «aleatorio (suelo)» de tab:poda
GATE_TOL_PP = 0.1            # 1e-3 en top-1
PCTS = [2.5, 5, 50, 95, 97.5]


def caidas_criterio(seeds: list[int]) -> dict[str, dict[int, float]]:
    """caída publicada de pesos y firma por semilla, del csv de b."""
    out: dict[str, dict[int, float]] = {"pesos": {}, "firma": {}}
    with open(CSV_CRITERIO, newline="") as fh:
        for fila in csv.DictReader(fh):
            s, crit = int(fila["seed"]), fila["criterio"]
            if s in seeds and crit in out:
                out[crit][s] = float(fila["caida_pp"])
    return out


def main() -> None:
    """corre los sorteos, pasa el gate y vuelca csv y resumen."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 43, 44, 45, 46])
    parser.add_argument("--n-sorteos", type=int, default=100)
    args = parser.parse_args()

    disp = "cuda" if torch.cuda.is_available() else "cpu"
    val = val_loader_completo()
    filas: list[dict] = []
    for seed in tqdm(args.seeds, desc="semillas"):
        ckpt = (f"artifacts/checkpoints/vitb_clean/"
                f"attnA_base_seed{seed}_last.pt")
        modelo = carga_modelo(ckpt).to(disp).eval()
        n_capas = len(modelo.blocks)
        base = top1(modelo, val, disp)
        acum: list[float] = []
        for s in tqdm(range(args.n_sorteos), desc="sorteos",
                      leave=False):
            sel = seleccion_aleatoria(n_capas, K_PODA, s)
            backup = podar(modelo, sel)
            podado = top1(modelo, val, disp)
            restaurar(modelo, backup)
            caida = round(100 * (base - podado), 3)
            acum.append(caida)
            filas.append({"semilla": seed, "sorteo": s,
                          "top1_podado": round(podado, 4),
                          "caida_pp": caida})
            if (s + 1) % 10 == 0:
                print(f"  seed {seed} sorteo {s + 1}: caída media "
                      f"{statistics.mean(acum):.3f} pp", flush=True)
        del modelo
        torch.cuda.empty_cache()

    out = Path(SALIDA)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        wr.writeheader()
        wr.writerows(filas)
    print(f"[guardado] {out}  ({len(filas)} filas)")

    # gate de reproducción: sorteos 42/43/44, la fila de regresión
    reg = [statistics.mean(f["caida_pp"] for f in filas
                           if f["semilla"] == seed
                           and f["sorteo"] in SORTEOS)
           for seed in args.seeds]
    media_reg = statistics.mean(reg)
    print(f"[gate] sorteos {SORTEOS}: {media_reg:.3f} pp frente a "
          f"{GATE_ALEATORIO_PP} publicado")
    if abs(media_reg - GATE_ALEATORIO_PP) > GATE_TOL_PP:
        raise SystemExit("gate de reproducción fallado: sin veredicto")

    # estadísticos por semilla y agregados
    crit = caidas_criterio(args.seeds)
    todas = np.array([f["caida_pp"] for f in filas])
    res: list[dict] = []
    bate = {"pesos": 0, "firma": 0}
    dana = {"pesos": 0, "firma": 0}
    for seed in args.seeds:
        c = np.array([f["caida_pp"] for f in filas
                      if f["semilla"] == seed])
        p = {f"p{q}": float(np.percentile(c, q)) for q in PCTS}
        fila = {"semilla": seed, "media": float(c.mean()),
                "desv": float(c.std(ddof=1)), **p}
        for k in ("pesos", "firma"):
            v = crit[k][seed]
            fila[f"{k}_caida"] = v
            fila[f"{k}_pct_conjunto"] = float((todas < v).mean() * 100)
            bate[k] += int(v < p["p5"])
            dana[k] += int(v > p["p95"])
        res.append(fila)
    agg = {"semilla": "agregado",
           "media": float(np.mean([r["media"] for r in res])),
           "desv": float(np.mean([r["desv"] for r in res]))}
    for q in PCTS:
        agg[f"p{q}"] = float(np.percentile(todas, q))
    for k in ("pesos", "firma"):
        vals = [crit[k][s] for s in args.seeds]
        agg[f"{k}_caida"] = float(np.mean(vals))
        agg[f"{k}_pct_conjunto"] = float(
            np.mean([(todas < v).mean() * 100 for v in vals]))
    res.append(agg)
    with open(RESUMEN, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(res[-1].keys()))
        wr.writeheader()
        wr.writerows(res)
    print(f"[guardado] {RESUMEN}")
    n = len(args.seeds)
    for k in ("pesos", "firma"):
        print(f"{k:>6}: bate al azar en {bate[k]}/{n} semillas "
              f"(p5); daña más que el azar en {dana[k]}/{n} (p95); "
              f"percentil conjunto {agg[k + '_pct_conjunto']:.1f}")
    print(f"nulo: media {agg['media']:.3f} pp, p5 {agg['p5']:.3f}, "
          f"p95 {agg['p95']:.3f}")

    # 15 observaciones aleatorias del artefacto
    rng = random.Random(0)
    for fila in rng.sample(filas, min(15, len(filas))):
        print(fila)


if __name__ == "__main__":
    main()
