"""g6a — certifica la demo celda a celda contra las corridas del paper.

la demo no reimplementa nada: monta un portador ligero y deja que
`aplica_gauge_ov` y `firma_exacta` corran sin un byte cambiado. eso se
comprueba con números, y admite una forma más fuerte que una banda.
las corridas publicadas guardan cada celda ---capa, fuerza, gauge---
con su semilla derivada `1000*capa + indice`, así que la demo puede
recorrer exactamente las mismas y contrastar valor contra valor.

qué se contrasta y qué no:

* **fase g (tab:gauge)** — `desv_R` y `deriva_wo` contra
  `fase_G/gauge_flip.csv`. son la misma aritmética sobre los mismos
  tensores y deben coincidir a precisión de máquina. la corrida
  publicada eleva el modelo a fp64, que es justo el régimen del
  portador.
* **tab:decision** — `par_top` y `solape_topk` contra
  `decision_rota.csv` (rebanada de la semilla 42, la única que el
  portador lleva) y `decision_rota_lm.csv` entero. la comparación es
  de identidad: mismo par, mismo solape.
* `deriva_ov` **no** entra en el contraste. la corrida publicada la
  estima sobre el circuito materializado [d, d]; la demo usa la
  factorización qr sobre [d_h, d], mejor condicionada. misma
  afirmación, estimador distinto: se reporta, no se iguala.
* el criterio «firma» del paper tampoco: exige forwards sobre el probe
  y la demo no los hace. su segundo criterio es el invariante
  estático, otro objeto.

**se corre contra el árbol desplegado, no contra el repo**: el Space
lleva su propia copia de `src/` y es esa la que atiende visitantes.

uso:
    python scripts/paridad_demo.py --arbol /tmp/space_v6
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

CSV_G = "artifacts/logs/fase_G/gauge_flip.csv"
CSV_D = "artifacts/logs/decision_rota/decision_rota.csv"
CSV_L = "artifacts/logs/decision_rota/decision_rota_lm.csv"
# el índice en esta lista es el que entra en la semilla del gauge
ESCALAS = [128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0]
ESCALA_DECISION = 8.0
SEMILLA_CKPT = 42
TOL = 1e-9


def carga_app(arbol: str):
    """importa la app desde el árbol indicado, no desde el repo.

    Args:
        arbol: raíz del árbol desplegado.

    Returns:
        el módulo de la app ya importado.
    """
    sys.path.insert(0, str(Path(arbol).resolve()))
    import app                                  # noqa: PLC0415

    return app


def paridad_fase_g(app) -> tuple[int, int, list[str]]:
    """recorre las celdas de la fase g y las contrasta una a una.

    Args:
        app: módulo de la demo.

    Returns:
        tupla (celdas comparadas, coincidencias, lista de discrepancias).
    """
    d = pd.read_csv(CSV_G)
    d = d[(d["arch"] == "vitb")
          & d["seed"].str.contains(f"seed{SEMILLA_CKPT}_")]
    p = app.portador("vitb")
    n, ok, malas = 0, 0, []
    ovs = []
    # los índices vienen de pandas como int64 y
    # `manual_seed` exige un entero de python
    for capa in tqdm([int(x) for x in sorted(d["capa"].unique())],
                     desc="fase g", leave=False):
        antes = app._v1_pesos(p, capa)
        antes_ov = app._v1_invariante(p, capa)
        for j, escala in enumerate(ESCALAS):
            fila = d[(d["capa"] == capa) & (d["escala_id"] == escala)]
            if fila.empty:
                continue
            fila = fila.iloc[0]
            q, desv = app._con_gauge("vitb", capa, 1000 * capa + j,
                                     escala, "gen")
            d_wo = float(
                (1.0 - app._cos_abs(antes, app._v1_pesos(q, capa)))
                .mean())
            ovs.append(float(
                (1.0 - app._cos_abs(antes_ov,
                                    app._v1_invariante(q, capa)))
                .mean()))
            n += 1
            e_d = abs(desv - float(fila["desv_R"]))
            e_w = abs(d_wo - float(fila["deriva_wo"]))
            if e_d < TOL and e_w < TOL:
                ok += 1
            else:
                malas.append(f"capa {capa} escala {escala:g}: "
                             f"desv_R Δ{e_d:.2e}, deriva_wo Δ{e_w:.2e}")
    print(f"  deriva_ov de la demo: máx {max(ovs):.2e} "
          f"(publicada con el otro estimador: "
          f"{d['deriva_ov'].max():.2e})")
    return n, ok, malas


def paridad_decision(app, col: str) -> tuple[int, int, list[str]]:
    """contrasta par y solape celda a celda contra la corrida publicada.

    Args:
        app: módulo de la demo.
        col: clave de columna.

    Returns:
        tupla (celdas comparadas, coincidencias, discrepancias).
    """
    if col == "vitb":
        d = pd.read_csv(CSV_D)
        d = d[d["seed"] == SEMILLA_CKPT]
    else:
        d = pd.read_csv(CSV_L)
    d = d[(d["criterio"] == "pesos") & (d["gauge_idx"] > 0)]
    p = app.portador(col)
    k = app.COLUMNAS[col]["k"]
    n, ok, malas = 0, 0, []
    for capa in tqdm([int(x) for x in sorted(d["capa"].unique())],
                     desc=f"decisión {col}", leave=False):
        par0, top0 = app.decisiones(app._v1_pesos(p, capa), k)
        for g in range(5):
            fila = d[(d["capa"] == capa) & (d["gauge_idx"] == g + 1)]
            if fila.empty:
                continue
            fila = fila.iloc[0]
            q, _ = app._con_gauge(col, capa, 1000 * capa + g,
                                  ESCALA_DECISION, "gen")
            par1, top1 = app.decisiones(app._v1_pesos(q, capa), k)
            sol = len(set(top0) & set(top1)) / k
            n += 1
            if (str(par1) == str(fila["par_top"])
                    and abs(sol - float(fila["solape_topk"])) < 1e-4):
                ok += 1
            else:
                malas.append(f"capa {capa} gauge {g + 1}: "
                             f"demo {par1}/{sol:.2f} vs publicado "
                             f"{fila['par_top']}/{fila['solape_topk']}")
    return n, ok, malas


def main() -> None:
    """corre las tres paridades y falla si alguna celda no coincide."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arbol", required=True)
    args = ap.parse_args()
    app = carga_app(args.arbol)
    dt = app.portador("vitb").blocks[0].attn.proj.weight.dtype
    print(f"[árbol]  {args.arbol}")
    print(f"[dtype]  portador en {dt}  "
          f"(la fase g publicada corre el modelo en float64)\n")

    total, malas = 0, []
    print("=== fase g · tab:gauge · ViT-B semilla 42 ===")
    n, ok, m = paridad_fase_g(app)
    print(f"  {ok}/{n} celdas idénticas a la corrida publicada")
    total += n
    malas += m
    for col in ("vitb", "pythia"):
        print(f"\n=== tab:decision · {col} ===")
        n, ok, m = paridad_decision(app, col)
        print(f"  {ok}/{n} celdas idénticas (par y solape)")
        total += n
        malas += m
    print(f"\n[resultado] {total} celdas contrastadas, "
          f"{len(malas)} discrepancias")
    for x in malas[:15]:
        print(f"  {x}")
    sys.exit(1 if malas else 0)


if __name__ == "__main__":
    main()
