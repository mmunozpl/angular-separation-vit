"""b1 — curva de inestabilidad decisional frente a la fuerza del gauge.

materializa, a cada una de las siete fuerzas del barrido de la fase G,
la decisión que la práctica toma por la lectura estática v1(w_o) ---par
más redundante y top-3 de poda--- y mide con qué frecuencia cambia
respecto a la decisión sin gauge. cualifica por fuerza la frase «la
decisión se rompe»: el radio local no es cero, de modo que en régimen
pequeño la decisión sobrevive, y la curva localiza dónde arranca la
inestabilidad.

preregistro en `Paper_X/prereg_B1_inestabilidad.md` (congelado
26-08-2026, umbrales ratificados 28-08). estático: cero forwards, solo
álgebra sobre los pesos.

dos gates de reproducción anteceden a todo estadístico ---el script
aborta sin imprimir si alguno falla---:

1. la decisión base sin gauge clava, por (semilla, capa), la
   almacenada en `decision_rota.csv` (gauge_idx 0, criterio pesos);
2. el punto de la curva a escala_id 8,0 usa la r de semilla
   1000*capa+4, que es la réplica g=4 de `decision_rota.py`: su
   decisión clava la fila gauge_idx 5 del mismo csv.

uso:
    python scripts/curva_inestabilidad.py [--disp cpu|cuda]
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from scripts.run_fase_G import ESCALAS, carga_modelo  # noqa: E402
from src.firma_funcional import w_o_por_cabeza  # noqa: E402
from src.gauge_flip import aplica_gauge_ov  # noqa: E402
from src.nucleo_lectura import decisiones, firma_exacta  # noqa: E402

SEMILLAS = [42, 43, 44, 45, 46]
ALMACENADO = "artifacts/logs/decision_rota/decision_rota.csv"
SALIDA = "artifacts/logs/curva_inestabilidad/curva_inestabilidad.csv"
RESUMEN = ("artifacts/logs/curva_inestabilidad/"
           "curva_inestabilidad_resumen.csv")
N_CABEZAS, DIM_CABEZA = 12, 64
K_PODA = 3
ESCALA_ANCLA = 8.0   # el punto que solapa con decision_rota (g=4)
IDX_ANCLA = 5        # su fila almacenada: base=0, réplicas g+1
DELTA_PEQUENO = 0.13  # frontera del régimen pequeño (preregistro)
FLIP_PEQUENO = 0.05   # veredicto de seguridad en ese régimen
FLIP_ARRANQUE = 0.40  # umbral de delta*, heredado de tab:decision


@torch.no_grad()
def decision_por_pesos(modelo, capa: int) -> tuple[tuple, list]:
    """decisión de poda por la lectura estática v1(w_o) de una capa.

    Args:
        modelo: el vit.
        capa: índice de la capa.

    Returns:
        tupla (par más redundante, top-3 por redundancia media).
    """
    w_o = w_o_por_cabeza(modelo, capa, N_CABEZAS, DIM_CABEZA)
    return decisiones(firma_exacta(w_o, "der"), K_PODA)


@torch.no_grad()
def instantanea(modelo, capa: int) -> list[torch.Tensor]:
    """clona los pesos que el gauge toca, para restaurarlos después.

    Args:
        modelo: el vit.
        capa: índice de la capa intervenida.

    Returns:
        lista de clones [w_qkv, b_qkv | None, w_proj].
    """
    attn = modelo.blocks[capa].attn
    sesgo = None if attn.qkv.bias is None else attn.qkv.bias.clone()
    return [attn.qkv.weight.clone(), sesgo, attn.proj.weight.clone()]


@torch.no_grad()
def restaura(modelo, capa: int, copia: list) -> None:
    """devuelve la capa a los pesos previos al gauge.

    Args:
        modelo: el vit.
        capa: índice de la capa intervenida.
        copia: la lista que devolvió `instantanea`.
    """
    attn = modelo.blocks[capa].attn
    attn.qkv.weight.copy_(copia[0])
    if copia[1] is not None:
        attn.qkv.bias.copy_(copia[1])
    attn.proj.weight.copy_(copia[2])


def gate_reproduccion(filas: list[dict]) -> None:
    """comprueba los dos gates contra las decisiones almacenadas.

    Args:
        filas: las filas materializadas por el barrido.

    Raises:
        SystemExit: si base o ancla discrepan en alguna combinación;
            el barrido no imprime estadístico alguno.
    """
    if not os.path.exists(ALMACENADO):
        raise SystemExit(f"sin csv almacenado: {ALMACENADO}")
    alm = pd.read_csv(ALMACENADO)
    alm = alm[alm["criterio"] == "pesos"]
    clave = ["seed", "capa"]
    base_alm = alm[alm["gauge_idx"] == 0].set_index(clave)
    ancla_alm = alm[alm["gauge_idx"] == IDX_ANCLA].set_index(clave)
    fallos: list[str] = []
    vistos_base: set[tuple[int, int]] = set()
    for f in filas:
        k = (f["seed"], f["capa"])
        if k not in vistos_base:
            vistos_base.add(k)
            ref = base_alm.loc[k]
            if (f["par_base"] != ref.par_top
                    or f["topk_base"] != ref.topk):
                fallos.append(f"base {k}: {f['par_base']}/"
                              f"{f['topk_base']} vs {ref.par_top}/"
                              f"{ref.topk}")
        if f["escala_id"] == ESCALA_ANCLA:
            ref = ancla_alm.loc[k]
            if f["par_top"] != ref.par_top or f["topk"] != ref.topk:
                fallos.append(f"ancla {k}: {f['par_top']}/{f['topk']}"
                              f" vs {ref.par_top}/{ref.topk}")
    if fallos:
        print(f"[gate] {len(fallos)} discrepancias; las 10 primeras:")
        for linea in fallos[:10]:
            print("  " + linea)
        raise SystemExit("gate de reproducción fallado: sin veredicto")
    n_ancla = sum(f["escala_id"] == ESCALA_ANCLA for f in filas)
    print(f"[gate] base {len(vistos_base)}/60 y ancla {n_ancla}/60 "
          f"clavan lo almacenado")


def barrido(disp: str) -> list[dict]:
    """materializa las 420 decisiones capa x semilla x fuerza.

    Args:
        disp: dispositivo del cómputo.

    Returns:
        lista de filas, una por combinación.
    """
    filas: list[dict] = []
    for seed in tqdm(SEMILLAS, desc="semillas"):
        ckpt = (f"artifacts/checkpoints/vitb_clean/"
                f"attnA_base_seed{seed}_last.pt")
        modelo = carga_modelo(ckpt).to(disp).eval()
        for capa in tqdm(range(len(modelo.blocks)), desc="capas",
                         leave=False):
            par0, topk0 = decision_por_pesos(modelo, capa)
            copia = instantanea(modelo, capa)
            for j, escala in enumerate(ESCALAS):
                # se siembra como la fase g: una r por capa x fuerza
                desv = aplica_gauge_ov(modelo, capa, N_CABEZAS,
                                       DIM_CABEZA,
                                       semilla=1000 * capa + j,
                                       escala_id=escala)
                par1, topk1 = decision_por_pesos(modelo, capa)
                restaura(modelo, capa, copia)
                filas.append({
                    "seed": seed, "capa": capa, "fuerza_idx": j,
                    "escala_id": escala, "desv_R": desv,
                    "par_base": str(par0), "topk_base": str(topk0),
                    "par_top": str(par1), "topk": str(topk1),
                    "cambia_par": par1 != par0,
                    "solape_topk": round(
                        len(set(topk0) & set(topk1)) / K_PODA, 4)})
        del modelo
        if disp == "cuda":
            torch.cuda.empty_cache()
    return filas


def resume(df: pd.DataFrame) -> pd.DataFrame:
    """agrega por fuerza con banda entre semillas.

    Args:
        df: el barrido completo.

    Returns:
        tabla por fuerza: delta_R, flip y solape con su banda.
    """
    por_semilla = df.groupby(["escala_id", "seed"])[
        ["cambia_par", "solape_topk"]].mean().reset_index()
    agg = por_semilla.groupby("escala_id").agg(
        flip=("cambia_par", "mean"), flip_std=("cambia_par", "std"),
        solape=("solape_topk", "mean"),
        solape_std=("solape_topk", "std"))
    agg["delta_R"] = df.groupby("escala_id").desv_R.mean()
    agg["n"] = df.groupby("escala_id").size()
    return agg.sort_values("delta_R").reset_index()


def veredictos(res: pd.DataFrame) -> None:
    """lee la curva contra los umbrales ratificados el 28-08.

    Args:
        res: la tabla resumida por fuerza.
    """
    pequeno = res[res.delta_R <= DELTA_PEQUENO]
    seguro = bool((pequeno.flip < FLIP_PEQUENO).all())
    print(f"\nrégimen pequeño (delta_R <= {DELTA_PEQUENO}): "
          f"flip máx {pequeno.flip.max():.3f} -> "
          f"{'seguridad' if seguro else 'SIN seguridad'} "
          f"(umbral {FLIP_PEQUENO})")
    arranque = res[res.flip >= FLIP_ARRANQUE]
    if len(arranque):
        fila = arranque.iloc[0]
        print(f"delta* (primera fuerza con flip >= "
              f"{FLIP_ARRANQUE:.0%}): delta_R = {fila.delta_R:.3f} "
              f"(escala_id {fila.escala_id:.0f}), flip {fila.flip:.3f}")
    else:
        print(f"delta*: ninguna fuerza alcanza flip >= "
              f"{FLIP_ARRANQUE:.0%}; se reporta tal cual")
    monot = res.flip.is_monotonic_increasing
    print(f"monotonía del flip con delta_R: "
          f"{'sí' if monot else 'NO ---se reporta tal cual'}")


def main() -> None:
    """corre el barrido, pasa los gates y vuelca la curva."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disp", default="cpu",
                        choices=["cpu", "cuda"],
                        help="dispositivo; el preregistro declara cpu")
    args = parser.parse_args()
    filas = barrido(args.disp)
    gate_reproduccion(filas)          # nada se imprime si falla
    df = pd.DataFrame(filas)
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    df.to_csv(SALIDA, index=False)                # se guarda en csv
    res = resume(df)
    res.to_csv(RESUMEN, index=False)              # ídem el resumen
    print(f"\n[guardado] {SALIDA} ({len(df)} filas) y {RESUMEN}")
    print(df.sample(15, random_state=0).to_string())  # 15 obs
    print("\ndelta_R | escala | flip (m±s) | solape top-3 (m±s) | n")
    for _, f in res.iterrows():
        print(f"{f.delta_R:7.3f} | {f.escala_id:6.0f} | "
              f"{f.flip:.3f}±{f.flip_std:.3f} | "
              f"{f.solape:.3f}±{f.solape_std:.3f} | {int(f.n)}")
    veredictos(res)


if __name__ == "__main__":
    main()
