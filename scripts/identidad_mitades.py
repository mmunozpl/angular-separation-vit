"""e2 — identidad de cabeza entre mitades de la sonda.

ejecuta el prerregistro `Paper_X/prereg_E2_identidad.md` (respaldo
c7773e4, 21-09-2026): prueba gauge-invariante de que la firma de
respuesta es específica por cabeza, sin v1(w_o). por semilla y capa se
recomputan las firmas de las dos mitades de la sonda con el mismo
código de b2 y se mide la tasa de identificación ---fracción de
cabezas cuya firma en la mitad a tiene por vecina más cercana su
propia firma en la mitad b--- contra un nulo de permutación.

particiones: «rotado» (mitades balanceadas por clase, p en la rejilla
de b2; p=1000 es la primaria) y «bloques» (la sonda publicada de mil
imágenes con su split posicional i <-> i+500, robustez).

uso:
    python scripts/identidad_mitades.py [--seeds 42 ...] [--rejilla ...]
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.carga import cargar_probe_tensor, cargar_vit_base  # noqa: E402
from src.firma_funcional import w_o_por_cabeza  # noqa: E402
from scripts.curva_sonda import (SONDA, grams,  # noqa: E402
                                 particion_por_clase)
from scripts.radio_firma import (DIM_CABEZA, N_CABEZAS,  # noqa: E402
                                 PROBE, lectura_capa)

DIR_SAL = "artifacts/logs/identidad_mitades"
REJILLA = [250, 500, 1000, 2000, 5000]
SEMILLAS = [42, 43, 44, 45, 46]
N_PERM = 1000
UMBRAL_CAPA = 10 / 12


def firmas_mitades(modelo, imgs: torch.Tensor,
                   pares: list[tuple[int, int]], sueltas: list[int],
                   disp: str) -> tuple[list[torch.Tensor],
                                       list[torch.Tensor]]:
    """firmas unitarias de cada mitad, capa a capa, con el código de b2.

    Args:
        modelo: vit en eval sobre disp.
        imgs: sonda decodificada completa, en cpu.
        pares: apareamiento (posición en la mitad a, posición en b).
        sueltas: posiciones fuera del apareamiento (no entran).
        disp: dispositivo.

    Returns:
        tupla (firmas_a, firmas_b): listas por capa de tensores [h, d].
    """
    _, m_a, m_b, _ = grams(modelo, imgs, pares, sueltas, disp)
    nulo = torch.zeros_like(m_a[0])
    fa, fb = [], []
    for c in range(len(modelo.blocks)):
        w_o = w_o_por_cabeza(modelo, c, N_CABEZAS, DIM_CABEZA)
        a = lectura_capa(m_a[c], nulo, w_o)[0].detach().double().cpu()
        b = lectura_capa(m_b[c], nulo, w_o)[0].detach().double().cpu()
        # se exige forma [h, d] y norma unidad antes de comparar
        assert a.shape == b.shape == (N_CABEZAS, w_o.shape[-1])
        uno = torch.ones(N_CABEZAS, dtype=a.dtype)
        assert torch.allclose(a.norm(dim=1), uno, atol=1e-6)
        assert torch.allclose(b.norm(dim=1), uno, atol=1e-6)
        fa.append(a)
        fb.append(b)
    return fa, fb


def tasa_id(m: torch.Tensor) -> float:
    """fracción de cabezas cuyo máximo por fila cae en la diagonal."""
    return float((m.argmax(dim=1) == torch.arange(m.shape[0]))
                 .double().mean())


def tasa_hungaro(m: torch.Tensor) -> float:
    """fracción de cabezas bien asignadas con emparejamiento óptimo."""
    fil, col = linear_sum_assignment(-m.numpy())
    return float(np.mean(fil == col))


def p99_nulo(m: torch.Tensor, g: torch.Generator, n: int) -> float:
    """percentil 99 de la tasa bajo permutación de etiquetas de b."""
    tasas = []
    for _ in range(n):
        perm = torch.randperm(m.shape[1], generator=g)
        tasas.append(tasa_id(m[:, perm]))
    return float(np.percentile(tasas, 99))


def filas_particion(fa: list, fb: list, g: torch.Generator,
                    etiqueta: dict) -> list[dict]:
    """estadísticos por capa de una partición."""
    filas = []
    for c, (a, b) in enumerate(zip(fa, fb)):
        m = (a @ b.t()).abs()
        filas.append({**etiqueta, "capa": c, "id": tasa_id(m),
                      "id_hungaro": tasa_hungaro(m),
                      "p99_nulo": p99_nulo(m, g, N_PERM)})
    return filas


def main() -> None:
    """corre las particiones por semilla y vuelca los dos artefactos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=SEMILLAS)
    parser.add_argument("--rejilla", nargs="+", type=int,
                        default=REJILLA)
    args = parser.parse_args()
    disp = "cuda" if torch.cuda.is_available() else "cpu"

    blob = torch.load(SONDA, weights_only=False)
    etiquetas = blob["labels"].tolist()
    pos = {r: i for i, r in enumerate(blob["paths"])}
    alm = torch.load(PROBE, weights_only=False)["paths"]
    idx_alm = [pos[r] for r in alm]
    mitad = len(idx_alm) // 2
    pares_bloques = [(idx_alm[i], idx_alm[mitad + i])
                     for i in range(mitad)]
    imgs = cargar_probe_tensor(SONDA)
    print(f"[sonda] {imgs.shape[0]} imágenes; sonda publicada "
          f"{len(idx_alm)} en bloques de clases")

    g = torch.Generator().manual_seed(0)
    filas: list[dict] = []
    for seed in tqdm(args.seeds, desc="semillas"):
        ckpt = (f"artifacts/checkpoints/vitb_clean/"
                f"attnA_base_seed{seed}_last.pt")
        modelo = cargar_vit_base(ckpt, device=disp).eval()
        fa, fb = firmas_mitades(modelo, imgs, pares_bloques, [], disp)
        filas += filas_particion(fa, fb, g, {"seed": seed, "P": 1000,
                                             "particion": "bloques"})
        for p in tqdm(args.rejilla, desc="rejilla", leave=False):
            pares, sueltas = particion_por_clase(etiquetas[:p])
            fa, fb = firmas_mitades(modelo, imgs, pares, sueltas, disp)
            filas += filas_particion(fa, fb, g, {"seed": seed, "P": p,
                                                 "particion": "rotado"})
            ult = [f for f in filas if f["seed"] == seed
                   and f["P"] == p and f["particion"] == "rotado"]
            print(f"  seed {seed} P={p}: id media "
                  f"{np.mean([f['id'] for f in ult]):.3f}, p99 nulo "
                  f"{np.mean([f['p99_nulo'] for f in ult]):.3f}",
                  flush=True)
        del modelo
        if disp == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(DIR_SAL, exist_ok=True)
    df = pd.DataFrame(filas)
    df.to_csv(f"{DIR_SAL}/identidad.csv", index=False)
    df["alcanza"] = df["id"] >= UMBRAL_CAPA - 1e-9
    por_seed = df.groupby(["P", "particion", "seed"]).agg(
        id_m=("id", "mean"), hung_m=("id_hungaro", "mean"),
        p99_m=("p99_nulo", "mean"),
        capas_10_12=("alcanza", "sum")).reset_index()
    por_seed["supera_p99"] = por_seed["id_m"] > por_seed["p99_m"]
    res = por_seed.groupby(["P", "particion"]).agg(
        id_m=("id_m", "mean"), id_s=("id_m", "std"),
        hung_m=("hung_m", "mean"), p99_m=("p99_m", "mean"),
        capas_10_12=("capas_10_12", "mean"),
        semillas_superan=("supera_p99", "sum"),
        n_semillas=("seed", "count")).reset_index()
    res.to_csv(f"{DIR_SAL}/resumen.csv", index=False)
    print(f"\n[guardado] {DIR_SAL} (identidad.csv, {len(df)} filas; "
          f"resumen.csv)")
    print(res.to_string())
    print("\n--- 15 observaciones aleatorias ---")
    print(df.drop(columns="alcanza").sample(15, random_state=0)
            .to_string())


if __name__ == "__main__":
    main()
