"""b2 — estabilidad de la firma frente al tamaño de sonda.

ejecuta el preregistro de `Paper_X/prereg_B2_sonda.md` (congelado
26-08-2026, resolución (a) y umbrales ratificados 28-08): sobre la
rejilla P ∈ {250, 500, 1000, 2000, 5000} de sondas anidadas en orden
rotado, mide por semilla y capa la coincidencia split-half de la
decisión por firma ---par más redundante y conjunto de poda---, la
mediana de r_h/‖E‖ del lema del radio y el recuento fuera de
dominio. la curva convierte el 8/12 de P=1000 en un punto
caracterizado.

el veredicto de 12/12 se lee en sentido estricto ---las cinco
semillas a doce capas---, criterio fijado aquí antes de correr; el
techo se reporta si no se alcanza, sin presuponer monotonía.

toda la curva usa el orden rotado, con split-half y apareamiento de
diferencias **dentro de cada clase** (primeras ⌊m/2⌋ ocurrencias
contra últimas; con m impar la sobrante queda fuera del
apareamiento), de modo que la semántica no cambie con P. el orden
almacenado corre solo para el gate.

dos gates anteceden a todo estadístico ---el script aborta sin
imprimir si alguno falla---:

1. invariancia de escala de r/‖E‖ bajo C -> 10C, heredada de
   `radio_firma.py`;
2. a P=1000 con el orden almacenado y su split posicional i<->i+500,
   reproducción del 8/12 y de la mediana 0,001 publicados.

uso:
    python scripts/curva_sonda.py [--seeds 42 ...] [--rejilla 250 ...]
"""

import argparse
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.carga import cargar_probe_tensor, cargar_vit_base  # noqa: E402
from src.firma_funcional import (CapturaContexto,  # noqa: E402
                                 w_o_por_cabeza)
from src.nucleo_lectura import decisiones  # noqa: E402
from scripts.radio_firma import (DIM_CABEZA, K_PODA,  # noqa: E402
                                 N_CABEZAS, PARES_LOTE, PROBE,
                                 lectura_capa, margen_y_pares)

SONDA = "artifacts/probe_set/imagenet100_val_5k_rotado.pt"
DIR_SAL = "artifacts/logs/curva_sonda"
REJILLA = [250, 500, 1000, 2000, 5000]
SEMILLAS = [42, 43, 44, 45, 46]
GATE_COINC = 8          # 8/12 publicado (seed 42, orden almacenado)
GATE_MEDIANA = 0.00105  # mediana r_h/‖E‖ publicada
GATE_TOL = 0.02         # tolerancia relativa de la mediana


def particion_por_clase(
    etiquetas: list[int],
) -> tuple[list[tuple[int, int]], list[int]]:
    """mitades y apareamiento dentro de cada clase, en orden rotado.

    Args:
        etiquetas: clase de cada posición del prefijo, en su orden.

    Returns:
        tupla (pares, sueltas): pares (posición de la primera mitad,
        posición de la última) y posiciones que quedan sin aparear.
    """
    por_clase: dict[int, list[int]] = {}
    for i, c in enumerate(etiquetas):
        por_clase.setdefault(c, []).append(i)
    pares: list[tuple[int, int]] = []
    sueltas: list[int] = []
    for c in sorted(por_clase):
        occ = por_clase[c]
        m = len(occ) // 2
        pares += list(zip(occ[:m], occ[len(occ) - m:]))
        if len(occ) % 2:
            sueltas.append(occ[m])
    return pares, sueltas


@torch.no_grad()
def grams(modelo, imgs: torch.Tensor, pares: list[tuple[int, int]],
          sueltas: list[int], disp: str) -> tuple[dict, dict, dict,
                                                  dict]:
    """gram total, de cada mitad y de las diferencias apareadas.

    Args:
        modelo: vit en eval sobre disp.
        imgs: tensor de la sonda completa en cpu.
        pares: apareamiento (primera mitad, última mitad).
        sueltas: posiciones fuera del apareamiento.
        disp: dispositivo.

    Returns:
        tupla (total, mitad_a, mitad_b, dif): dicts capa -> [h,dh,dh].
    """
    n_capas = len(modelo.blocks)
    cap = CapturaContexto(modelo, N_CABEZAS, DIM_CABEZA)
    cero = {c: torch.zeros(N_CABEZAS, DIM_CABEZA, DIM_CABEZA,
                           device=disp, dtype=torch.float64)
            for c in range(n_capas)}
    tot = {c: cero[c].clone() for c in range(n_capas)}
    m_a = {c: cero[c].clone() for c in range(n_capas)}
    m_b = {c: cero[c].clone() for c in range(n_capas)}
    dif = {c: cero[c].clone() for c in range(n_capas)}
    ia = [p[0] for p in pares]
    ib = [p[1] for p in pares]
    for ini in tqdm(range(0, len(pares), PARES_LOTE), desc="pares",
                    leave=False):
        sl = slice(ini, ini + PARES_LOTE)
        b = len(ia[sl])
        lote = torch.cat([imgs[ia[sl]], imgs[ib[sl]]])
        cap.limpiar()
        _ = modelo(lote.to(disp))
        for c in range(n_capas):
            av = cap.contexto[c][:, 1:, :, :].double()
            av = av.permute(2, 0, 1, 3)            # [h, 2b, p, dh]
            a = av[:, :b].reshape(N_CABEZAS, -1, DIM_CABEZA)
            z = av[:, b:].reshape(N_CABEZAS, -1, DIM_CABEZA)
            d = (av[:, b:] - av[:, :b]).reshape(N_CABEZAS, -1,
                                                DIM_CABEZA)
            m_a[c] += torch.einsum("hnd,hne->hde", a, a)
            m_b[c] += torch.einsum("hnd,hne->hde", z, z)
            dif[c] += torch.einsum("hnd,hne->hde", d, d)
    for ini in tqdm(range(0, len(sueltas), 2 * PARES_LOTE),
                    desc="sueltas", leave=False):
        idx = sueltas[ini:ini + 2 * PARES_LOTE]
        cap.limpiar()
        _ = modelo(imgs[idx].to(disp))
        for c in range(n_capas):
            av = cap.contexto[c][:, 1:, :, :].double()
            av = av.permute(2, 0, 1, 3).reshape(N_CABEZAS, -1,
                                                DIM_CABEZA)
            tot[c] += torch.einsum("hnd,hne->hde", av, av)
    cap.quitar()
    for c in range(n_capas):
        # el total suma las dos mitades y lo que quedó sin aparear
        tot[c] += m_a[c] + m_b[c]
    return tot, m_a, m_b, dif


def lee_capas(modelo, tot: dict, m_a: dict, m_b: dict, dif: dict,
              etiqueta: dict) -> tuple[list[dict], list[dict]]:
    """coincidencia entre mitades y radio por cabeza, capa a capa.

    Args:
        modelo: el vit.
        tot: gram de la sonda completa por capa.
        m_a: gram de la primera mitad.
        m_b: gram de la última mitad.
        dif: gram de las diferencias apareadas.
        etiqueta: campos comunes de la corrida (semilla, P).

    Returns:
        tupla (filas de capa, filas de cabeza).

    Raises:
        SystemExit: si la invariancia de escala falla (gate 1).
    """
    f_capa, f_cab = [], []
    nulo = None
    for c in range(len(modelo.blocks)):
        w_o = w_o_por_cabeza(modelo, c, N_CABEZAS, DIM_CABEZA)
        if nulo is None:
            nulo = torch.zeros_like(tot[c])
        firmas, s1, s2, ne = lectura_capa(tot[c], dif[c], w_o)
        # gate 1: c -> 10c no puede mover el cociente r/‖e‖
        _, s1x, s2x, nex = lectura_capa(100.0 * tot[c],
                                        100.0 * dif[c], w_o)
        for h in range(N_CABEZAS):
            r_a = (s1[h] - s2[h]) / max(ne[h], 1e-30)
            r_b = (s1x[h] - s2x[h]) / max(nex[h], 1e-30)
            if abs(r_a - r_b) >= 1e-8 * max(abs(r_a), 1.0):
                raise SystemExit(
                    f"gate de escala roto: capa {c} cabeza {h}")
        fa = lectura_capa(m_a[c], nulo, w_o)[0]
        fb = lectura_capa(m_b[c], nulo, w_o)[0]
        pa, ta = decisiones(fa.float(), K_PODA)
        pb, tb = decisiones(fb.float(), K_PODA)
        _, _, margen, _ = margen_y_pares(firmas)
        ratios, fuera = [], 0
        for h in range(N_CABEZAS):
            hueco = s1[h] - s2[h]
            r_h = hueco * margen / (8.0 + margen)
            en_dom = ne[h] < hueco
            fuera += int(not en_dom)
            ratio = r_h / ne[h] if ne[h] > 0 else float("inf")
            ratios.append(ratio)
            f_cab.append({**etiqueta, "capa": c, "cabeza": h,
                          "sigma1": s1[h], "sigma2": s2[h],
                          "hueco": hueco, "margen": margen,
                          "norma_E": ne[h], "r_h": r_h,
                          "ratio": ratio, "en_dominio": en_dom})
        f_capa.append({**etiqueta, "capa": c, "par_mitad1": str(pa),
                       "par_mitad2": str(pb),
                       "par_igual": str(pa) == str(pb),
                       "top_igual": sorted(ta) == sorted(tb),
                       "ratio_mediana": statistics.median(ratios),
                       "fuera_dominio": fuera, "margen": margen})
    return f_capa, f_cab


def gate_almacenado(modelo, imgs: torch.Tensor, idx_alm: list[int],
                    disp: str) -> tuple[int, float]:
    """reproduce el punto publicado con el orden almacenado (gate 2).

    Args:
        modelo: el vit de la semilla 42.
        imgs: la sonda decodificada completa.
        idx_alm: posiciones de las mil imágenes en su orden publicado.
        disp: dispositivo.

    Returns:
        tupla (coincidencias de par sobre 12, mediana de r_h/‖E‖).
    """
    mitad = len(idx_alm) // 2
    pares = [(idx_alm[i], idx_alm[mitad + i]) for i in range(mitad)]
    tot, m_a, m_b, dif = grams(modelo, imgs, pares, [], disp)
    f_capa, f_cab = lee_capas(modelo, tot, m_a, m_b, dif,
                              {"seed": 42, "P": 1000,
                               "orden": "almacenado"})
    coinc = sum(f["par_igual"] for f in f_capa)
    return coinc, statistics.median(f["ratio"] for f in f_cab)


def main() -> None:
    """corre los gates y la curva, y vuelca los tres artefactos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=SEMILLAS)
    parser.add_argument("--rejilla", nargs="+", type=int,
                        default=REJILLA)
    args = parser.parse_args()
    if 42 not in args.seeds:
        raise SystemExit("el gate 2 vive en la semilla 42: inclúyela")
    disp = "cuda" if torch.cuda.is_available() else "cpu"

    blob = torch.load(SONDA, weights_only=False)
    etiquetas = blob["labels"].tolist()
    pos = {r: i for i, r in enumerate(blob["paths"])}
    alm = torch.load(PROBE, weights_only=False)["paths"]
    idx_alm = [pos[r] for r in alm]     # el orden publicado, mapeado
    imgs = cargar_probe_tensor(SONDA)
    print(f"[sonda] {imgs.shape[0]} imágenes en orden rotado")

    filas_capa, filas_cab = [], []
    for seed in tqdm(args.seeds, desc="semillas"):
        ckpt = (f"artifacts/checkpoints/vitb_clean/"
                f"attnA_base_seed{seed}_last.pt")
        modelo = cargar_vit_base(ckpt, device=disp).eval()
        if seed == 42:
            coinc, mediana = gate_almacenado(modelo, imgs, idx_alm,
                                             disp)
            ok = (coinc == GATE_COINC
                  and abs(mediana - GATE_MEDIANA)
                  <= GATE_TOL * GATE_MEDIANA)
            print(f"[gate 2] orden almacenado: par igual "
                  f"{coinc}/12 (publicado {GATE_COINC}/12), mediana "
                  f"{mediana:.5f} (publicada {GATE_MEDIANA})")
            if not ok:
                raise SystemExit("gate 2 fallado: sin veredicto")
        for p in tqdm(args.rejilla, desc="rejilla", leave=False):
            pares, sueltas = particion_por_clase(etiquetas[:p])
            tot, m_a, m_b, dif = grams(modelo, imgs, pares, sueltas,
                                       disp)
            eti = {"seed": seed, "P": p, "orden": "rotado"}
            fc, fh = lee_capas(modelo, tot, m_a, m_b, dif, eti)
            filas_capa += fc
            filas_cab += fh
        del modelo
        if disp == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(DIR_SAL, exist_ok=True)
    df_c = pd.DataFrame(filas_capa)
    df_h = pd.DataFrame(filas_cab)
    df_c.to_csv(f"{DIR_SAL}/curva_sonda.csv", index=False)
    df_h.to_csv(f"{DIR_SAL}/curva_sonda_cabezas.csv", index=False)

    por_seed = df_c.groupby(["P", "seed"]).agg(
        par=("par_igual", "sum"), top=("top_igual", "sum"),
        fuera=("fuera_dominio", "sum")).reset_index()
    res = por_seed.groupby("P").agg(
        par_m=("par", "mean"), par_s=("par", "std"),
        top_m=("top", "mean"), top_s=("top", "std"),
        fuera_m=("fuera", "mean")).reset_index()
    res["ratio_mediana"] = df_h.groupby("P").ratio.median().values
    res.to_csv(f"{DIR_SAL}/curva_sonda_resumen.csv", index=False)
    print(f"\n[guardado] {DIR_SAL} (3 csv, {len(df_c)} filas de capa)")
    print(df_c.sample(15, random_state=0).to_string())  # 15 obs
    print("\n--- 15 observaciones del artefacto de cabezas ---")
    print(df_h.sample(15, random_state=0).to_string())  # 15 obs

    print("\n    P | par igual /12 (m±s) | top-3 /12 (m±s) | "
          "mediana r/‖E‖ | fuera dom/144")
    for _, f in res.iterrows():
        print(f"{int(f.P):>5} | {f.par_m:>6.1f}±{f.par_s:.1f}      | "
              f"{f.top_m:>5.1f}±{f.top_s:.1f}     | "
              f"{f.ratio_mediana:>12.5f} | {f.fuera_m:>6.1f}")

    # criterio estricto declarado en el prereg: 12/12 exige que las
    # cinco semillas coincidan en las doce capas
    llega = res[res.par_m == 12.0]
    if len(llega):
        print(f"\n[veredicto] primer P con 12/12 (todas las "
              f"semillas): {int(llega.iloc[0].P)}")
    else:
        techo = res.iloc[-1]
        print(f"\n[veredicto] no se alcanza 12/12; techo a P="
              f"{int(techo.P)}: {techo.par_m:.1f}/12")
    creciente = res.par_m.is_monotonic_increasing
    print(f"[monotonía] coincidencia creciente con P: "
          f"{'sí' if creciente else 'NO ---se reporta tal cual'}")


if __name__ == "__main__":
    main()
