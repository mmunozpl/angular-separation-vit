"""diagnóstico post-hoc del recambio real, fuera del preregistro.

recomputa la firma sobre cada mitad de la sonda por separado y
compara la decisión del par más redundante entre mitades y contra la
almacenada. es el complemento del veredicto de radio_firma.py: aquel
acota el peor caso adversario; este mide la perturbación real.
etiquetado como diagnóstico fuera del preregistro en el canónico
(sección de la firma y Límites).

uso:
    python scripts/diag_recambio.py
"""

import os
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
from scripts.radio_firma import (CKPT, CSV_DEC, DIM_CABEZA,  # noqa: E402
                                 K_PODA, N_CABEZAS, PARES_LOTE, PROBE,
                                 lectura_capa)

SALIDA = "artifacts/logs/radio_firma/diag_recambio.csv"


@torch.no_grad()
def grams_por_mitad(modelo, imgs: torch.Tensor, disp: str):
    """gram del contexto por capa y cabeza, separado por mitades.

    Args:
        modelo: vit en eval sobre disp.
        imgs: probe completo [1000, 3, 224, 224] en cpu.
        disp: dispositivo.

    Returns:
        tupla (g1, g2): dicts capa -> [h, dh, dh], una por mitad.
    """
    n_capas = len(modelo.blocks)
    mitad = imgs.shape[0] // 2
    cap = CapturaContexto(modelo, N_CABEZAS, DIM_CABEZA)
    g1 = {c: torch.zeros(N_CABEZAS, DIM_CABEZA, DIM_CABEZA,
                         device=disp, dtype=torch.float64)
          for c in range(n_capas)}
    g2 = {c: torch.zeros_like(g1[c]) for c in range(n_capas)}
    for ini in tqdm(range(0, mitad, PARES_LOTE), desc="mitades",
                    leave=False):
        b = min(PARES_LOTE, mitad - ini)
        lote = torch.cat([imgs[ini:ini + b],
                          imgs[mitad + ini:mitad + ini + b]])
        cap.limpiar()
        _ = modelo(lote.to(disp))
        for c in range(n_capas):
            av = cap.contexto[c][:, 1:, :, :].double()
            av = av.permute(2, 0, 1, 3)
            a1 = av[:, :b].reshape(N_CABEZAS, -1, DIM_CABEZA)
            a2 = av[:, b:].reshape(N_CABEZAS, -1, DIM_CABEZA)
            g1[c] += torch.einsum("hnd,hne->hde", a1, a1)
            g2[c] += torch.einsum("hnd,hne->hde", a2, a2)
    cap.quitar()
    return g1, g2


def main() -> None:
    """compara decisiones por mitad y guarda el csv."""
    disp = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = cargar_vit_base(CKPT, device=disp).eval()
    imgs = cargar_probe_tensor(PROBE)
    g1, g2 = grams_por_mitad(modelo, imgs, disp)

    dec = pd.read_csv(CSV_DEC)
    dec = dec[(dec.seed == 42) & (dec.criterio == "firma")
              & (dec.gauge_idx == 0)]
    filas = []
    nulo = None
    for c in range(len(modelo.blocks)):
        w_o = w_o_por_cabeza(modelo, c, N_CABEZAS, DIM_CABEZA)
        if nulo is None:
            nulo = torch.zeros_like(g1[c])
        fa = lectura_capa(g1[c], nulo, w_o)[0]
        fb = lectura_capa(g2[c], nulo, w_o)[0]
        pa, ta = decisiones(fa.float(), K_PODA)
        pb, tb = decisiones(fb.float(), K_PODA)
        alm = dec[dec.capa == c].iloc[0]["par_top"]
        filas.append({"capa": c, "par_mitad1": str(pa),
                      "par_mitad2": str(pb), "par_almacenado": alm,
                      "par_igual": str(pa) == str(pb),
                      "top_igual": sorted(ta) == sorted(tb)})
        print(f"  capa {c:2d}: mitad1 {pa} mitad2 {pb} "
              f"almacenado {alm}")

    df = pd.DataFrame(filas)
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    df.to_csv(SALIDA, index=False)                # se guarda en csv
    print(df.sample(min(15, len(df)), random_state=0))  # 15 obs
    print(f"\n  par igual entre mitades: "
          f"{int(df.par_igual.sum())}/12 | top-{K_PODA} igual: "
          f"{int(df.top_igual.sum())}/12")


if __name__ == "__main__":
    main()
