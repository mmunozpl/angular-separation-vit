"""pasada 2 — radio certificado de la firma frente al estimador.

ejecuta el preregistro enmendado de Paper_X/R0 §8 (25-08, pre-dato):
por cabeza, r_h = (sigma_1 - sigma_2) * m_l / (8 + m_l) es cota
certificada sobre ||E||_2 rectangular ---cadena cuerda + Wedin con
denominador reparado por Weyl---, y la fluctuación real del
estimador se mide con el half-split de mismo tamaño E = C_swap - C
(las filas de la primera mitad del probe sustituidas por las de la
segunda, apareadas por posición). nada se materializa en P x d: el
gram por cabeza en d_h da sigma y firma exactas vía B = L^T W_O, y
||E||_2 = sigma_1(L_D^T W_O) con D el gram de las diferencias.

dos gates antes de leer veredicto alguno: invariancia de escala de
r/||E|| bajo C -> 10C, y reproducción exacta de las decisiones por
firma almacenadas (decision_rota.csv, seed 42, gauge_idx 0). si un
gate falla, el script aborta sin imprimir estadístico.

uso:
    python scripts/radio_firma.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.carga import cargar_probe_tensor, cargar_vit_base  # noqa: E402
from src.firma_funcional import (CapturaContexto,  # noqa: E402
                                 w_o_por_cabeza)
from src.nucleo_lectura import decisiones  # noqa: E402

CKPT = "artifacts/checkpoints/vitb_clean/attnA_base_seed42_last.pt"
PROBE = "artifacts/probe_set/imagenet100_val_1k.pt"
CSV_DEC = "artifacts/logs/decision_rota/decision_rota.csv"
SALIDA = "artifacts/logs/radio_firma/radio_firma.csv"
N_CABEZAS, DIM_CABEZA = 12, 64
K_PODA = 3
PARES_LOTE = 24          # imágenes i e i+500 viajan en el mismo lote


@torch.no_grad()
def acumula_grams(modelo, imgs: torch.Tensor, disp: str):
    """gram total y gram de diferencias apareadas, por capa y cabeza.

    Args:
        modelo: vit en eval sobre disp.
        imgs: probe completo [1000, 3, 224, 224] en cpu.
        disp: dispositivo.

    Returns:
        tupla (gram, dif): dicts capa -> tensor [h, dh, dh] con
        sum avᵀav sobre todo el probe y sum (av₂−av₁)ᵀ(av₂−av₁)
        sobre los pares (i, i+500).
    """
    n_capas = len(modelo.blocks)
    mitad = imgs.shape[0] // 2
    captura = CapturaContexto(modelo, N_CABEZAS, DIM_CABEZA)
    gram = {c: torch.zeros(N_CABEZAS, DIM_CABEZA, DIM_CABEZA,
                           device=disp, dtype=torch.float64)
            for c in range(n_capas)}
    dif = {c: torch.zeros_like(gram[c]) for c in range(n_capas)}
    for ini in tqdm(range(0, mitad, PARES_LOTE), desc="grams",
                    leave=False):
        b = min(PARES_LOTE, mitad - ini)
        lote = torch.cat([imgs[ini:ini + b],
                          imgs[mitad + ini:mitad + ini + b]])
        captura.limpiar()
        _ = modelo(lote.to(disp))
        for c in range(n_capas):
            av = captura.contexto[c][:, 1:, :, :].double()
            av = av.permute(2, 0, 1, 3)          # [h, 2b, p, dh]
            todo = av.reshape(N_CABEZAS, -1, DIM_CABEZA)
            gram[c] += torch.einsum("hnd,hne->hde", todo, todo)
            d = (av[:, b:] - av[:, :b]).reshape(N_CABEZAS, -1,
                                                DIM_CABEZA)
            dif[c] += torch.einsum("hnd,hne->hde", d, d)
    captura.quitar()
    return gram, dif


def lectura_capa(gram_h: torch.Tensor, dif_h: torch.Tensor,
                 w_o: torch.Tensor):
    """sigma, firma y ||E|| de cada cabeza de una capa.

    Args:
        gram_h: [h, dh, dh] gram del contexto.
        dif_h: [h, dh, dh] gram de las diferencias apareadas.
        w_o: [h, dh, d] proyección de salida por cabeza.

    Returns:
        tupla (firmas [h, d], sigma1 [h], sigma2 [h], normE [h]).
    """
    firmas, s1, s2, ne = [], [], [], []
    for h in range(gram_h.shape[0]):
        val, vec = torch.linalg.eigh(gram_h[h])
        ele = vec * val.clamp_min(0).sqrt()
        b = ele.t() @ w_o[h].double()
        u, s, vh = torch.linalg.svd(b, full_matrices=False)
        firmas.append(vh[0])
        s1.append(float(s[0]))
        s2.append(float(s[1]))
        val_d, vec_d = torch.linalg.eigh(dif_h[h])
        ele_d = vec_d * val_d.clamp_min(0).sqrt()
        ne.append(float(torch.linalg.svdvals(
            ele_d.t() @ w_o[h].double())[0]))
    return torch.stack(firmas), s1, s2, ne


def margen_y_pares(firmas: torch.Tensor):
    """par ganador, subcampeón y margen de |cos| de una capa.

    Args:
        firmas: [h, d] firmas unitarias.

    Returns:
        tupla (par ganador, par subcampeón, margen, cabezas
        implicadas).
    """
    g = (firmas @ firmas.t()).abs()
    g.fill_diagonal_(-1.0)
    planos = []
    n = g.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            planos.append((float(g[i, j]), (i, j)))
    planos.sort(reverse=True)
    (c1, p1), (c2, p2) = planos[0], planos[1]
    return p1, p2, c1 - c2, sorted(set(p1) | set(p2))


def main() -> None:
    """corre gates y preregistro, y guarda el csv con veredicto."""
    disp = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = cargar_vit_base(CKPT, device=disp).eval()
    imgs = cargar_probe_tensor(PROBE)
    print(f"[probe] {imgs.shape[0]} imágenes")
    gram, dif = acumula_grams(modelo, imgs, disp)

    filas, ratios, gate_repro = [], [], []
    dec = pd.read_csv(CSV_DEC)
    dec = dec[(dec.seed == 42) & (dec.criterio == "firma")
              & (dec.gauge_idx == 0)]
    fuera_dominio = 0
    for capa in tqdm(range(len(modelo.blocks)), desc="capas"):
        w_o = w_o_por_cabeza(modelo, capa, N_CABEZAS, DIM_CABEZA)
        firmas, s1, s2, ne = lectura_capa(gram[capa], dif[capa], w_o)

        # gate 1: invariancia de escala, C -> 10C en esta capa
        f10, s1x, s2x, nex = lectura_capa(100.0 * gram[capa],
                                          100.0 * dif[capa], w_o)
        for h in range(N_CABEZAS):
            r_a = (s1[h] - s2[h]) / max(ne[h], 1e-30)
            r_b = (s1x[h] - s2x[h]) / max(nex[h], 1e-30)
            assert abs(r_a - r_b) < 1e-8 * max(abs(r_a), 1.0), (
                f"escala rota: capa {capa} cabeza {h}")

        # gate 2: la decisión por firma clava la almacenada
        par, _ = decisiones(firmas.float(), K_PODA)
        alm = dec[dec.capa == capa].iloc[0]["par_top"]
        gate_repro.append(str(par) == str(alm))

        p1, p2, margen, impl = margen_y_pares(firmas)
        for h in range(N_CABEZAS):
            hueco = s1[h] - s2[h]
            r_h = hueco * margen / (8.0 + margen)
            en_dominio = ne[h] < hueco
            if not en_dominio:
                fuera_dominio += 1
            ratio = (r_h / ne[h]) if ne[h] > 0 else float("inf")
            # fuera de dominio el certificado es vacío: cuenta <= 1
            ratio_cons = ratio if en_dominio else min(ratio, 1.0)
            ratios.append((capa, h, ratio, ratio_cons,
                           h in impl, hueco, margen))
            filas.append({
                "capa": capa, "cabeza": h, "sigma1": s1[h],
                "sigma2": s2[h], "hueco": hueco, "margen": margen,
                "norma_E": ne[h], "r_h": r_h, "ratio": ratio,
                "en_dominio": en_dominio, "implicada": h in impl,
                "par_ganador": str(p1), "par_subcampeon": str(p2)})

    assert all(gate_repro), (
        f"reproducción rota: {gate_repro.count(False)} capas "
        f"no clavan la decisión almacenada")
    print(f"[gate] escala: invariante | reproducción: 12/12 capas")
    print(f"[dominio] cabezas con ||E|| >= hueco: {fuera_dominio}")

    df = pd.DataFrame(filas)
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    df.to_csv(SALIDA, index=False)                # se guarda en csv
    print(df.sample(15, random_state=0))          # 15 observaciones

    # veredicto preregistrado: mediana del primario y conservador
    mediana = float(df["ratio"].median())
    df["ratio_cons"] = [r[3] for r in ratios]
    frac = float((df.groupby("capa")["ratio_cons"].min() > 1).mean())
    # g = minimo del hueco sobre TODAS las cabezas de la capa:
    # cualquier cabeza puede entrar en un par competidor (enmienda
    # del cuantificador, 25-08); la fluctuacion, el peor caso igual
    cert = []
    for capa, sub in df.groupby("capa"):
        cert.append(float(sub.hueco.min() * sub.margen.iloc[0]
                          / (8 + sub.margen.iloc[0])
                          / sub.norma_E.max()))
    frac_cert = sum(c > 1 for c in cert) / len(cert)
    print(f"\n[primario] mediana r_h/||E|| = {mediana:.3f}")
    print(f"[conservador] capas con min ratio > 1: {frac:.0%}")
    print(f"[certificado por capa] fracción > 1: {frac_cert:.0%} "
          f"(min {min(cert):.3f}, mediana "
          f"{sorted(cert)[len(cert)//2]:.3f})")
    v = ("POSITIVO Y NO VACUO" if mediana >= 2 else
         "POSITIVO" if mediana > 1 else
         "el radio no supera la fluctuación a este P")
    print(f"[veredicto] {v}")


if __name__ == "__main__":
    main()
