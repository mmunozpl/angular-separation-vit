"""fase g — gauge-flip valor-salida demostrado, sin reentrenar.

carga los attnA_base ya entrenados y, por capa, aplica varias rotaciones
de gauge valor-salida que dejan la salida intacta a precisión de máquina.
reporta tres cifras lado a lado: error de salida (~precisión), deriva de
u1(w_o) (gauge-variante, lo que la poda lee) y deriva del circuito ov
(invariante, svd exacta). dos asserts blindan el resultado: el script se
niega a mentir si el gauge cambia la función o si el circuito deriva.

uso:
    python scripts/run_fase_G.py [--arch vitb|vitl]
"""

import argparse
import copy
import glob
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.carga import ARQUITECTURAS, cargar_probe_tensor, cargar_vit_base
from src.firma_funcional import w_o_por_cabeza, w_v_por_cabeza
from src.gauge_flip import aplica_gauge_ov

# --- campos por arquitectura (main los fija según --arch) -----------
CKPTS = sorted(glob.glob(
    "artifacts/checkpoints/vitb_clean/attnA_base_seed*_last.pt"))
PROBE = "artifacts/probe_set/imagenet100_val_1k.pt"
SALIDA = "artifacts/logs/fase_G/gauge_flip.csv"
ARCH = "vitb"  # etiqueta de arquitectura; se sobreescribe con --arch
# --------------------------------------------------------------------
N_CABEZAS, DIM_CABEZA = 12, 64
MODEL_NAME = "vit_base_patch16_224"
IMG_SIZE, CROP_PCT, INTERP = 224, 0.875, "bilinear"
# se barre la fuerza del gauge por el coeficiente de la identidad: menor
# valor -> r más lejos de un múltiplo escalar de i -> gauge más fuerte.
# así se denota que TODA rotación mueve u1(w_o) y NINGUNA el circuito ov.
ESCALAS = [128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0]
TOL_SALIDA = 1e-4  # el gauge no puede cambiar la función
TOL_OV = 1e-3      # el circuito ov es invariante


def carga_modelo(ckpt: str):
    """carga el vit-b/16 afinado desde un checkpoint, en eval.

    delega en el loader probado del repo; el vit devuelto expone
    blocks[i].attn.qkv y .proj con la convención de timm.

    args:
        ckpt: ruta del checkpoint attnA_base.

    returns:
        el vit interno de timm en eval, sobre cpu (main lo mueve).
    """
    return cargar_vit_base(ckpt, device="cpu", model_name=MODEL_NAME,
                           img_size=IMG_SIZE)


@torch.no_grad()
def firma_exacta(matriz: torch.Tensor, lado: str) -> torch.Tensor:
    """vector singular dominante por svd exacta (no iteración).

    el diagnóstico de invariancia exige svd exacta: medirla con iteración
    de potencia introduce ruido de init que la enmascara.

    args:
        matriz: tensor [h, m, n] con h matrices.
        lado: 'izq' devuelve u[:,0] (en R^m); 'der' devuelve vh[0] (R^n).

    returns:
        tensor [h, k] con la dirección unitaria por cabeza.
    """
    out = []
    for h in range(matriz.shape[0]):
        u, _, vh = torch.linalg.svd(matriz[h], full_matrices=False)
        out.append(u[:, 0] if lado == "izq" else vh[0])
    return torch.stack(out)


@torch.no_grad()
def deriva(antes: torch.Tensor, despues: torch.Tensor) -> float:
    """deriva media entre firmas, 1 - |cos|, sin signo.

    args:
        antes: tensor [h, d] de firmas antes del gauge.
        despues: tensor [h, d] de firmas después del gauge.

    returns:
        deriva media entre cabezas en [0, 1].
    """
    cos = (antes * despues).sum(-1).abs().clamp(max=1.0)
    return float((1.0 - cos).mean())


def firmas_capa(
    modelo,
    capa: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """firma gauge-variante u1(w_o) y firma invariante del circuito ov.

    ambas por svd exacta y en R^768, para un contraste apples-to-apples:
    única diferencia gauge-variante frente a invariante.

    args:
        modelo: el vit.
        capa: índice de la capa.

    returns:
        tupla (firma_wo [h, d], firma_ov [h, d]).
    """
    w_o = w_o_por_cabeza(modelo, capa, N_CABEZAS, DIM_CABEZA)  # [h,dh,d]
    w_v = w_v_por_cabeza(modelo, capa, N_CABEZAS, DIM_CABEZA)  # [h,dh,d]
    ov = w_v.transpose(-2, -1) @ w_o                          # [h, d, d]
    return firma_exacta(w_o, "der"), firma_exacta(ov, "izq")


@torch.no_grad()
def _logits_lotes(modelo, imgs: torch.Tensor,
                  lote: int = 8) -> torch.Tensor:
    """forward por lotes; en fp64 y resolución alta no cabe entero.

    args:
        modelo: el vit en eval.
        imgs: tensor [n, 3, h, w] en el mismo device/dtype del modelo.
        lote: tamaño de lote.

    returns:
        logits [n, clases] concatenados.
    """
    return torch.cat([modelo(imgs[i:i + lote])
                      for i in range(0, imgs.shape[0], lote)])


def main() -> None:
    """corre el gauge-flip sobre los attnA_base y guarda el csv."""
    global CKPTS, PROBE, ARCH, N_CABEZAS, DIM_CABEZA, MODEL_NAME
    global IMG_SIZE, CROP_PCT, INTERP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="vitb",
                        choices=sorted(ARQUITECTURAS),
                        help="arquitectura: fija checkpoints, etiqueta "
                             "del csv, geometría y resolución nativa")
    args = parser.parse_args()
    ARCH = args.arch
    spec = ARQUITECTURAS[ARCH]
    MODEL_NAME = spec["model_name"]
    N_CABEZAS, DIM_CABEZA = spec["n_cabezas"], spec["dim_cabeza"]
    IMG_SIZE, CROP_PCT = spec["img_size"], spec["crop_pct"]
    INTERP = spec["interp"]
    PROBE = spec["probe"]
    CKPTS = sorted(glob.glob(
        f"artifacts/checkpoints/{ARCH}_clean/attnA_base_seed*_last.pt"))
    if not CKPTS:
        raise SystemExit(f"sin checkpoints base para {ARCH}")
    disp = "cuda" if torch.cuda.is_available() else "cpu"
    imgs = cargar_probe_tensor(PROBE, max_imgs=64, img_size=IMG_SIZE,
                               crop_pct=CROP_PCT, interp=INTERP).to(disp)
    filas: list[dict] = []
    for ckpt in tqdm(CKPTS, desc="semillas"):
        # todo en fp64: el forward fp32 acumula error por encima de la
        # tolerancia en columnas grandes (dinov2-l a 518: 1,8e-4 con
        # invariancia exacta 1e-13 en fp64); medir en fp64 verifica la
        # afirmación matemática a precisión de máquina, con el mismo
        # umbral en todas las columnas. el forward va por lotes para
        # que la resolución nativa alta quepa en memoria.
        modelo = carga_modelo(ckpt).to(disp).double().eval()
        imgs64 = imgs.double()
        logits0 = _logits_lotes(modelo, imgs64)
        for capa in tqdm(range(len(modelo.blocks)), desc="capas",
                         leave=False):
            f_wo_0, f_ov_0 = firmas_capa(modelo, capa)
            for j, escala in enumerate(ESCALAS):
                m = copy.deepcopy(modelo)
                # se barre la fuerza del gauge por escala_id
                desv = aplica_gauge_ov(m, capa, N_CABEZAS, DIM_CABEZA,
                                       semilla=1000 * capa + j,
                                       escala_id=escala)
                err = float(
                    (logits0 - _logits_lotes(m, imgs64)).norm()
                    / logits0.norm().clamp_min(1e-12))
                f_wo_1, f_ov_1 = firmas_capa(m, capa)
                d_wo = deriva(f_wo_0, f_wo_1)
                d_ov = deriva(f_ov_0, f_ov_1)
                # las dos guardas: el script se niega a mentir
                assert err < TOL_SALIDA, (ckpt, capa, escala, err)
                assert d_ov < TOL_OV, (ckpt, capa, escala, d_ov)
                filas.append({
                    "seed": ckpt, "capa": capa, "escala_id": escala,
                    "desv_R": desv, "err_salida": err,
                    "deriva_wo": d_wo, "deriva_ov": d_ov})
                del m
        torch.cuda.empty_cache()

    df = pd.DataFrame(filas)
    df["arch"] = ARCH
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    # merge-por-arch: conserva otras arquitecturas, reemplaza esta
    if os.path.exists(SALIDA):
        prev = pd.read_csv(SALIDA)
        df = pd.concat([prev[prev["arch"] != ARCH], df],
                       ignore_index=True)
    df.to_csv(SALIDA, index=False)  # se guarda en csv
    # los resúmenes de consola se leen SOLO de esta arch: si el csv ya
    # tenía otras, el groupby sin filtrar las mezclaría (fallo silencioso)
    df = df[df["arch"] == ARCH].reset_index(drop=True)
    print(df.sample(min(15, len(df)), random_state=0))  # 15 obs

    # resumen 1: tres cifras por capa (agregado entre semillas y fuerzas)
    por_capa = df.groupby("capa")[
        ["err_salida", "deriva_wo", "deriva_ov"]].mean()
    print("\ncapa | err_salida | deriva_wo | deriva_ov")
    for capa, fila in por_capa.iterrows():
        print(f"{capa:>4} | {fila.err_salida:.2e} | "
              f"{fila.deriva_wo:.4f} | {fila.deriva_ov:.2e}")

    # resumen 2: la deriva de v1(w_o) crece con la fuerza de r, la del
    # circuito ov se queda en ~precisión sea cual sea r (cierra al revisor).
    # deriva_wo en media±std (std muestral ddof=1, criterio único) sobre
    # las 60 combinaciones semilla x capa por fuerza ---tab:gauge---
    cols_f = ["desv_R", "err_salida", "deriva_wo", "deriva_ov"]
    grp = df.groupby("escala_id")[cols_f]
    f_mean, f_std = grp.mean(), grp.std()  # std de pandas = ddof=1
    print("\nescala_id | desv_R | err_salida | deriva_wo (m±s) | deriva_ov")
    for esc in f_mean.sort_index(ascending=False).index:
        m, s = f_mean.loc[esc], f_std.loc[esc]
        print(f"{esc:>9.0f} | {m.desv_R:.3f} | {m.err_salida:.2e} "
              f"| {m.deriva_wo:.4f}±{s.deriva_wo:.4f} | {m.deriva_ov:.2e}")


if __name__ == "__main__":
    main()
