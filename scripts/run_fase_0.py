"""fase 0 — ángulo computado/estático del portador, por capa.

carga los attnA_base afinados, acumula el gram del contexto sobre el
conjunto congelado y, por capa, compara la firma computada v1(a_h v_h
w_o) con la estática v1(w_o). reporta el ángulo emparejado (computada de
h vs estática de h) contra el nulo de permutación (computada de h vs
estática de h'!=h) y la banda entre las cinco semillas. ambas firmas
viven en el espacio fila de w_o (64-dim), así que el azar es la
permutación dentro de ese subespacio, no el de R^768.

uso:
    python scripts/run_fase_0.py [--arch vitb|vitl]
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.carga import ARQUITECTURAS, cargar_probe_tensor
from src.firma_funcional import CapturaContexto, w_o_por_cabeza
import scripts.run_fase_G as fase_g
from scripts.run_fase_G import carga_modelo, firma_exacta

# --- campos por arquitectura (main los fija según --arch) -----------
CKPTS = sorted(glob.glob(
    "artifacts/checkpoints/vitb_clean/attnA_base_seed*_last.pt"))
PROBE = "artifacts/probe_set/imagenet100_val_1k.pt"
SALIDA = "artifacts/logs/fase_0/computado_estatico.csv"
ARCH = "vitb"  # etiqueta de arquitectura; se sobreescribe con --arch
# --------------------------------------------------------------------
N_CABEZAS, DIM_CABEZA = 12, 64
LOTE = 64  # lote del gram; main lo reduce a resolución nativa alta


@torch.no_grad()
def gram_contexto(
    modelo,
    imgs: torch.Tensor,
    captura: CapturaContexto,
    n_capas: int,
    disp: str,
) -> dict[int, torch.Tensor]:
    """acumula sum (a_h v_h)^t (a_h v_h) por capa sobre el conjunto.

    el gram en d_h basta para derivar luego la firma computada exacta sin
    materializar la matriz d x d.

    args:
        modelo: el vit en eval.
        imgs: tensor [n, c, h, w] del conjunto congelado.
        captura: hook de contexto por capa.
        n_capas: número de capas.
        disp: dispositivo.

    returns:
        dict capa -> tensor [h, dh, dh].
    """
    gram = {l: torch.zeros(N_CABEZAS, DIM_CABEZA, DIM_CABEZA, device=disp)
            for l in range(n_capas)}
    for ini in tqdm(range(0, imgs.shape[0], LOTE), desc="gram",
                    leave=False):
        captura.limpiar()
        _ = modelo(imgs[ini:ini + LOTE].to(disp))
        for l in range(n_capas):
            g = captura.contexto[l][:, 1:, :, :].permute(2, 0, 1, 3)
            g = g.reshape(N_CABEZAS, -1, DIM_CABEZA)   # [h, b*p, dh]
            gram[l] = gram[l] + torch.einsum("hnd,hne->hde", g, g)
    return gram


@torch.no_grad()
def firma_computada(
    gram64: torch.Tensor,
    w_o: torch.Tensor,
) -> torch.Tensor:
    """firma computada exacta v1(a_h v_h w_o), vía el gram en d_h.

    con g = l l^t (eigh), la firma es vh[0] de (l^t w_o), svd exacta de
    una matriz dh x d; no se materializa d x d.

    args:
        gram64: tensor [h, dh, dh] con sum (a_h v_h)^t (a_h v_h).
        w_o: tensor [h, dh, d].

    returns:
        tensor [h, d] con la firma computada unitaria por cabeza.
    """
    out = []
    for h in range(gram64.shape[0]):
        val, vec = torch.linalg.eigh(gram64[h])      # g simétrica psd
        ele = vec * val.clamp_min(0).sqrt()          # g = ele ele^t
        b = ele.t() @ w_o[h]                          # [dh, d]
        out.append(torch.linalg.svd(b, full_matrices=False).Vh[0])
    return torch.stack(out)


def main() -> None:
    """ángulo computado/estático por capa, con nulo de permutación."""
    global CKPTS, PROBE, ARCH, N_CABEZAS, DIM_CABEZA, LOTE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="vitb",
                        choices=sorted(ARQUITECTURAS),
                        help="arquitectura: fija checkpoints, etiqueta "
                             "del csv, geometría y resolución nativa")
    args = parser.parse_args()
    ARCH = args.arch
    spec = ARQUITECTURAS[ARCH]
    N_CABEZAS, DIM_CABEZA = spec["n_cabezas"], spec["dim_cabeza"]
    PROBE = spec["probe"]
    # el hook de contexto guarda [b, t, h, dh] por capa: a 518 (t=1370,
    # 24 capas) el lote 64 no cabe; se reduce el lote, no la resolución
    if spec["img_size"] > 224:
        LOTE = 8
    # carga_modelo vive en run_fase_G y lee sus globales al llamarse
    fase_g.MODEL_NAME = spec["model_name"]
    fase_g.IMG_SIZE = spec["img_size"]
    CKPTS = sorted(glob.glob(
        f"artifacts/checkpoints/{ARCH}_clean/attnA_base_seed*_last.pt"))
    if not CKPTS:
        raise SystemExit(f"sin checkpoints base para {ARCH}")
    disp = "cuda" if torch.cuda.is_available() else "cpu"
    imgs = cargar_probe_tensor(PROBE, img_size=spec["img_size"],
                               crop_pct=spec["crop_pct"],
                               interp=spec["interp"])
    fuera = ~torch.eye(N_CABEZAS, dtype=torch.bool, device=disp)
    filas: list[dict] = []
    for ckpt in tqdm(CKPTS, desc="semillas"):
        modelo = carga_modelo(ckpt).to(disp).eval()
        n_capas = len(modelo.blocks)
        captura = CapturaContexto(modelo, N_CABEZAS, DIM_CABEZA)
        gram = gram_contexto(modelo, imgs, captura, n_capas, disp)
        captura.quitar()
        for capa in range(n_capas):
            w_o = w_o_por_cabeza(modelo, capa, N_CABEZAS, DIM_CABEZA)
            comp = firma_computada(gram[capa], w_o)       # [h, d]
            estat = firma_exacta(w_o, "der")              # [h, d]
            cos = (comp @ estat.t()).abs().clamp(max=1.0)  # [h, h]
            ang = torch.rad2deg(torch.arccos(cos))         # [h, h]
            emparejado = float(torch.diagonal(ang).mean())
            nulo = float(ang[fuera].mean())               # permutación
            filas.append({"seed": ckpt, "capa": capa,
                          "ang_emparejado": emparejado,
                          "ang_nulo": nulo})
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

    # banda entre semillas por capa: emparejado vs nulo de permutación
    g = df.groupby("capa")[["ang_emparejado", "ang_nulo"]]
    res = g.agg(["mean", "std"])
    print("\ncapa | emparejado (media±sd) | nulo (media±sd)")
    for capa, fila in res.iterrows():
        em = fila[("ang_emparejado", "mean")]
        es = fila[("ang_emparejado", "std")]
        nm = fila[("ang_nulo", "mean")]
        ns = fila[("ang_nulo", "std")]
        print(f"{capa:>4} | {em:5.1f} ± {es:4.1f} | "
              f"{nm:5.1f} ± {ns:4.1f}")


if __name__ == "__main__":
    main()
