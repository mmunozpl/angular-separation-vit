"""demo de decisión rota: la poda por v1(w_o) cambia bajo gauge.

materializa la decisión que la práctica toma ---par más redundante y
top-3 de poda por redundancia media--- con dos criterios: las
direcciones estáticas v1(w_o) (gauge-variantes) y las firmas de
respuesta v1(c_h^p) sobre el probe congelado (gauge-invariantes). se
aplican por capa cinco gauges valor-salida a la fuerza saturada
(escala_id=8.0 -> desv_r ~ 1.0, la de fase g; semillas de r
1000*capa+j, j=0..4, la misma familia de fase g), fijados antes de ver
resultados, y se mide si la decisión cambia. la firma post-gauge se
recomputa con forward real sobre el probe (comprobación empírica, no
algebraica).

uso:
    python scripts/decision_rota.py [--sanity] [--seeds 42 ...]

sanity: escala_id=1e6 (r ~ múltiplo de la identidad) -> par y ranking
deben quedar idénticos a sin-gauge en pesos y en firma; si no, el
pipeline está roto, no el resultado.
"""

import argparse
import copy
import csv
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_fase_G import carga_modelo, firma_exacta
from scripts.run_fase_0 import firma_computada, gram_contexto
from src.carga import cargar_probe_tensor
from src.firma_funcional import CapturaContexto, w_o_por_cabeza
from src.gauge_flip import aplica_gauge_ov

PROBE = "artifacts/probe_set/imagenet100_val_1k.pt"
SALIDA = "artifacts/logs/decision_rota/decision_rota.csv"
N_CABEZAS, DIM_CABEZA = 12, 64
ESCALA_FUERZA = 8.0    # desv_r ~ 1.0, régimen saturado (tab:gauge)
N_GAUGES = 5
K_PODA = 3


def decisiones(firmas: torch.Tensor) -> tuple[tuple[int, int],
                                              list[int]]:
    """par más redundante y top-k de poda desde unas firmas.

    Args:
        firmas: tensor [h, d] de direcciones unitarias por cabeza.

    Returns:
        tupla (par_top ordenado, top-k de cabezas por redundancia
        media descendente).
    """
    c = (firmas @ firmas.t()).abs().clamp(max=1.0)
    c.fill_diagonal_(0.0)
    h = c.shape[0]
    idx = int(torch.argmax(c).item())
    par = tuple(sorted((idx // h, idx % h)))
    red_media = c.sum(dim=1) / (h - 1)
    topk = torch.argsort(red_media, descending=True)[:K_PODA]
    return par, [int(i) for i in topk]


def grams_del_modelo(
    modelo,
    imgs: torch.Tensor,
    disp: str,
) -> dict[int, torch.Tensor]:
    """gram del contexto a_h v_h por capa, con un solo forward.

    Args:
        modelo: el vit en eval sobre disp.
        imgs: probe congelado [n, 3, 224, 224] en cpu.
        disp: dispositivo.

    Returns:
        dict capa -> tensor [h, dh, dh].
    """
    captura = CapturaContexto(modelo, N_CABEZAS, DIM_CABEZA)
    gram = gram_contexto(modelo, imgs, captura,
                         len(modelo.blocks), disp)
    captura.quitar()
    return gram


def par_de_firmas(
    modelo,
    gram_capa: torch.Tensor,
    capa: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """v1(w_o) estática y firma computada v1(c_h^p) de una capa.

    Args:
        modelo: el vit.
        gram_capa: gram [h, dh, dh] de esa capa.
        capa: índice de la capa.

    Returns:
        tupla (v1_wo [h, d], firma_computada [h, d]).
    """
    w_o = w_o_por_cabeza(modelo, capa, N_CABEZAS, DIM_CABEZA)
    return firma_exacta(w_o, "der"), firma_computada(gram_capa, w_o)


def main() -> None:
    """corre la demo de decisión rota y guarda el csv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 43, 44, 45, 46])
    parser.add_argument("--sanity", action="store_true",
                        help="gauges ~identidad: nada debe cambiar")
    args = parser.parse_args()
    escala = 1.0e6 if args.sanity else ESCALA_FUERZA

    disp = "cuda" if torch.cuda.is_available() else "cpu"
    imgs = cargar_probe_tensor(PROBE)
    filas: list[dict] = []
    for seed in tqdm(args.seeds, desc="semillas"):
        ckpt = (f"artifacts/checkpoints/vitb_clean/"
                f"attnA_base_seed{seed}_last.pt")
        modelo = carga_modelo(ckpt).to(disp).eval()
        gram0 = grams_del_modelo(modelo, imgs, disp)
        base: dict[int, dict] = {}
        for capa in range(len(modelo.blocks)):
            v1_wo, firma = par_de_firmas(modelo, gram0[capa], capa)
            base[capa] = {"pesos": decisiones(v1_wo),
                          "firma": decisiones(firma)}
            for crit in ("pesos", "firma"):
                par, topk = base[capa][crit]
                filas.append({
                    "seed": seed, "capa": capa, "gauge_idx": 0,
                    "criterio": crit, "par_top": str(par),
                    "topk": str(topk), "estable_par": True,
                    "solape_topk": 1.0})
        for capa in tqdm(range(len(modelo.blocks)), desc="capas",
                         leave=False):
            for g in range(N_GAUGES):
                m2 = copy.deepcopy(modelo)
                aplica_gauge_ov(m2, capa, N_CABEZAS, DIM_CABEZA,
                                semilla=1000 * capa + g,
                                escala_id=escala)
                gram_g = grams_del_modelo(m2, imgs, disp)
                v1_g, firma_g = par_de_firmas(m2, gram_g[capa], capa)
                post = {"pesos": decisiones(v1_g),
                        "firma": decisiones(firma_g)}
                for crit in ("pesos", "firma"):
                    par0, topk0 = base[capa][crit]
                    par1, topk1 = post[crit]
                    solape = len(set(topk0) & set(topk1)) / K_PODA
                    filas.append({
                        "seed": seed, "capa": capa,
                        "gauge_idx": g + 1, "criterio": crit,
                        "par_top": str(par1), "topk": str(topk1),
                        "estable_par": par1 == par0,
                        "solape_topk": round(solape, 4)})
                del m2
        del modelo
        torch.cuda.empty_cache()

    out = Path(SALIDA)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        wr.writeheader()
        wr.writerows(filas)
    print(f"[guardado] {out}  ({len(filas)} filas)")

    # 15 observaciones aleatorias del artefacto
    rng = random.Random(0)
    for fila in rng.sample(filas, min(15, len(filas))):
        print(fila)

    # agregado para tab:decision (solo filas con gauge)
    con_g = [f for f in filas if f["gauge_idx"] > 0]
    print("\ncriterio | % par cambia | solape top-3 medio")
    for crit in ("pesos", "firma"):
        sel = [f for f in con_g if f["criterio"] == crit]
        pct = 100.0 * sum(not f["estable_par"] for f in sel) / len(sel)
        sol = sum(f["solape_topk"] for f in sel) / len(sel)
        print(f"{crit:>8} | {pct:12.1f} | {sol:.3f}")


if __name__ == "__main__":
    main()
