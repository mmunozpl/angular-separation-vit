"""f6 — extrae el sector valor-salida y deriva el payload de la demo.

la demo de la órbita de gauge no necesita servidor: el objeto que se
manipula es de 64x64 por cabeza. con w_o = m b^t (svd reducida,
m = u sigma de 64x64 y b la base ortonormal del espacio fila), un
gauge r actúa solo sobre m ---m <- r^{-1} m--- y v1 en coordenadas es
el primer vector singular derecho de esa matriz pequeña. el coseno
entre cabezas sale de los gramianos cruzados g_hh' = b_h^t b_h', que
también son 64x64. de ahí que todo el payload quepa en pocos mb y la
aritmética corra en el navegador.

salidas:
  artifacts/demo/sector_vo_<col>.safetensors  referencia completa
      (w_v, b_v, w_o por cabeza; el sesgo viaja porque el gauge lo
      transforma y sin él la salida no es invariante)
  artifacts/demo/payload_<col>.safetensors    reducido para js
      (m_h y los gramianos por par, por capa)
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.carga import ARQUITECTURAS, cargar_vit_base  # noqa: E402

CKPT_VITB = "artifacts/checkpoints/vitb_clean/attnA_base_seed42_last.pt"
MODELO_LM = "EleutherAI/pythia-410m"
SALIDA = "artifacts/demo"


def sector_vit(modelo, capa: int, h: int, dh: int) -> dict:
    """w_v, b_v y w_o de una cabeza del vit, convención del paper.

    Args:
        modelo: vit interno de timm.
        capa: índice de bloque.
        h: índice de cabeza.
        dh: dimensión por cabeza.

    Returns:
        dict con w_v [dh, d], b_v [dh] y w_o [dh, d].
    """
    attn = modelo.blocks[capa].attn
    w = attn.qkv.weight.detach()
    base = 2 * w.shape[1] + h * dh
    fil = slice(base, base + dh)
    col = slice(h * dh, (h + 1) * dh)
    sesgo = (attn.qkv.bias.detach()[fil] if attn.qkv.bias is not None
             else torch.zeros(dh))
    return {"w_v": w[fil, :].clone(), "b_v": sesgo.clone(),
            "w_o": attn.proj.weight.detach()[:, col].t().contiguous()}


def sector_lm(attn, h: int, dh: int) -> dict:
    """ídem para gpt-neox, cuyo qkv se ordena por cabeza.

    cada cabeza ocupa 3*dh filas contiguas [q_h|k_h|v_h]; el valor es
    el último tercio. anatomía verificada en decision_rota_lm.

    Args:
        attn: módulo de atención de una capa gpt-neox.
        h: índice de cabeza.
        dh: dimensión por cabeza.

    Returns:
        dict con w_v [dh, d], b_v [dh] y w_o [dh, d].
    """
    base = h * 3 * dh + 2 * dh
    fil = slice(base, base + dh)
    col = slice(h * dh, (h + 1) * dh)
    return {"w_v": attn.query_key_value.weight.detach()[fil, :].clone(),
            "b_v": attn.query_key_value.bias.detach()[fil].clone(),
            "w_o": attn.dense.weight.detach()[:, col].t().contiguous()}


def payload_capa(w_o_capa: torch.Tensor) -> dict:
    """deriva m_h y los gramianos cruzados de una capa.

    Args:
        w_o_capa: tensor [h, dh, d] con w_o por cabeza.

    Returns:
        dict con m [h, dh, dh] y gram [n_pares, dh, dh].
    """
    u, s, vh = torch.linalg.svd(w_o_capa.double(), full_matrices=False)
    m = u @ torch.diag_embed(s)                 # [h, dh, dh]
    b = vh.transpose(1, 2)                      # [h, d, dh]
    n = w_o_capa.shape[0]
    grams = [b[i].t() @ b[j] for i in range(n) for j in range(i + 1, n)]
    return {"m": m.float(), "gram": torch.stack(grams).float()}


def verifica(ref: dict, modelo, arch: str, cfg: dict) -> None:
    """ida y vuelta: lo guardado debe ser lo que el modelo tiene.

    Args:
        ref: tensores extraídos.
        modelo: el modelo del que salieron.
        arch: etiqueta de columna.
        cfg: configuración de la columna.

    Raises:
        AssertionError: si algún tensor no coincide.
    """
    capa, h = 0, 0
    if arch == "pythia":
        vivo = sector_lm(modelo.gpt_neox.layers[capa].attention, h,
                         cfg["dim_cabeza"])
    else:
        vivo = sector_vit(modelo, capa, h, cfg["dim_cabeza"])
    for k, v in vivo.items():
        guardado = ref[f"L{capa}.h{h}.{k}"]
        assert torch.equal(guardado, v), f"{arch}: {k} no coincide"
    print(f"[verificación] {arch}: ida y vuelta OK sobre L0.h0")


def procesa(arch: str, salida: str) -> None:
    """extrae, guarda y deriva el payload de una columna.

    Args:
        arch: vitb o pythia.
        salida: directorio de salida.
    """
    if arch == "pythia":
        from transformers import AutoModelForCausalLM
        modelo = AutoModelForCausalLM.from_pretrained(
            MODELO_LM, dtype=torch.float32,
            attn_implementation="eager").eval()
        cfg = {"n_cabezas": modelo.config.num_attention_heads,
               "dim_cabeza": (modelo.config.hidden_size
                              // modelo.config.num_attention_heads)}
        capas = len(modelo.gpt_neox.layers)
    else:
        cfg = dict(ARQUITECTURAS[arch])
        modelo = cargar_vit_base(
            CKPT_VITB, device="cpu", num_classes=100,
            model_name=cfg["model_name"], img_size=cfg["img_size"])
        capas = len(modelo.blocks)

    dh, nh = cfg["dim_cabeza"], cfg["n_cabezas"]
    ref, pay = {}, {}
    for capa in tqdm(range(capas), desc=f"{arch}", leave=False):
        w_os = []
        for h in range(nh):
            s = (sector_lm(modelo.gpt_neox.layers[capa].attention, h, dh)
                 if arch == "pythia" else sector_vit(modelo, capa, h, dh))
            for k, v in s.items():
                ref[f"L{capa}.h{h}.{k}"] = v
            w_os.append(s["w_o"])
        red = payload_capa(torch.stack(w_os))
        pay[f"L{capa}.m"] = red["m"]
        pay[f"L{capa}.gram"] = red["gram"]

    verifica(ref, modelo, arch, cfg)
    os.makedirs(salida, exist_ok=True)
    f_ref = f"{salida}/sector_vo_{arch}.safetensors"
    f_pay = f"{salida}/payload_{arch}.safetensors"
    save_file(ref, f_ref)
    save_file(pay, f_pay)
    for f in (f_ref, f_pay):
        print(f"[guardado] {f} ({os.path.getsize(f) / 1e6:.1f} MB)")

    # 15 tensores al azar del artefacto, con forma y norma
    recargado = load_file(f_ref)
    g = torch.Generator().manual_seed(0)
    claves = sorted(recargado)
    idx = torch.randperm(len(claves), generator=g)[:15]
    print("\n15 tensores al azar del artefacto:")
    for i in idx:
        k = claves[int(i)]
        v = recargado[k]
        print(f"  {k:22s} {str(tuple(v.shape)):14s} "
              f"norma {v.norm():.4f}")


def main() -> None:
    """punto de entrada."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cols", default="vitb")
    ap.add_argument("--salida", default=SALIDA)
    args = ap.parse_args()
    for arch in args.cols.split(","):
        procesa(arch, args.salida)


if __name__ == "__main__":
    main()
