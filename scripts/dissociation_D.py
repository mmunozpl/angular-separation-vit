"""Sonda D — disociación W_O / Q·K (geometría no traza función).

Test de la tesis v3: r(s_func, s_WO) ≈ 0 mientras r(s_func, s_QK) > 0,
sobre la BASE (geometría no manipulada; test primario) y la BLANDA
(descriptivo; replica el 0,33 de v2). Diseño congelado en pipeline_v3.md
§4.D (bloques fechados 05-jun); estatus: replicación exploratoria.

TRES SALVAGUARDAS DE IMPLEMENTACIÓN (pipeline_v3.md §4.D):
  1. s_func se computa UNA vez por (modelo, capa) y alimenta TODAS las
     correlaciones y TODOS los k de la curva. No se recomputa nunca.
     Idem s_QK. Solo s_WO/s_Wq varían con k.
  2. Williams dependiente reporta las TRES correlaciones (r_WO, r_QK,
     r_WO_QK) además del p — para distinguir "diferencia pequeña" de
     "r_WO_QK alta mató la potencia".
  3. Agregado canónico = PROMEDIO de r intra-capa (opción b), con
     leave-deep-out aplicado a las r de capa (excluir 10-11), NO a los
     pares.

Curva de k = cierre de rango-1: s_WO a k=1,4,8,16,64; s_func y s_QK
fijos. Wq confirmatorio (mismo k). Williams por capa (n=66 pares).
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.carga import (ARQUITECTURAS, cargar_pesos_en_modelo,
                       cargar_probe_loader)
from src.metrics.attention import functional_similarity_matrix
from src.models.vit_backbone import HeadProjections, capture_attention
from scripts.principal_angles import subspace_similarity

SEEDS = [42, 43, 44, 45, 46]
KS = [1, 4, 8, 16, 64]
DEVICE = "cuda"


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman rho (correlación de rangos) entre dos vectores 1D.

    el claim es de predominio monótono ---la organización se reubica
    hacia q·k con la profundidad---, no de relación lineal; las
    variables son similitudes acotadas (cosenos, ángulos principales)
    con distribuciones sesgadas. spearman capta la monotonía sin
    asumir forma funcional y es robusto a los extremos. scipy resuelve
    los empates por rango promedio.
    """
    from scipy.stats import spearmanr
    xv = x.flatten().cpu().numpy()
    yv = y.flatten().cpu().numpy()
    rho, _ = spearmanr(xv, yv)
    return float(rho)


def _williams(r12: float, r13: float, r23: float, n: int) -> float:
    """p (dos colas) del test de Williams: r12 vs r13 con var 1 común.

    var 1 = s_func, 2 = W_O, 3 = Q·K. r12=r(func,WO), r13=r(func,QK),
    r23=r(WO,QK). df = n-3.
    """
    if any(map(lambda v: v != v, (r12, r13, r23))) or n <= 3:
        return float("nan")
    from scipy import stats as st
    detR = (1.0 - r12 ** 2 - r13 ** 2 - r23 ** 2
            + 2.0 * r12 * r13 * r23)
    avg = (r12 + r13) / 2.0
    num = (r12 - r13) * math.sqrt((n - 1) * (1.0 + r23))
    den_arg = (2.0 * ((n - 1) / (n - 3)) * detR
               + avg ** 2 * (1.0 - r23) ** 3)
    if den_arg <= 1.0e-12:        # matriz de correlación degenerada
        return float("nan")
    t = num / math.sqrt(den_arg)
    return float(2.0 * st.t.sf(abs(t), df=n - 3))


def _upper(mat: torch.Tensor) -> torch.Tensor:
    """66 valores del triángulo superior estricto de una (H,H)."""
    h = mat.shape[0]
    iu = torch.triu_indices(h, h, offset=1)
    return mat[iu[0], iu[1]].cpu()


def _subspace_pairs(w_layer: torch.Tensor, k: int) -> torch.Tensor:
    """s_subespacio de los 66 pares de cabezas a dimensión k.

    Args:
        w_layer: (D, H, hd) — submatriz por cabeza (W_O o Wq^T).
        k: dimensión del subespacio top-k (vectores singulares izq.).
    """
    d, h, hd = w_layer.shape
    bases = []
    for hi in range(h):
        u, _, _ = torch.linalg.svd(
            w_layer[:, hi, :].float(), full_matrices=False)
        bases.append(u[:, :k])
    vals = torch.zeros(h, h)
    for i in range(h):
        for j in range(i + 1, h):
            s, _ = subspace_similarity(bases[i], bases[j])
            vals[i, j] = s
    return _upper(vals)


def _capture(model, loader):
    """Forward único: atención post-softmax y logits QK pre-softmax.

    Returns:
        (avg_attn, avg_logits) cada uno (L, H, T, T), promediado sobre
        el probe set. UN solo pase — misma población para s_func y s_QK.
    """
    base = model.model
    blocks = base.blocks
    d, h, hd = model.embed_dim, model.num_heads, model.head_dim
    qkv_outs: list = []

    def hook(_m, _inp, out):
        qkv_outs.append(out)

    handles = [b.attn.qkv.register_forward_hook(hook) for b in blocks]
    attn_sum = None
    logit_sum = None
    n_imgs = 0
    with capture_attention(base) as maps:
        for imgs, _ in loader:
            maps.clear()
            qkv_outs.clear()
            with torch.no_grad():
                _ = model(imgs.to(DEVICE, non_blocking=True))
            if attn_sum is None:
                attn_sum = [torch.zeros_like(m.sum(0)) for m in maps]
                logit_sum = [None] * len(maps)
            for li, m in enumerate(maps):
                attn_sum[li] += m.sum(0)
            for li, q_out in enumerate(qkv_outs):
                b_, t_, _ = q_out.shape
                qkv = q_out.view(b_, t_, 3, h, hd).permute(2, 0, 3, 1, 4)
                q, k = qkv[0], qkv[1]            # (B,H,T,hd)
                lg = (q @ k.transpose(-2, -1)).sum(0)   # (H,T,T)
                if logit_sum[li] is None:
                    logit_sum[li] = torch.zeros_like(lg)
                logit_sum[li] += lg
            n_imgs += imgs.shape[0]
    for hh in handles:
        hh.remove()
    avg_attn = torch.stack([a / n_imgs for a in attn_sum])
    avg_logit = torch.stack([l_ / n_imgs for l_ in logit_sum])
    return avg_attn, avg_logit


def _qk_pairs(logit_layer: torch.Tensor) -> torch.Tensor:
    """66 valores sim_QK: coseno entre logits aplanados por cabeza."""
    h = logit_layer.shape[0]
    flat = logit_layer.reshape(h, -1)
    flat = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    return _upper(flat @ flat.t())


def main() -> None:
    """Disociación D sobre base (test) y blanda (descriptivo)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+",
                        default=["base", "blanda"])
    parser.add_argument("--out", default="artifacts/tables/dissociation_D")
    parser.add_argument("--arch", default="vitb",
                        choices=sorted(ARQUITECTURAS),
                        help="arquitectura; fija la columna vit y lee "
                             "de checkpoints/<arch>_clean")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                        help="semillas a leer (piloto: solo 42)")
    args = parser.parse_args()
    arch = args.arch
    seeds = args.seeds

    spec = ARQUITECTURAS[arch]
    img_size = spec["img_size"]
    # a resolución nativa alta (dinov2, 518 -> t=1370) los mapas t x t
    # por capa no caben con lote 64; se reduce el lote, no la resolución
    lote = 64 if img_size <= 224 else 2
    loader, _ = cargar_probe_loader(
        spec["probe"], batch_size=lote, num_workers=4,
        img_size=img_size, crop_pct=spec["crop_pct"],
        interp=spec["interp"])
    model = HeadProjections(
        model_name=spec["model_name"],
        pretrained=False, num_classes=100, img_size=img_size,
    ).to(DEVICE).eval()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in args.variants:
        for seed in seeds:
            ck = Path(
                f"artifacts/checkpoints/{arch}_clean/"
                f"attnA_{variant}_seed{seed}_last.pt")
            cargar_pesos_en_modelo(ck, model, DEVICE)
            avg_attn, avg_logit = _capture(model, loader)
            n_layers = avg_attn.shape[0]
            d, h, hd = model.embed_dim, model.num_heads, model.head_dim
            # pesos estáticos: W_O (D,H,hd) y Wq^T (D,H,hd)
            blocks = model.model.blocks
            for li in range(n_layers):
                # SALVAGUARDA 1: s_func y s_QK se computan UNA vez aquí,
                # fuera del bucle de k.
                s_func = functional_similarity_matrix(avg_attn[li])
                s_func = _upper(s_func)
                s_qk = _qk_pairs(avg_logit[li])
                r_qk = _spearman(s_func, s_qk)
                # W_O: cabezas en el eje de ENTRADA (columnas) -> view
                # parte bien; wo[:,hi,:]=(d,hd) = lo que la cabeza ESCRIBE.
                wo = blocks[li].attn.proj.weight.view(d, h, hd)
                # Wq: cabezas en el eje de SALIDA (filas) -> partir cabezas
                # ANTES de transponer. wq[:,hi,:]=(d,hd) = espacio fila de
                # la query = lo que la cabeza LEE en R^768 (homólogo a W_O).
                wq = (blocks[li].attn.qkv.weight[:d, :]
                      .view(h, hd, d).permute(2, 0, 1).contiguous())
                for k in KS:
                    s_wo = _subspace_pairs(wo, k)
                    s_wq = _subspace_pairs(wq, k)
                    r_wo = _spearman(s_func, s_wo)
                    r_wq = _spearman(s_func, s_wq)
                    r_wo_qk = _spearman(s_wo, s_qk)
                    # williams sobre los rho de spearman: aproximado, se
                    # reporta pero el cruce se apoya en la reproducibilidad
                    # entre semillas, no en este p
                    p_w = _williams(r_wo, r_qk, r_wo_qk, n=s_func.numel())
                    rows.append({
                        "arch": arch,
                        "variant": variant, "seed": seed, "layer": li,
                        "k": k, "r_WO": r_wo, "r_QK": r_qk,
                        "r_Wq": r_wq, "r_WO_QK": r_wo_qk,
                        "williams_p": p_w,
                    })
            print(f"  {variant} seed{seed} listo", flush=True)

    # merge por (arch, variante): reemplaza SOLO lo recomputado en esta
    # invocación y conserva el resto. reemplazar por arch a secas
    # borraba las variantes no pedidas (una corrida --variants base se
    # comió las filas blanda de vitb el 11-07).
    csv_path = out / "by_layer.csv"
    recomputadas = set(args.variants)
    previas = []
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            previas = [r for r in csv.DictReader(f)
                       if not (r.get("arch") == arch
                               and r.get("variant") in recomputadas)]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(previas)
        w.writerows(rows)

    # SALVAGUARDA 3: agregado = promedio de r intra-capa (opción b),
    # leave-deep-out sobre las r de capa.
    print("\n=== agregado (promedio de r intra-capa, por variante/k) ===")
    print(f"{'variant':>8} {'k':>3} {'subset':>9} "
          f"{'r_WO(m±s)':>16} {'r_QK(m±s)':>16} {'r_Wq(m±s)':>16}")
    # leave-deep-out = las dos últimas capas de ESTA arquitectura
    # (10-11 en vitb, 22-23 en 24 capas), no el {10,11} literal
    deep = {n_layers - 2, n_layers - 1}
    et_deep = f"sin{n_layers - 2}-{n_layers - 1}"
    for variant in args.variants:
        for k in KS:
            for subset, layers in (("todas", None), (et_deep, deep)):
                per_seed = {"r_WO": [], "r_QK": [], "r_Wq": []}
                for seed in seeds:
                    sel = [r for r in rows
                           if r["variant"] == variant and r["seed"] == seed
                           and r["k"] == k
                           and (layers is None or r["layer"] not in layers)]
                    for key in per_seed:
                        vals = [r[key] for r in sel if r[key] == r[key]]
                        per_seed[key].append(sum(vals) / len(vals))

                def ms(xs):
                    m = sum(xs) / len(xs)
                    # ddof=1; con n=1 (piloto 1 semilla) la std no está
                    # definida -> 0.0 en vez de dividir por cero
                    s = ((sum((x - m) ** 2 for x in xs)
                          / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0)
                    return f"{m:+.3f}±{s:.3f}"
                print(f"{variant:>8} {k:>3} {subset:>9} "
                      f"{ms(per_seed['r_WO']):>16} "
                      f"{ms(per_seed['r_QK']):>16} "
                      f"{ms(per_seed['r_Wq']):>16}")


if __name__ == "__main__":
    main()
