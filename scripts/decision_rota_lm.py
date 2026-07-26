"""decisión rota sobre un transformer de lenguaje (pythia-410m).

clona la lógica de ``decision_rota.py`` --par más redundante y top-k
de poda por redundancia media, bajo gauge-flip valor-salida-- sobre
la anatomía gpt-neox: ``query_key_value`` fusiona q, k y v por cabeza
(orden [q_h|k_h|v_h] intercalado, no por-bloque como en timm), y
``dense`` es la proyección de salida (el w_o del paper). la decisión
por firma no se porta: exige forwards con probe y queda fuera de
alcance de este bloque.

sanity obligatorio: el troceo se contrasta contra un forward real de
la capa (la entrada a ``query_key_value`` está normalizada por
``input_layernorm``, no son los embeddings crudos) antes de fiarse de
él para nada más.

uso:
    python scripts/decision_rota_lm.py [--sanity] [--capas N]
"""

import argparse
import copy
import csv
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.decision_rota import decisiones

MODELO_ID = "EleutherAI/pythia-410m"
SALIDA = "artifacts/logs/decision_rota/decision_rota_lm.csv"
ESCALA_FUERZA = 8.0    # misma fuerza saturada que ESCALA_FUERZA del ViT
N_GAUGES = 5
K_PODA = 4              # 25% de 16 cabezas: mismo presupuesto RELATIVO
                        # que el top-3/12 (25%) del ancla ViT-B


def carga_modelo(dtype: torch.dtype = torch.float32):
    """carga pythia-410m y verifica su anatomía contra el config real.

    Args:
        dtype: precisión de carga.

    Returns:
        el modelo en modo evaluación.
    """
    from transformers import AutoModelForCausalLM

    modelo = AutoModelForCausalLM.from_pretrained(
        MODELO_ID, dtype=dtype, attn_implementation="eager")
    modelo.eval()
    cfg = modelo.config
    dim_cabeza = cfg.hidden_size // cfg.num_attention_heads
    assert dim_cabeza == 64, f"dim_cabeza inesperada: {dim_cabeza}"
    assert cfg.num_attention_heads == 16, "num_attention_heads != 16"
    assert cfg.hidden_size == 1024, "hidden_size != 1024"
    assert len(modelo.gpt_neox.layers) == 24, "num capas != 24"
    return modelo


def w_v_por_cabeza(
    attn, h_idx: int, n_cabezas: int, dim_cabeza: int,
) -> torch.Tensor:
    """filas de valor w_v^(h) desde el qkv fusionado de gpt-neox.

    el orden es por-cabeza (no por-bloque como en timm): cada cabeza
    ocupa ``3*dim_cabeza`` filas contiguas ``[q_h|k_h|v_h]``; v_h es
    el último tercio de ese bloque.

    Args:
        attn: módulo de atención gpt-neox de una capa.
        h_idx: índice de cabeza.
        n_cabezas: cabezas h (sin usar, se mantiene por simetría de
            firma con la versión ViT).
        dim_cabeza: d_h.

    Returns:
        tensor [dh, d] con w_v^(h).
    """
    w = attn.query_key_value.weight  # [3d, d]
    base = h_idx * 3 * dim_cabeza + 2 * dim_cabeza
    return w[base:base + dim_cabeza, :]


def b_v_por_cabeza(attn, h_idx: int, dim_cabeza: int) -> torch.Tensor:
    """sesgo de valor b_v^(h) desde el qkv fusionado.

    Args:
        attn: módulo de atención gpt-neox de una capa.
        h_idx: índice de cabeza.
        dim_cabeza: d_h.

    Returns:
        tensor [dh] con b_v^(h).
    """
    b = attn.query_key_value.bias
    base = h_idx * 3 * dim_cabeza + 2 * dim_cabeza
    return b[base:base + dim_cabeza]


def w_o_por_cabeza(attn, h_idx: int, dim_cabeza: int) -> torch.Tensor:
    """proyección de salida w_o^(h) desde ``dense``.

    Args:
        attn: módulo de atención gpt-neox de una capa.
        h_idx: índice de cabeza.
        dim_cabeza: d_h.

    Returns:
        tensor [d, dh] con w_o^(h) (columnas de ``dense.weight``).
    """
    w = attn.dense.weight  # [d, d]
    return w[:, h_idx * dim_cabeza:(h_idx + 1) * dim_cabeza]


@torch.no_grad()
def sanity_troceo(modelo, capa: int, n_cabezas: int, dim_cabeza: int,
                  tol: float = 1.0e-5) -> None:
    """reconstruye la salida de atención de un forward pequeño.

    compara el troceo de v (desde ``query_key_value``) y de w_o
    (desde ``dense``) contra las activaciones reales, capturadas con
    hooks. si no cuadra, el layout está mal partido y nada de lo
    posterior vale.

    Args:
        modelo: el modelo cargado.
        capa: índice de capa a comprobar.
        n_cabezas: cabezas h.
        dim_cabeza: d_h.
        tol: tolerancia absoluta (fp32).

    Raises:
        AssertionError: si el error de reconstrucción excede ``tol``.
    """
    attn = modelo.gpt_neox.layers[capa].attention
    capturado: dict[str, torch.Tensor] = {}

    def hook_qkv_in(_m, inp):
        capturado["qkv_in"] = inp[0].detach()

    def hook_qkv_out(_m, _inp, out):
        capturado["qkv_out"] = out.detach()

    def hook_dense_in(_m, inp):
        capturado["dense_in"] = inp[0].detach()

    def hook_dense_out(_m, _inp, out):
        capturado["dense_out"] = out.detach()

    handles = [
        attn.query_key_value.register_forward_pre_hook(hook_qkv_in),
        attn.query_key_value.register_forward_hook(hook_qkv_out),
        attn.dense.register_forward_pre_hook(hook_dense_in),
        attn.dense.register_forward_hook(hook_dense_out),
    ]
    x = torch.randn(1, 5, modelo.config.hidden_size)
    pos_ids = torch.arange(5).unsqueeze(0)
    modelo.gpt_neox(inputs_embeds=x, position_ids=pos_ids)
    for h in handles:
        h.remove()

    qkv_in, qkv_out = capturado["qkv_in"], capturado["qkv_out"]
    dense_in, dense_out = capturado["dense_in"], capturado["dense_out"]

    b, n, _ = qkv_out.shape
    qkv_view = qkv_out.view(
        b, n, n_cabezas, 3 * dim_cabeza).transpose(1, 2)
    _, _, v_real = qkv_view.chunk(3, dim=-1)  # cada [b,H,n,dh]

    err_v = 0.0
    for h in range(n_cabezas):
        w_v = w_v_por_cabeza(attn, h, n_cabezas, dim_cabeza)
        b_v = b_v_por_cabeza(attn, h, dim_cabeza)
        v_mio = qkv_in @ w_v.t() + b_v
        err_v = max(err_v, (v_mio - v_real[:, h]).abs().max().item())

    recon = torch.zeros_like(dense_out)
    for h in range(n_cabezas):
        w_o = w_o_por_cabeza(attn, h, dim_cabeza)
        chunk = dense_in[..., h * dim_cabeza:(h + 1) * dim_cabeza]
        recon = recon + chunk @ w_o.t()
    recon = recon + attn.dense.bias
    err_o = (recon - dense_out).abs().max().item()

    print(f"  [sanity capa {capa}] err_v={err_v:.2e}  err_o={err_o:.2e}")
    assert err_v < tol, f"troceo de V roto en capa {capa}: {err_v:.2e}"
    assert err_o < tol, f"troceo de O roto en capa {capa}: {err_o:.2e}"


@torch.no_grad()
def aplica_gauge_ov_lm(
    attn,
    n_cabezas: int,
    dim_cabeza: int,
    semilla: int = 0,
    escala_id: float | None = None,
) -> float:
    """gauge valor-salida por cabeza sobre gpt-neox, in situ.

    misma operación que ``src.gauge_flip.aplica_gauge_ov`` sobre el
    ViT (w_v <- r^t w_v, b_v <- r^t b_v, w_o <- w_o r^{-t}), con los
    índices de fila/columna propios del layout de gpt-neox. no se
    reutiliza la función del ViT: está acoplada al layout por-bloque
    de timm (``base = 2*d``), incompatible con el layout por-cabeza
    de gpt-neox.

    Args:
        attn: módulo de atención de UNA capa, modificado in situ.
        n_cabezas: cabezas h.
        dim_cabeza: d_h.
        semilla: semilla de la r aleatoria.
        escala_id: coeficiente de la identidad en r = randn + escala_id*i.

    Returns:
        desviación media de r respecto a su mejor múltiplo escalar de
        la identidad, sobre las cabezas.
    """
    if escala_id is None:
        escala_id = dim_cabeza ** 0.5
    g = torch.Generator(device="cpu").manual_seed(semilla)
    w_qkv = attn.query_key_value.weight
    b_qkv = attn.query_key_value.bias
    w_dense = attn.dense.weight
    ident = torch.eye(dim_cabeza, dtype=torch.float64)
    desvs = []
    for h in range(n_cabezas):
        rt = torch.randn(dim_cabeza, dim_cabeza, generator=g,
                         dtype=torch.float64)
        r = rt + escala_id * ident
        s = r.diagonal().mean()
        desvs.append(float((r - s * ident).norm()
                           / (s.abs() * dim_cabeza ** 0.5 + 1.0e-8)))
        r_inv_t = torch.linalg.inv(r).t()
        base = h * 3 * dim_cabeza + 2 * dim_cabeza
        fil = slice(base, base + dim_cabeza)
        col = slice(h * dim_cabeza, (h + 1) * dim_cabeza)
        w_qkv[fil, :] = (r.t() @ w_qkv[fil, :].double()).to(w_qkv.dtype)
        b_qkv[fil] = (r.t() @ b_qkv[fil].double()).to(b_qkv.dtype)
        w_dense[:, col] = (
            w_dense[:, col].double() @ r_inv_t).to(w_dense.dtype)
    return float(sum(desvs) / len(desvs))


@torch.no_grad()
def circuito_ov(attn, n_cabezas: int, dim_cabeza: int) -> torch.Tensor:
    """circuito ov por cabeza m_h = w_v^(h)T w_o^(h), invariante de gauge.

    Args:
        attn: módulo de atención.
        n_cabezas: cabezas h.
        dim_cabeza: d_h.

    Returns:
        tensor [h, d, d] con el circuito por cabeza.
    """
    w_v = torch.stack([
        w_v_por_cabeza(attn, h, n_cabezas, dim_cabeza).double()
        for h in range(n_cabezas)])
    w_o = torch.stack([
        w_o_por_cabeza(attn, h, dim_cabeza).double()
        for h in range(n_cabezas)])
    return torch.einsum("hed,hfe->hdf", w_v, w_o)


@torch.no_grad()
def v1_wo_por_cabeza(attn, n_cabezas: int, dim_cabeza: int) -> torch.Tensor:
    """primer vector singular derecho de w_o^(h), por cabeza.

    Args:
        attn: módulo de atención.
        n_cabezas: cabezas h.
        dim_cabeza: d_h.

    Returns:
        tensor [h, d] con v_1(w_o^(h)) unitario por cabeza.
    """
    dirs = []
    for h in range(n_cabezas):
        w_o = w_o_por_cabeza(attn, h, dim_cabeza)  # [d, dh]
        vh = torch.linalg.svd(w_o, full_matrices=False).U[:, 0]
        dirs.append(vh)
    return torch.stack(dirs)


def main() -> None:
    """corre la decisión rota sobre pythia-410m y guarda el csv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanity", action="store_true",
                        help="solo corre el sanity del troceo y sale")
    parser.add_argument("--capas", type=int, default=None,
                        help="limita a las primeras N capas (debug)")
    args = parser.parse_args()

    print(f"[carga] {MODELO_ID}")
    modelo = carga_modelo()
    cfg = modelo.config
    n_cabezas = cfg.num_attention_heads
    dim_cabeza = cfg.hidden_size // n_cabezas
    n_capas = len(modelo.gpt_neox.layers)
    print(f"  L={n_capas}  H={n_cabezas}  d={cfg.hidden_size}  "
          f"d_h={dim_cabeza}")

    print("[sanity] verificando troceo contra forward real "
          "(capas 0, 11, 23)")
    for capa in (0, n_capas // 2, n_capas - 1):
        sanity_troceo(modelo, capa, n_cabezas, dim_cabeza)
    print("[sanity] OK: troceo de V y O reproduce el forward real")
    if args.sanity:
        return

    # el resto del bloque es forma cerrada sobre los pesos (sin
    # forward): se pasa el modelo a fp64 para que el gauge no dependa
    # de la suerte del numero de condicion de R al castear a fp32 (un
    # R casi singular en una cabeza concreta amplifica el redondeo de
    # ~1e-7 relativo a ~1e-5 al escribir en fp32; en fp64 puro la
    # deriva del circuito queda a precision de maquina, ~1e-13, igual
    # que en el ViT).
    modelo = modelo.double()

    n_capas_correr = args.capas or n_capas
    filas: list[dict] = []
    for capa in tqdm(range(n_capas_correr), desc="capas"):
        attn = modelo.gpt_neox.layers[capa].attention
        circ0 = circuito_ov(attn, n_cabezas, dim_cabeza)
        v1_0 = v1_wo_por_cabeza(attn, n_cabezas, dim_cabeza)
        par0, topk0 = decisiones(v1_0)
        filas.append({
            "arch": "pythia410m", "capa": capa, "gauge_idx": 0,
            "criterio": "pesos", "par_top": str(par0),
            "topk": str(topk0), "estable_par": True,
            "solape_topk": 1.0, "k_poda": K_PODA})

        for g in range(N_GAUGES):
            attn2 = copy.deepcopy(attn)
            desv = aplica_gauge_ov_lm(
                attn2, n_cabezas, dim_cabeza,
                semilla=1000 * capa + g, escala_id=ESCALA_FUERZA)
            circ1 = circuito_ov(attn2, n_cabezas, dim_cabeza)
            deriva_ov = (circ1 - circ0).norm() / circ0.norm().clamp_min(
                1.0e-12)
            assert deriva_ov < 1.0e-9, (
                f"circuito OV no invariante en capa {capa}, "
                f"gauge {g}: deriva={deriva_ov:.2e}")
            v1_g = v1_wo_por_cabeza(attn2, n_cabezas, dim_cabeza)
            par1, topk1 = decisiones(v1_g)
            solape = len(set(topk0) & set(topk1)) / K_PODA
            filas.append({
                "arch": "pythia410m", "capa": capa,
                "gauge_idx": g + 1, "criterio": "pesos",
                "par_top": str(par1), "topk": str(topk1),
                "estable_par": par1 == par0,
                "solape_topk": round(solape, 4), "k_poda": K_PODA,
                "desv_r": round(desv, 4)})
            del attn2

    out = Path(SALIDA)
    out.parent.mkdir(parents=True, exist_ok=True)
    campos = sorted({k for f in filas for k in f})
    with out.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=campos)
        wr.writeheader()
        wr.writerows(filas)
    print(f"[guardado] {out}  ({len(filas)} filas)")

    contador = Counter((f["arch"], f["capa"], f["gauge_idx"])
                       for f in filas)
    dup = {k: v for k, v in contador.items() if v > 1}
    print(f"[counter] claves duplicadas: {len(dup)} "
          f"(esperado 0)")
    assert not dup, f"claves duplicadas en el csv: {dup}"

    rng = random.Random(0)
    print("\n15 observaciones aleatorias:")
    for fila in rng.sample(filas, min(15, len(filas))):
        print(fila)

    con_g = [f for f in filas if f["gauge_idx"] > 0]
    pct = 100.0 * sum(not f["estable_par"] for f in con_g) / len(con_g)
    sol = sum(f["solape_topk"] for f in con_g) / len(con_g)
    print(f"\n[lectura ciega, k_poda={K_PODA}] "
          f"par cambia: {pct:.1f}%  solape top-{K_PODA}: {sol:.3f}")
    inestable = pct >= 40.0 or sol < 0.67
    print(f"[preregistrado] esperado INESTABLE "
          f"(par>=40% o solape<0,67): "
          f"{'CONFIRMADO inestable' if inestable else 'ESTABLE (no esperado, PARAR y reportar)'}")


if __name__ == "__main__":
    main()
