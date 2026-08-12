"""m2 — inercia de la sonda blanda a nivel de salida.

s_func compara mapas de atención, y a_h = softmax(qk^t) no contiene
w_o por vía directa: medir ahí la inercia de la sonda está cerca de
garantizado por construcción. esto la mide donde el modelo se juega
la función, en los logits sobre el conjunto sonda congelado.

  d_par     = kl(base(s) || blanda(s)), pareado por semilla
  d_suelo   = kl(base(s) || base(s')), s != s', ambas direcciones
  d_control = kl(base(s) || dura(s)), control positivo

umbrales pre-registrados (lock del 10-08-2026, no se tocan):
v-c si mediana(d_par) <= max(d_suelo) y mediana(d_control) >
max(d_suelo); v-d si mediana(d_par) > max(d_suelo); y si el control
no separa, no hay veredicto sobre h1 ---el instrumento no distingue
ni la variante que sabemos dañina---.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.firma_funcional import w_v_columnas  # noqa: E402
import scripts.run_fase_0 as f0  # noqa: E402
from scripts.run_fase_0 import firma_computada, gram_contexto  # noqa: E402
from src.carga import (ARQUITECTURAS, cargar_probe_tensor,  # noqa: E402
                       cargar_vit_base)
from src.firma_funcional import CapturaContexto, w_o_por_cabeza  # noqa: E402

VARIANTES = ("base", "blanda", "dura")
SALIDA = "artifacts/logs/inercia_salida/inercia_salida.csv"


def kl_media(p_log: torch.Tensor, q_log: torch.Tensor) -> tuple[float, float]:
    """kl(p||q) sobre logits, en fp64, media y mediana por imagen.

    Args:
        p_log: logits de referencia [n, c].
        q_log: logits comparados [n, c].

    Returns:
        tupla (media, mediana) de la kl por imagen, en nats.
    """
    p = torch.softmax(p_log.double(), dim=1)
    lp = torch.log_softmax(p_log.double(), dim=1)
    lq = torch.log_softmax(q_log.double(), dim=1)
    kl = (p * (lp - lq)).sum(dim=1)
    return float(kl.mean()), float(kl.median())


def comprueba_carga(modelo, ckpt: str) -> None:
    """verifica que los pesos medidos son los del checkpoint.

    Args:
        modelo: vit ya cargado.
        ckpt: ruta del checkpoint.

    Raises:
        AssertionError: si el tensor vivo no coincide con el fichero.
    """
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = blob.get("model", blob)
    sufijo = "blocks.0.attn.qkv.weight"
    claves = [k for k in sd if k.endswith(sufijo)]
    assert len(claves) == 1, f"{ckpt}: {len(claves)} claves con {sufijo}"
    viva = modelo.blocks[0].attn.qkv.weight.detach().cpu()
    assert torch.equal(sd[claves[0]].cpu(), viva), (
        f"{ckpt}: carga parcial silenciosa")


@torch.no_grad()
def logits_y_firma(
    modelo, imgs: torch.Tensor, disp: str, cfg: dict, con_firma: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """logits sobre la sonda y, si se pide, la firma de respuesta.

    Args:
        modelo: vit en eval sobre disp.
        imgs: sonda congelada [n, 3, s, s] en cpu, orden fijo.
        disp: dispositivo.
        cfg: entrada de ARQUITECTURAS de la columna.
        con_firma: si además se calcula v1(c_h^p) por capa.

    Returns:
        tupla (logits [n, c] en cpu, firmas [capas, h, d] o None).
    """
    salidas = []
    for i in range(0, len(imgs), 64):
        lote = imgs[i:i + 64].to(disp, non_blocking=True)
        salidas.append(modelo(lote).float().cpu())
    logits = torch.cat(salidas)
    if not con_firma:
        return logits, None
    h, dh = cfg["n_cabezas"], cfg["dim_cabeza"]
    # gram_contexto fija h y d_h como constantes de vitb; se ajustan
    # para reusar exactamente su camino de cálculo en vit-l
    prev_h, prev_dh = f0.N_CABEZAS, f0.DIM_CABEZA
    f0.N_CABEZAS, f0.DIM_CABEZA = h, dh
    try:
        captura = CapturaContexto(modelo, h, dh)
        gram = gram_contexto(modelo, imgs, captura,
                             len(modelo.blocks), disp)
        captura.quitar()
    finally:
        f0.N_CABEZAS, f0.DIM_CABEZA = prev_h, prev_dh
    firmas = torch.stack([
        firma_computada(gram[c], w_o_por_cabeza(modelo, c, h, dh))
        for c in range(len(modelo.blocks))])
    return logits, firmas.cpu()


def carga(arch: str, variante: str, seed: int, raiz: str, disp: str):
    """carga un checkpoint de una variante y comprueba la carga.

    Args:
        arch: etiqueta de arquitectura.
        variante: base, blanda o dura.
        seed: semilla.
        raiz: directorio de checkpoints.
        disp: dispositivo.

    Returns:
        el vit en eval sobre disp, o None si el fichero no está.
    """
    cfg = ARQUITECTURAS[arch]
    ckpt = f"{raiz}/{arch}_clean/attnA_{variante}_seed{seed}_last.pt"
    if not os.path.exists(ckpt):
        print(f"[aviso] no está {ckpt}; se salta")
        return None
    m = cargar_vit_base(ckpt, device=disp, num_classes=100,
                        model_name=cfg["model_name"],
                        img_size=cfg["img_size"])
    comprueba_carga(m, ckpt)
    return m.to(disp).eval()


def plomeria(logits: dict, arch: str, seed: int) -> None:
    """autopar: la kl de una corrida consigo misma debe ser nula.

    Args:
        logits: dict (variante, seed) -> logits.
        arch: etiqueta de arquitectura.
        seed: semilla de referencia.

    Raises:
        AssertionError: si el autopar no es ~0.
    """
    m, _ = kl_media(logits[("base", seed)], logits[("base", seed)])
    assert m < 1e-12, f"{arch}: autopar kl = {m:.3e}, pipeline no estable"
    print(f"[plomeria] autopar kl(base||base) = {m:.2e}  OK")


def recorre(arch: str, semillas: list[int], raiz: str,
            disp: str) -> pd.DataFrame:
    """calcula las tres familias de kl y la deriva de firma.

    Args:
        arch: etiqueta de arquitectura.
        semillas: semillas disponibles.
        raiz: directorio de checkpoints.
        disp: dispositivo.

    Returns:
        dataframe con una fila por comparación.
    """
    cfg = ARQUITECTURAS[arch]
    imgs = cargar_probe_tensor(cfg["probe"], img_size=cfg["img_size"],
                               crop_pct=cfg["crop_pct"],
                               interp=cfg["interp"])
    logits, firmas = {}, {}
    for var in VARIANTES:
        for s in tqdm(semillas, desc=f"{arch} {var}", leave=False):
            m = carga(arch, var, s, raiz, disp)
            if m is None:
                continue
            lg, fi = logits_y_firma(m, imgs, disp, cfg,
                                    con_firma=var in ("base", "blanda"))
            logits[(var, s)] = lg
            if fi is not None:
                firmas[(var, s)] = fi
            del m
            torch.cuda.empty_cache()
    presentes = sorted({s for (v, s) in logits if v == "base"})
    plomeria(logits, arch, presentes[0])

    filas = []
    for s in presentes:
        for var in ("blanda", "dura"):
            if (var, s) not in logits:
                continue
            med, mdn = kl_media(logits[("base", s)], logits[(var, s)])
            filas.append({"arch": arch, "familia": (
                "d_par" if var == "blanda" else "d_control"),
                "ref": s, "otra": s, "kl_media": med, "kl_mediana": mdn})
    for s in presentes:
        for t in presentes:
            if s == t:
                continue
            med, mdn = kl_media(logits[("base", s)], logits[("base", t)])
            filas.append({"arch": arch, "familia": "d_suelo", "ref": s,
                          "otra": t, "kl_media": med, "kl_mediana": mdn})
    # secundaria: deriva de la firma de respuesta, 1 - |cos|
    for s in presentes:
        if ("blanda", s) not in firmas:
            continue
        d = 1 - (firmas[("base", s)] * firmas[("blanda", s)]).sum(-1).abs()
        filas.append({"arch": arch, "familia": "firma_par", "ref": s,
                      "otra": s, "kl_media": float(d.mean()),
                      "kl_mediana": float(d.median())})
    for s in presentes:
        for t in presentes:
            if s == t:
                continue
            d = 1 - (firmas[("base", s)] * firmas[("base", t)]).sum(-1).abs()
            filas.append({"arch": arch, "familia": "firma_suelo", "ref": s,
                          "otra": t, "kl_media": float(d.mean()),
                          "kl_mediana": float(d.median())})
    return pd.DataFrame(filas)


def lee_veredicto(df: pd.DataFrame) -> None:
    """aplica los umbrales bloqueados, por arquitectura.

    Args:
        df: dataframe con todas las comparaciones.
    """
    for arch, sub in df.groupby("arch"):
        def val(f):
            return sub[sub["familia"] == f]["kl_media"].to_numpy()
        par, suelo, ctrl = val("d_par"), val("d_suelo"), val("d_control")
        if not len(par) or not len(suelo):
            continue
        techo = float(suelo.max())
        m_par, m_ctrl = float(np.median(par)), float(np.median(ctrl))
        pct = float((suelo < m_par).mean() * 100)
        frac = float((par <= techo).mean())
        if m_ctrl <= techo:
            ver = "SIN VEREDICTO (el control no separa: instrumento ciego)"
        elif m_par <= techo:
            ver = "V-C (inercia sostenida a nivel de salida)"
        else:
            ver = "V-D (la sonda blanda sí mueve la función)"
        print(f"\n=== {arch} ===")
        print(f"  d_par mediana   {m_par:.5f} nats  (n={len(par)})")
        print(f"  d_suelo         [{suelo.min():.5f}, {techo:.5f}] "
              f"(n={len(suelo)})")
        print(f"  d_control       {m_ctrl:.5f} nats  (n={len(ctrl)})")
        print(f"  VEREDICTO: {ver}")
        print(f"  descriptores: {100 * frac:.0f} % de semillas con d_par "
              f"<= techo; d_par cae en el percentil {pct:.0f} del suelo")
        fp, fs = val("firma_par"), val("firma_suelo")
        if len(fp) and len(fs):
            print(f"  secundaria (firma 1-|cos|): par {np.median(fp):.5f} "
                  f"vs suelo max {fs.max():.5f} -> "
                  f"{'dentro' if np.median(fp) <= fs.max() else 'fuera'}")


def main() -> None:
    """punto de entrada: medida, csv y veredicto."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", default="vitb,vitl")
    ap.add_argument("--semillas", default="42,43,44,45,46")
    ap.add_argument("--raiz", default="artifacts/checkpoints")
    ap.add_argument("--salida", default=SALIDA)
    args = ap.parse_args()

    disp = "cuda" if torch.cuda.is_available() else "cpu"
    semillas = [int(s) for s in args.semillas.split(",")]
    partes = [recorre(a, semillas, args.raiz, disp)
              for a in args.archs.split(",")]
    df = pd.concat([p for p in partes if len(p)], ignore_index=True)

    os.makedirs(os.path.dirname(args.salida), exist_ok=True)
    df.to_csv(args.salida, index=False)
    print(f"\n[guardado] {args.salida} ({len(df)} filas)")
    print("\n15 observaciones al azar del artefacto:")
    print(df.sample(min(15, len(df)), random_state=0).to_string(index=False))
    lee_veredicto(df)


if __name__ == "__main__":
    main()
