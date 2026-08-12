"""m1-bis — coincidencia de decisión en el gauge natural.

m1 midió que el gauge elegido por el entrenamiento acerca
v1(w_o) al objeto invariante (18-35 grados). esto pregunta lo
siguiente: ¿basta ese acercamiento para que la decisión de poda
coincida? sin inyectar ningún gauge, se materializan las dos
decisiones típicas con dos criterios sobre el mismo checkpoint:

  decision a (proxy):      sobre v1(w_o^(h))
  decision b (invariante): sobre v1(w_v^(h) w_o^(h)), estático

y se mide si coinciden. mide **acuerdo entre criterios**, no calidad
de poda: que ninguna similitud bata al azar como criterio de poda ya
está establecido aparte, y las dos preguntas no se mezclan.

umbrales heredados de la tabla de decisión, fijados antes de que
existiera esta medida: inestable si el par cambia en >= 40 % de las
capas o el solape medio del top-k baja de 0,67.
"""

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.nucleo_lectura import decisiones  # noqa: E402
from scripts.run_fase_G import firma_exacta  # noqa: E402
from src.carga import ARQUITECTURAS, cargar_vit_base  # noqa: E402
from src.firma_funcional import (w_o_por_cabeza,  # noqa: E402
                                 w_v_columnas)
from src.gauge_flip import (aplica_gauge_ortogonal,  # noqa: E402
                            aplica_gauge_ov)

K_PODA = {"vitb": 3, "vitl": 4}          # 25 % relativo, como el paper
UMBRAL_SOLAPE = 0.67                     # heredado de la tabla 3
UMBRAL_PAR_CAMBIA = 0.40                 # heredado de la tabla 3
SALIDA = "artifacts/logs/balance/decision_natural.csv"


def dir_invariante(w_v: torch.Tensor, w_o: torch.Tensor) -> torch.Tensor:
    """v1 del circuito ov por cabeza, sin materializar (d, d).

    con w_v = p r (qr reducida), w_v w_o = p (r w_o) y p tiene
    columnas ortonormales, así que los vectores singulares derechos
    del circuito son los de r w_o. la svd la hace `firma_exacta`, el
    mismo camino que usa la decisión por pesos.

    Args:
        w_v: tensor [h, d, dh].
        w_o: tensor [h, dh, d].

    Returns:
        tensor [h, d] con la dirección invariante por cabeza.
    """
    r = torch.linalg.qr(w_v.double(), mode="reduced")[1]
    return firma_exacta(r @ w_o.double(), "der")


def dir_proxy(w_o: torch.Tensor) -> torch.Tensor:
    """v1(w_o) por cabeza, el objeto que la práctica leería.

    Args:
        w_o: tensor [h, dh, d].

    Returns:
        tensor [h, d] con la dirección dominante por cabeza.
    """
    return firma_exacta(w_o.double(), "der")


def angulos(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """ángulo en grados entre direcciones pareadas, salvo signo.

    Args:
        a: tensor [h, d].
        b: tensor [h, d].

    Returns:
        array de h ángulos en grados.
    """
    cos = (a * b).sum(dim=1).abs().clamp(max=1.0)
    return np.degrees(torch.arccos(cos).cpu().numpy())


def decide(dirs: torch.Tensor, k: int) -> tuple[tuple[int, int], list[int]]:
    """par más redundante y top-k, por el camino del núcleo.

    Args:
        dirs: tensor [h, d] de direcciones unitarias.
        k: tamaño del conjunto de poda.

    Returns:
        tupla (par ordenado, lista de k cabezas).
    """
    return decisiones(dirs.float(), k)


def plomeria(modelo, arch: str, capa: int = 0) -> None:
    """las tres comprobaciones exigidas antes de leer nada.

    (1) determinismo de la decisión invariante; (2) un gauge genérico
    mueve el proxy y no el invariante; (3) un gauge ortogonal no mueve
    ninguno de los dos ---c1 operando como assert---. además se
    comprueba que la ruta qr del circuito coincide con la svd del
    circuito materializado.

    Args:
        modelo: el vit cargado.
        arch: etiqueta de arquitectura.
        capa: capa sobre la que se prueba.

    Raises:
        AssertionError: si alguna comprobación falla.
    """
    cfg = ARQUITECTURAS[arch]
    h, dh, k = cfg["n_cabezas"], cfg["dim_cabeza"], K_PODA[arch]
    w_o = w_o_por_cabeza(modelo, capa, h, dh)
    w_v = w_v_columnas(modelo, capa, h, dh)
    a0, b0 = dir_proxy(w_o), dir_invariante(w_v, w_o)

    # ruta qr contra circuito materializado, una cabeza
    pleno = firma_exacta((w_v[:1].double() @ w_o[:1].double()), "der")
    assert float((pleno[0] @ b0[0]).abs()) > 1 - 1e-9, "ruta qr != svd plena"

    b0b = dir_invariante(w_v_columnas(modelo, capa, h, dh),
                         w_o_por_cabeza(modelo, capa, h, dh))
    assert decide(b0, k) == decide(b0b, k), "decisión b no determinista"
    print("[plomeria] (1) determinismo de la decisión invariante  OK")

    m_gen = copy.deepcopy(modelo)
    aplica_gauge_ov(m_gen, capa, h, dh, semilla=1000 * capa, escala_id=8.0)
    ag = dir_proxy(w_o_por_cabeza(m_gen, capa, h, dh))
    bg = dir_invariante(w_v_columnas(m_gen, capa, h, dh),
                        w_o_por_cabeza(m_gen, capa, h, dh))
    d_a, d_b = angulos(a0, ag).mean(), angulos(b0, bg).mean()
    assert d_a > 5.0, f"gauge genérico no movió el proxy ({d_a:.2f} deg)"
    assert d_b < 1e-3, f"gauge genérico movió el invariante ({d_b:.2e})"
    print(f"[plomeria] (2) gauge genérico: proxy {d_a:.1f} deg, "
          f"invariante {d_b:.1e} deg  OK")

    m_ort = copy.deepcopy(modelo)
    aplica_gauge_ortogonal(m_ort, capa, h, dh, semilla=7 + capa)
    ao = dir_proxy(w_o_por_cabeza(m_ort, capa, h, dh))
    bo = dir_invariante(w_v_columnas(m_ort, capa, h, dh),
                        w_o_por_cabeza(m_ort, capa, h, dh))
    d_ao, d_bo = angulos(a0, ao).mean(), angulos(b0, bo).mean()
    assert d_ao < 1e-2, f"c1 falla: el ortogonal movió el proxy {d_ao:.2e}"
    assert d_bo < 1e-2, f"el ortogonal movió el invariante {d_bo:.2e}"
    print(f"[plomeria] (3) gauge ortogonal: proxy {d_ao:.1e} deg, "
          f"invariante {d_bo:.1e} deg ---C1 como assert---  OK")
    del m_gen, m_ort


def recorre(arch: str, semillas: list[int], raiz: str) -> pd.DataFrame:
    """mide la coincidencia de decisión en todas las capas y semillas.

    Args:
        arch: etiqueta de arquitectura.
        semillas: semillas de la variante base.
        raiz: directorio de checkpoints.

    Returns:
        dataframe con una fila por capa y semilla.
    """
    cfg = ARQUITECTURAS[arch]
    h, dh, k = cfg["n_cabezas"], cfg["dim_cabeza"], K_PODA[arch]
    filas, primera = [], True
    for s in semillas:
        ckpt = f"{raiz}/{arch}_clean/attnA_base_seed{s}_last.pt"
        if not os.path.exists(ckpt):
            print(f"[aviso] no está {ckpt}; se salta")
            continue
        modelo = cargar_vit_base(
            ckpt, device="cpu", num_classes=100,
            model_name=cfg["model_name"], img_size=cfg["img_size"])
        if primera:
            plomeria(modelo, arch)
            primera = False
        for capa in tqdm(range(len(modelo.blocks)),
                         desc=f"{arch} seed{s}", leave=False):
            w_o = w_o_por_cabeza(modelo, capa, h, dh)
            w_v = w_v_columnas(modelo, capa, h, dh)
            a, b = dir_proxy(w_o), dir_invariante(w_v, w_o)
            par_a, top_a = decide(a, k)
            par_b, top_b = decide(b, k)
            filas.append({
                "arch": arch, "seed": s, "capa": capa,
                "par_proxy": str(par_a), "par_invariante": str(par_b),
                "coincide_par": par_a == par_b,
                "solape_topk": len(set(top_a) & set(top_b)) / k,
                "alfa_medio_capa": float(angulos(a, b).mean()),
            })
        del modelo
    return pd.DataFrame(filas)


def lee_veredicto(df: pd.DataFrame) -> None:
    """aplica los umbrales heredados, por arquitectura.

    Args:
        df: dataframe con una fila por capa y semilla.
    """
    for arch, sub in df.groupby("arch"):
        k = K_PODA[arch]
        h = ARQUITECTURAS[arch]["n_cabezas"]
        solape = float(sub["solape_topk"].mean())
        coincide = float(sub["coincide_par"].mean())
        suelo_par = 2.0 / (h * (h - 1))
        r = float(np.corrcoef(sub["alfa_medio_capa"],
                              sub["solape_topk"])[0, 1])
        v_e = solape >= UMBRAL_SOLAPE and coincide >= 0.60
        print(f"\n=== {arch} (k={k} de {h}, {len(sub)} capas x semillas) ===")
        print(f"  solape top-k medio: {solape:.3f}   "
              f"(umbral {UMBRAL_SOLAPE}, suelo azar {k / h:.2f})")
        print(f"  el par coincide en: {100 * coincide:.1f} % de capas   "
              f"(suelo azar {100 * suelo_par:.1f} %)")
        print(f"  alfa medio por capa: {sub['alfa_medio_capa'].mean():.2f} "
              f"deg")
        etiqueta = ("V-E (la canonización alcanza a la decisión)"
                    if v_e else "V-F (no alcanza)")
        print(f"  VEREDICTO: {etiqueta}")
        print(f"  descriptor r(alfa, solape) por capa: {r:+.3f}")


def main() -> None:
    """punto de entrada: medida, csv y veredicto."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", default="vitb,vitl")
    ap.add_argument("--semillas", default="42,43,44,45,46")
    ap.add_argument("--raiz", default="artifacts/checkpoints")
    ap.add_argument("--salida", default=SALIDA)
    args = ap.parse_args()

    semillas = [int(s) for s in args.semillas.split(",")]
    partes = [recorre(a, semillas, args.raiz)
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
