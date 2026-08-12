"""m1 — la vía de escape de la identificabilidad, medida.

bajo balance exacto de la factorización valor-salida (espectros
compartidos, mínimo de ||w_v||^2+||w_o||^2 sobre la órbita a circuito
fijo) se cumple v1(w_o) = v1(w_v w_o) exactamente: el proxy leería el
objeto invariante y la no-identificabilidad sería inocua en la
práctica. este script mide cuánto se aleja de eso el gauge que el
entrenamiento elige.

medida primaria, por cabeza: alfa = ángulo entre v1(w_o) y
v1(w_v w_o), ambos primeros vectores singulares derechos en r^d.
secundarias: el desbalance espectral separado en su componente de
escala ---irrelevante para v1, es la clase conforme ortogonal--- y su
componente de forma, más el hueco espectral sigma1/sigma2 de w_o, que
gobierna la sensibilidad de alfa.

umbrales pre-registrados (lock del 10-08-2026, no se tocan):
mediana alfa >= 60 grados -> v-a; <= 20 -> v-b; en medio, sin
veredicto fuerte. mediana delta_forma contra log(1,5). agregación por
arquitectura, nunca agrupando vitb con vitl.
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

from src.carga import ARQUITECTURAS, cargar_vit_base  # noqa: E402

# suelo aleatorio en r^768, calculado sobre 2000 pares (lock de m1)
SUELO_MEDIANA = 88.52
SUELO_P5 = 85.84
UMBRAL_VA = 60.0
UMBRAL_VB = 20.0
UMBRAL_FORMA = float(np.log(1.5))
GAP_ROBUSTEZ = 1.05


def v1_derecho(m: np.ndarray) -> np.ndarray:
    """primer vector singular derecho de una matriz.

    Args:
        m: matriz (p, q) en doble precisión.

    Returns:
        vector unitario de dimensión q.
    """
    return np.linalg.svd(m, full_matrices=False)[2][0]


def angulo(u: np.ndarray, v: np.ndarray) -> float:
    """ángulo en grados entre dos direcciones, salvo signo.

    el vector singular está definido salvo signo, así que se toma el
    valor absoluto del producto escalar.

    Args:
        u: primera dirección unitaria.
        v: segunda dirección unitaria.

    Returns:
        ángulo en grados, en [0, 90].
    """
    return float(np.degrees(np.arccos(min(1.0, abs(float(u @ v))))))


def v1_circuito_ov(w_v: np.ndarray, w_o: np.ndarray) -> np.ndarray:
    """primer vector singular derecho del circuito ov, sin formarlo.

    con w_v = p r (qr reducida) se tiene w_v w_o = p (r w_o), y como p
    tiene columnas ortonormales los vectores singulares derechos del
    circuito coinciden con los de r w_o, que es (d_h, d) en vez de
    (d, d). evita una svd de 768x768 por cabeza.

    Args:
        w_v: proyección de valor (d, d_h).
        w_o: proyección de salida (d_h, d).

    Returns:
        primer vector singular derecho del circuito, dimensión d.
    """
    r = np.linalg.qr(w_v, mode="reduced")[1]
    return v1_derecho(r @ w_o)


def sector_vo(
    modelo, capa: int, cabeza: int, dim_cabeza: int
) -> tuple[np.ndarray, np.ndarray]:
    """extrae w_v y w_o de una cabeza en la convención del paper.

    timm guarda qkv como [3d, d] y proj como [d, d]; el bloque de
    valor empieza en la fila 2d y el de salida en la columna h*d_h.

    Args:
        modelo: vit interno de timm, el que expone .blocks.
        capa: índice de bloque.
        cabeza: índice de cabeza dentro del bloque.
        dim_cabeza: dimensión por cabeza.

    Returns:
        tupla (w_v de forma (d, d_h), w_o de forma (d_h, d)), fp64.
    """
    attn = modelo.blocks[capa].attn
    w_qkv = attn.qkv.weight.detach()
    w_proj = attn.proj.weight.detach()
    base = 2 * w_qkv.shape[1]
    fil = slice(base + cabeza * dim_cabeza, base + (cabeza + 1) * dim_cabeza)
    col = slice(cabeza * dim_cabeza, (cabeza + 1) * dim_cabeza)
    w_v = w_qkv[fil, :].double().t().numpy()
    w_o = w_proj[:, col].double().t().numpy()
    return w_v, w_o


def metricas_cabeza(w_v: np.ndarray, w_o: np.ndarray) -> dict:
    """calcula alfa y el desbalance espectral de una cabeza.

    Args:
        w_v: proyección de valor (d, d_h).
        w_o: proyección de salida (d_h, d).

    Returns:
        diccionario con alfa_deg, delta_escala, delta_forma, gap_wo y
        los primeros valores singulares de ambas matrices.
    """
    sv = np.linalg.svd(w_v, compute_uv=False)
    so = np.linalg.svd(w_o, compute_uv=False)
    lr = np.log(sv / so)
    return {
        "alfa_deg": angulo(v1_derecho(w_o), v1_circuito_ov(w_v, w_o)),
        "delta_escala": float(abs(lr.mean())),
        "delta_forma": float(np.mean(np.abs(lr - lr.mean()))),
        "gap_wo": float(so[0] / so[1]),
        "sigma1_wv": float(sv[0]),
        "sigma1_wo": float(so[0]),
    }


def comprueba_carga(modelo, ckpt: str, dim_cabeza: int) -> None:
    """verifica que los pesos medidos son los del checkpoint.

    `cargar_vit_base` usa strict=False y descarta claves con forma
    incompatible, así que una carga parcial silenciosa produciría
    números de un modelo sin afinar. se compara un tensor concreto
    contra el del fichero.

    Args:
        modelo: vit ya cargado.
        ckpt: ruta del checkpoint.
        dim_cabeza: dimensión por cabeza, solo para el mensaje.

    Raises:
        AssertionError: si el tensor del modelo no coincide.
    """
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = blob.get("model", blob)
    # el checkpoint prefija 'backbone.model.'; se busca por sufijo para
    # no depender de cuántos niveles envuelven a la columna
    sufijo = "blocks.0.attn.qkv.weight"
    claves = [k for k in sd if k.endswith(sufijo)]
    assert len(claves) == 1, f"{ckpt}: {len(claves)} claves con {sufijo}"
    ref = sd[claves[0]].cpu()
    viva = modelo.blocks[0].attn.qkv.weight.detach().cpu()
    assert torch.equal(ref, viva), (
        f"{ckpt}: los pesos cargados no son los del fichero "
        f"(carga parcial silenciosa)"
    )


def autotest() -> None:
    """plomería obligatoria: el instrumento antes que la lectura.

    construye una factorización balanceada sintética y comprueba que
    da alfa ~ 0 y delta_forma ~ 0; luego le aplica un gauge escalar y
    comprueba que alfa sigue ~ 0 mientras delta_escala crece a
    2·log c. sin estos dos asserts, cualquier veredicto podría venir
    de un instrumento roto.

    Raises:
        AssertionError: si alguna de las dos comprobaciones falla.
    """
    rng = np.random.default_rng(0)
    d, d_h, c = 768, 64, 5.0
    a = np.linalg.qr(rng.standard_normal((d, d_h)))[0]
    b = np.linalg.qr(rng.standard_normal((d, d_h)))[0]
    sig = np.sort(rng.uniform(0.3, 3.0, d_h))[::-1]
    q = np.linalg.qr(rng.standard_normal((d_h, d_h)))[0]
    w_v = a @ np.diag(np.sqrt(sig)) @ q
    w_o = q.T @ np.diag(np.sqrt(sig)) @ b.T

    m = metricas_cabeza(w_v, w_o)
    assert m["alfa_deg"] < 1e-6, f"balance: alfa={m['alfa_deg']}"
    assert m["delta_forma"] < 1e-9, f"balance: forma={m['delta_forma']}"
    print(f"[plomeria] balance exacto -> alfa={m['alfa_deg']:.2e} deg, "
          f"delta_forma={m['delta_forma']:.2e}  OK")

    g = metricas_cabeza(w_v * c, w_o / c)
    assert g["alfa_deg"] < 1e-6, f"gauge escalar: alfa={g['alfa_deg']}"
    assert abs(g["delta_escala"] - 2 * np.log(c)) < 1e-9, "escala mal"
    print(f"[plomeria] gauge escalar c={c} -> alfa={g['alfa_deg']:.2e} deg "
          f"(invariante), delta_escala={g['delta_escala']:.4f} "
          f"(=2·log c={2*np.log(c):.4f})  OK")


def recorre(arch: str, semillas: list[int], raiz: str) -> pd.DataFrame:
    """mide todas las cabezas de todas las semillas de una columna.

    Args:
        arch: etiqueta de arquitectura (vitb o vitl).
        semillas: semillas de la variante base a recorrer.
        raiz: directorio de checkpoints.

    Returns:
        dataframe con una fila por cabeza.
    """
    cfg = ARQUITECTURAS[arch]
    filas = []
    for s in semillas:
        ckpt = f"{raiz}/{arch}_clean/attnA_base_seed{s}_last.pt"
        if not os.path.exists(ckpt):
            print(f"[aviso] no está {ckpt}; se salta")
            continue
        modelo = cargar_vit_base(
            ckpt, device="cpu", num_classes=100,
            model_name=cfg["model_name"], img_size=cfg["img_size"],
        )
        comprueba_carga(modelo, ckpt, cfg["dim_cabeza"])
        n_capas = len(modelo.blocks)
        total = n_capas * cfg["n_cabezas"]
        barra = tqdm(total=total, desc=f"{arch} seed{s}", leave=False)
        for capa in range(n_capas):
            for cabeza in range(cfg["n_cabezas"]):
                w_v, w_o = sector_vo(modelo, capa, cabeza,
                                     cfg["dim_cabeza"])
                fila = {"arch": arch, "seed": s, "capa": capa,
                        "cabeza": cabeza}
                fila.update(metricas_cabeza(w_v, w_o))
                filas.append(fila)
                barra.update(1)
        barra.close()
        med = np.median([f["alfa_deg"] for f in filas if f["seed"] == s])
        print(f"[{arch} seed{s}] {total} cabezas, mediana alfa = "
              f"{med:.2f} deg", flush=True)
        del modelo
    return pd.DataFrame(filas)


def lee_veredicto(df: pd.DataFrame) -> None:
    """aplica los umbrales bloqueados, sin margen de improvisación.

    Args:
        df: dataframe con todas las cabezas medidas.
    """
    for arch, sub in df.groupby("arch"):
        por_semilla = sub.groupby("seed")["alfa_deg"].median()
        med = float(por_semilla.median())
        robusta = sub[sub["gap_wo"] >= GAP_ROBUSTEZ]
        med_rob = (float(robusta.groupby("seed")["alfa_deg"].median()
                         .median()) if len(robusta) else float("nan"))
        forma = float(sub.groupby("seed")["delta_forma"].median().median())
        escala = float(sub.groupby("seed")["delta_escala"].median().median())
        if med >= UMBRAL_VA:
            ver = "V-A (la vía de escape está vacía)"
        elif med <= UMBRAL_VB:
            ver = "V-B (el proxy lee el invariante)"
        else:
            ver = "zona intermedia, sin veredicto fuerte"
        if med >= SUELO_P5:
            desc = "indistinguible de una dirección arbitraria"
        elif med >= UMBRAL_VA:
            desc = "lejos del invariante, con alineamiento residual"
        else:
            desc = "por debajo del umbral de V-A"
        print(f"\n=== {arch} (n={sub['seed'].nunique()} semillas, "
              f"{len(sub)} cabezas) ===")
        print(f"  mediana alfa por semilla: "
              f"{por_semilla.mean():.2f} +/- {por_semilla.std(ddof=1):.2f} "
              f"deg   (mediana de medianas {med:.2f})")
        print(f"  mediana alfa con gap >= {GAP_ROBUSTEZ}: {med_rob:.2f} deg "
              f"({len(robusta)} de {len(sub)} cabezas)")
        print(f"  delta_forma {forma:.4f} (umbral {UMBRAL_FORMA:.4f})   "
              f"delta_escala {escala:.4f} (irrelevante para v1)")
        print(f"  VEREDICTO: {ver}")
        print(f"  descriptor frente al suelo (p5 {SUELO_P5} deg): {desc}")


def control_antes_del_afinado(archs: list[str], salida: str) -> None:
    """diagnóstico: alfa en init aleatoria y en el preentrenado.

    no forma parte del lock ni mueve ningún veredicto; responde a la
    pregunta que la lectura de alfa deja abierta ---si el gauge casi
    canónico lo trae el afinado o ya venía de antes--- y separa alfa de
    delta_forma como medidas de balance.

    Args:
        archs: etiquetas de arquitectura a recorrer.
        salida: ruta del csv de control.
    """
    import timm

    filas = []
    for arch in archs:
        cfg = ARQUITECTURAS[arch]
        for etiqueta, pre in [("init", False), ("preentrenado", True)]:
            try:
                m = timm.create_model(cfg["model_name"], pretrained=pre,
                                      num_classes=100).eval()
            except Exception as exc:
                print(f"[control] {arch} {etiqueta}: no disponible "
                      f"({type(exc).__name__})")
                continue
            for capa in tqdm(range(len(m.blocks)),
                             desc=f"control {arch} {etiqueta}", leave=False):
                for cabeza in range(cfg["n_cabezas"]):
                    w_v, w_o = sector_vo(m, capa, cabeza, cfg["dim_cabeza"])
                    fila = {"arch": arch, "estado": etiqueta, "capa": capa,
                            "cabeza": cabeza}
                    fila.update(metricas_cabeza(w_v, w_o))
                    filas.append(fila)
            del m
    df = pd.DataFrame(filas)
    os.makedirs(os.path.dirname(salida), exist_ok=True)
    df.to_csv(salida, index=False)
    print(f"\n[guardado] {salida} ({len(df)} filas)")
    print("\n15 observaciones al azar del artefacto:")
    print(df.sample(min(15, len(df)), random_state=0).to_string(index=False))
    print("\nmedianas del control:")
    print(df.groupby(["arch", "estado"])[["alfa_deg", "delta_forma"]]
          .median().to_string())


def main() -> None:
    """punto de entrada: plomería, medida, csv y veredicto."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", default="vitb,vitl")
    ap.add_argument("--semillas", default="42,43,44,45,46")
    ap.add_argument("--raiz", default="artifacts/checkpoints")
    ap.add_argument("--salida",
                    default="artifacts/logs/balance/balance_espectral.csv")
    ap.add_argument("--solo-plomeria", action="store_true")
    ap.add_argument("--control", action="store_true",
                    help="diagnostico alfa antes del afinado")
    ap.add_argument(
        "--salida-control",
        default="artifacts/logs/balance/control_antes_afinado.csv")
    args = ap.parse_args()

    autotest()
    if args.solo_plomeria:
        return
    if args.control:
        control_antes_del_afinado(args.archs.split(","),
                                  args.salida_control)
        return

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
