"""lectura en frío del cuarto apoyo (inercia de s_func) a n=5.

lee las cinco base y las cinco blandas, promedia val_s_func_mean sobre
las últimas cinco épocas convergidas (ce>0), y reporta la cota: shift
emparejado y vs-media, band completo y core, separación de cero dado el
spread. no consolida "plana"; reporta el régimen donde efecto, ruido
entre semillas y ruido de corrida son del mismo tamaño. la lectura es
sin orden privilegiado: trata las cinco como cinco puntos.
"""

import argparse

import numpy as np
import pandas as pd


def s_func_convergido(ruta: str, ultimas: int = 5) -> float:
    """promedia val_s_func_mean sobre las últimas épocas con ce>0.

    s_func deriva hasta ~ep25 y se estabiliza; promediar las últimas
    convergidas quita el ruido de época única.

    Args:
        ruta: ruta al run.csv de la corrida.
        ultimas: número de épocas finales convergidas a promediar.

    Returns:
        media de val_s_func_mean sobre esas épocas.
    """
    df = pd.read_csv(ruta)
    conv = df[df["ce"] > 0].tail(ultimas)
    return float(conv["val_s_func_mean"].mean())


def main() -> None:
    """computa y reporta la inercia a n=5 en frío."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="vitb")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 43, 44, 45, 46])
    args = ap.parse_args()

    raiz = f"artifacts/logs/{args.arch}_clean"
    base = {}
    blanda = {}
    for s in args.seeds:
        base[s] = s_func_convergido(
            f"{raiz}/attnA_base_seed{s}/run.csv")
        blanda[s] = s_func_convergido(
            f"{raiz}/attnA_blanda_seed{s}/run.csv")

    bvals = np.array([base[s] for s in args.seeds])
    blvals = np.array([blanda[s] for s in args.seeds])
    band = float(bvals.std(ddof=1))
    media_base = float(bvals.mean())
    media_bl = float(blvals.mean())

    # core: band sin los dos outliers de corrida (más alto y más bajo)
    orden = np.argsort(bvals)
    nucleo = bvals[orden[1:-1]]
    core = float(nucleo.std(ddof=1))

    d_par = blvals - bvals
    d_vsm = blvals - media_base

    print("=== inercia n=5 en frío (s_func, prom. últimas 5 ce>0) ===")
    print(f"arch={args.arch}  semillas={args.seeds}")
    print("\n-- las cinco como cinco puntos (sin orden) --")
    for s in args.seeds:
        print(f"  seed{s}: base={base[s]:.4f}  blanda={blanda[s]:.4f}"
              f"  Δpar={blanda[s]-base[s]:+.4f}"
              f"  Δvsm={blanda[s]-media_base:+.4f}")

    print("\n-- distribuciones --")
    print(f"  base   media={media_base:.4f}  band(std n=5)={band:.4f}"
          f"  rango=[{bvals.min():.4f},{bvals.max():.4f}]")
    print(f"  blanda media={media_bl:.4f}  std={blvals.std(ddof=1):.4f}"
          f"  rango=[{blvals.min():.4f},{blvals.max():.4f}]")
    print(f"  core(band sin 2 outliers de corrida)={core:.4f}")
    print(f"  shift distribuciones (media_bl-media_base)="
          f"{media_bl-media_base:+.4f}")

    print("\n-- shift emparejado vs vs-media --")
    sp = "".join("+" if x > 0 else "-" for x in d_par)
    sv = "".join("+" if x > 0 else "-" for x in d_vsm)
    print(f"  Δpar  media={d_par.mean():+.4f}  std={d_par.std(ddof=1):.4f}"
          f"  signos={sp}")
    print(f"  Δvsm  media={d_vsm.mean():+.4f}  std={d_vsm.std(ddof=1):.4f}"
          f"  signos={sv}")

    print("\n-- separación de cero dado el spread (vs-media) --")
    sem = float(d_vsm.std(ddof=1) / np.sqrt(len(d_vsm)))
    print(f"  media Δvsm={d_vsm.mean():+.4f}  ±1.96·sem="
          f"[{d_vsm.mean()-1.96*sem:+.4f},{d_vsm.mean()+1.96*sem:+.4f}]")
    incluye = (d_vsm.mean() - 1.96 * sem) <= 0 <= (d_vsm.mean()
                                                   + 1.96 * sem)
    print(f"  ¿el intervalo incluye cero? {incluye}")
    print(f"  positivos vs-media: {(d_vsm > 0).sum()}/{len(d_vsm)}")

    print("\n-- straddle del band --")
    eff = abs(d_vsm.mean())
    print(f"  |efecto|={eff:.4f}  band_completo={band:.4f}  core={core:.4f}")
    print(f"  efecto dentro del band: {eff <= band}; "
          f"fuera del core: {eff > core}")
    print(f"  |band-core|={abs(band-core):.4f} (¿del orden del efecto? "
          f"{abs(band-core) <= 2*eff and abs(band-core) >= 0.3*eff})")


if __name__ == "__main__":
    main()
