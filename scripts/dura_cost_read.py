"""lectura del coste de la variante dura, homogénea con la inercia.

extrae val_top1 (y θ_min, s_func) con el MISMO criterio que s_func en
inertia_read: promedio de las últimas cinco épocas convergidas (ce>0),
las mismas filas, para que base y dura se comparen sobre la misma
ventana y el band sea homogéneo. reporta además la dispersión
intra-corrida de val_top1 por corrida, para distinguir "la dura cuesta X
en media" de "la dura cuesta X y además desestabiliza accuracy" (la
reproyección tras cada step es una perturbación que base no sufre). el
coste se lee contra el band de base de cada métrica, no contra cero ni
contra la dirección esperada: estándar simétrico con la blanda.
"""

import argparse

import numpy as np
import pandas as pd

# columnas de interés en run.csv (mismas filas, ventana convergida).
# theta_min_media es la PRIMARIA: el resto del paper (premisa, inercia)
# reporta la separación angular como la MEDIA (~52 base, ~85 sonda), no
# el mínimo. "θ_min" debe significar lo mismo en las tres secciones o el
# argumento se fractura. theta_min_minimo se reporta como dato secundario
# etiquetado, nunca como "el θ_min" sin más.
METRICAS = {
    "val_top1": "val_top1",
    "theta_min_media": "theta_min_mean_deg",
    "theta_min_minimo": "theta_min_min_deg",
    "s_func": "val_s_func_mean",
}


def ventana_convergida(ruta: str, ultimas: int = 5) -> pd.DataFrame:
    """devuelve las últimas épocas con ce>0 (misma ventana que s_func).

    Args:
        ruta: ruta al run.csv.
        ultimas: número de épocas finales convergidas.

    Returns:
        sub-DataFrame con esas filas.
    """
    df = pd.read_csv(ruta)
    return df[df["ce"] > 0].tail(ultimas)


def resumen(ruta: str) -> dict[str, tuple[float, float]]:
    """media y desviación intra-corrida de cada métrica en la ventana.

    Args:
        ruta: ruta al run.csv.

    Returns:
        dict métrica -> (media, std intra-corrida) sobre la ventana.
    """
    w = ventana_convergida(ruta)
    out = {}
    for k, col in METRICAS.items():
        out[k] = (float(w[col].mean()), float(w[col].std(ddof=1)))
    return out


def main() -> None:
    """computa el coste de la dura contra el band de base, simétrico."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="vitb")
    ap.add_argument("--variants", nargs="+", default=["base", "dura"])
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 43, 44, 45, 46])
    args = ap.parse_args()
    raiz = f"artifacts/logs/{args.arch}_clean"

    # media intra-corrida y dispersión intra-corrida por variante/semilla
    datos: dict[str, dict[int, dict]] = {}
    for v in args.variants:
        datos[v] = {}
        for s in args.seeds:
            ruta = f"{raiz}/attnA_{v}_seed{s}/run.csv"
            try:
                datos[v][s] = resumen(ruta)
            except FileNotFoundError:
                pass

    for m in METRICAS:
        print(f"\n=== {m} (prom. últimas 5 ce>0, misma ventana) ===")
        bandas = {}
        for v in args.variants:
            ss = sorted(datos[v])
            if not ss:
                print(f"  {v}: (sin corridas aún)")
                continue
            medias = np.array([datos[v][s][m][0] for s in ss])
            intra = np.array([datos[v][s][m][1] for s in ss])
            band = float(medias.std(ddof=1)) if len(medias) > 1 else 0.0
            bandas[v] = (float(medias.mean()), band)
            print(f"  {v} (n={len(ss)}): media={medias.mean():.4f}  "
                  f"band_entre_semillas={band:.4f}  "
                  f"std_intra_media={intra.mean():.4f}")
            detalle = "  ".join(
                f"s{s}={datos[v][s][m][0]:.4f}(±{datos[v][s][m][1]:.4f})"
                for s in ss)
            print(f"    {detalle}")

        # coste contra el band de base, si ambas están
        if "base" in bandas and "dura" in bandas:
            mb, bb = bandas["base"]
            md, _ = bandas["dura"]
            efecto = md - mb
            print(f"  -- efecto dura-base (vs-media) = {efecto:+.4f}  "
                  f"band_base={bb:.4f}  "
                  f"|efecto|/band={abs(efecto)/bb if bb else 0:.2f}")
            if bb:
                dentro = abs(efecto) <= bb
                borde = bb < abs(efecto) <= 2 * bb
                if dentro:
                    msg = ("DENTRO del ruido (cota: coste del orden del "
                           "ruido entre semillas)")
                elif borde:
                    msg = ("BORDE (1-2 band: coste comparable al ruido, "
                           "leer como borde no veredicto)")
                else:
                    msg = "FUERA del ruido (coste real, reportar magnitud)"
                print(f"     {msg}")

            # pareo por semilla: dura_s contra base_s homóloga, para ver
            # si el coste es uniforme (sistemático de la variante) o lo
            # domina una semilla (corrida atípica como base44/45). doble
            # pareo con el vs-media, como en la inercia.
            comunes = sorted(set(datos["base"]) & set(datos["dura"]))
            par = np.array([datos["dura"][s][m][0]
                            - datos["base"][s][m][0] for s in comunes])
            if len(par):
                det = "  ".join(f"s{s}={d:+.4f}"
                                for s, d in zip(comunes, par))
                print(f"  -- pareado (dura_s-base_s): {det}")
                rango = par.max() - par.min()
                unif = (rango <= abs(par.mean())) if par.mean() else False
                tag = ("uniforme (coste sistemático)" if unif else
                       "disperso (¿dominado por una semilla?)")
                print(f"     media={par.mean():+.4f}  "
                      f"rango={rango:.4f}  {tag}")

            # desestabilización: ¿la dura ondula más intra-corrida?
            if m == "val_top1":
                ib = np.mean([datos["base"][s][m][1]
                              for s in sorted(datos["base"])])
                idu = np.mean([datos["dura"][s][m][1]
                               for s in sorted(datos["dura"])])
                rel = idu / ib if ib else float("nan")
                if rel > 1.5:
                    z = "dura desestabiliza"
                elif rel >= 1.3:
                    z = "BORDE (cerca de 1,5, leer como borde no veredicto)"
                else:
                    z = "comparable"
                print(f"  -- estabilidad intra-corrida val_top1: "
                      f"base={ib:.4f}  dura={idu:.4f}  ratio={rel:.2f}"
                      f"  ({z})")


if __name__ == "__main__":
    main()
