"""comprueba que las cifras bloqueadas están en los tres PDF.

el episodio que lo justifica: el solape de la columna de lenguaje
vivió mal ---0,229 en vez de 0,379--- desde v5.x sin que nada lo
tocara, porque ninguna comprobación miraba las celdas de
`tab:decision`. una cifra que nadie vigila es una cifra que puede
derivar en silencio a través de tres superficies a la vez.

el oráculo hace dos cosas distintas:

1. **presencia**: cada cifra ratificada aparece en el PDF que le
   corresponde, con el separador decimal de su idioma. la coma del
   español y el punto del inglés se derivan de la misma entrada, así
   que declarar una cifra la vigila en las tres superficies.
2. **consistencia cruzada**: el port ciego no puede discrepar del
   derivado inglés en ninguna cifra vigilada; si discrepa, la cadena
   de propagación se rompió por el medio.

nota de procedencia: la versión de este oráculo que corrió el
12-08-2026 ---veinte comprobaciones, luego veinticuatro--- fue una
verificación suelta que no se guardó. esto es su reconstrucción como
script, con la lista declarada y a la vista en vez de en la memoria
de una sesión.

uso:
    python scripts/oraculo_cifras.py
"""

import argparse
import re
import subprocess
import sys

ES = "paper/separacion_angular_paper.pdf"
EN = "paper/angular_separation_paper_en.pdf"
PORT = "paper_tmlr/main.pdf"

# cifra en su forma española -> (qué es, ¿viaja al port ciego?).
# la forma inglesa se deriva cambiando la coma por punto
CIFRAS = {
    # tab:decision, las cuatro celdas que el bug de 2026-08 dejó al aire
    "92,7": ("decisión rota por pesos, visión", True),
    "0,378": ("solape de poda, visión", True),
    "90,0": ("decisión rota por pesos, lenguaje", True),
    "0,379": ("solape de poda, lenguaje", True),
    # tab:gauge / fase g
    "4,05": ("desv_R de la fuerza máxima", True),
    "0,66": ("deriva a media fuerza", True),
    "0,83": ("deriva de v1(W_O) en régimen saturado", True),
    # órbita certificada (app:orbita)
    "3960": ("pares con theta* alcanzado", True),
    "14,7": ("suelo angular medido", True),
    # intervalo del confirmatorio
    "0,0052": ("extremo inferior del IC pareado", True),
    "0,0077": ("extremo superior del IC pareado", True),
    # trío multi-arquitectura
    "24/24": ("capas con residuo en dinov2", True),
    "+0,052": ("s_func de la variante dura", True),
    "4,40": ("coste uniforme en puntos", True),
    # gauge natural (M1) y balance
    "64,06": ("alineación inicial de dinov2 ViT-L/14", True),
}


def texto(pdf: str) -> str:
    """extrae el texto de un pdf.

    Args:
        pdf: ruta del fichero.

    Returns:
        el texto plano, con los guiones de corte de línea unidos.

    Raises:
        SystemExit: si pdftotext falla o el pdf no existe.
    """
    r = subprocess.run(["pdftotext", "-q", pdf, "-"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"[error] no se pudo leer {pdf}")
    return re.sub(r"[­-]\s*\n\s*", "", r.stdout)


def main() -> None:
    """corre las dos comprobaciones y falla si alguna cifra no está."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--es", default=ES)
    ap.add_argument("--en", default=EN)
    ap.add_argument("--port", default=PORT)
    args = ap.parse_args()
    t_es, t_en, t_pt = (texto(args.es), texto(args.en),
                        texto(args.port))

    fallos = []
    print(f"{'cifra':>8s}  {'ES':>3s} {'EN':>3s} {'port':>4s}   qué es")
    for es, (qué, al_port) in CIFRAS.items():
        en = es.replace(",", ".")
        hay = (es in t_es, en in t_en,
               (en in t_pt) if al_port else None)
        marca = ["sí" if h else ("--" if h is None else "NO")
                 for h in hay]
        print(f"  {es:>6s}  {marca[0]:>3s} {marca[1]:>3s} "
              f"{marca[2]:>4s}   {qué}")
        for nombre, h in zip(("ES", "EN", "port"), hay):
            if h is False:
                fallos.append(f"{es} ({qué}) falta en {nombre}")

    n = len(CIFRAS)
    total = n * 2 + sum(1 for _, (_, p) in CIFRAS.items() if p)
    print(f"\n{total} comprobaciones sobre {n} cifras y 3 PDF")
    if fallos:
        print(f"[error] {len(fallos)} fallo(s):")
        for f in fallos:
            print(f"  {f}")
    else:
        print("[ok] todas las cifras bloqueadas están donde deben")
    sys.exit(1 if fallos else 0)


if __name__ == "__main__":
    main()
