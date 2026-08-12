"""regenera el port ciego de tmlr desde el derivado inglés.

el port no admite redacción propia: solo reformatea al estilo de la
casa. este script es esa carta hecha código, de modo que las reglas
sobrevivan a cada regeneración en vez de aplicarse a mano sobre
`body.tex` ---donde morirían a la siguiente---.

reglas del transformador, todas de formato:
  1. `\\cite{k}` -> `\\citep{k}`, y `Nombre et al. \\citep{k}` ->
     `\\citet{k}` (natbib consume el nombre; el render da
     «Nombre et al. (año)» sin duplicarlo).
  2. dentro de un paréntesis ya abierto, `\\citealp` para no anidar.
  3. `\\section*{Broader Impact Statement}` -> `\\subsubsection*{...}`,
     que es como lo trae la plantilla oficial de tmlr.
  4. la sección de disponibilidad de código y datos no viaja: llevaría
     los identificadores reales al envío ciego.

uso:
    python scripts/port_tmlr.py [--verifica]
"""

import argparse
import pathlib
import re
import sys

FUENTE = "paper/angular_separation_paper_en.tex"
DESTINO = "paper_tmlr"


def a_natbib(txt: str) -> str:
    """convierte el esquema numérico de citas al autor-año de natbib.

    Args:
        txt: fragmento de LaTeX del derivado inglés.

    Returns:
        el mismo fragmento con \\citep, \\citet y \\citealp.
    """
    txt = txt.replace("\\cite{", "\\citep{")
    txt = re.sub(r"\b[A-Z][A-Za-z]+ et al\.\s*\\citep\{([a-z0-9]+)\}",
                 r"\\citet{\1}", txt)
    txt = re.sub(r"\b[A-Z][A-Za-z]+ and [A-Z][A-Za-z]+\s*"
                 r"\\citep\{([a-z0-9]+)\}", r"\\citet{\1}", txt)
    txt = txt.replace("(\\texttt{timm} 1.0.22,\n\\citep{wightman2019timm})",
                      "(\\texttt{timm} 1.0.22,\n\\citealp{wightman2019timm})")
    return txt


def a_estilo_casa(txt: str) -> str:
    """aplica las desviaciones de encabezado que la plantilla pide.

    tmlr trae el impacto amplio como \\subsubsection*; heredarlo como
    \\section* del inglés es la clase de desviación de plantilla que
    su guía desaconseja.

    Args:
        txt: fragmento de LaTeX ya convertido a natbib.

    Returns:
        el fragmento con los encabezados de la casa.
    """
    return txt.replace(
        "\\section*{Broader Impact Statement}\n"
        "\\addcontentsline{toc}{section}{Broader Impact Statement}",
        "\\subsubsection*{Broader Impact Statement}")


def parte(lineas: list[str], desde: str, hasta: str,
          incluir_inicio: bool = True) -> str:
    """recorta el fragmento entre dos marcadores de línea.

    Args:
        lineas: el fichero fuente partido en líneas.
        desde: prefijo de la línea inicial.
        hasta: prefijo de la línea final, excluida.
        incluir_inicio: si la línea inicial entra en el recorte.

    Returns:
        el fragmento como texto.
    """
    i = next(k for k, l in enumerate(lineas) if l.startswith(desde)
             or l.strip() == desde)
    j = next(k for k, l in enumerate(lineas) if k > i
             and l.startswith(hasta))
    return "\n".join(lineas[i if incluir_inicio else i + 1:j]).strip() + "\n"


def main() -> None:
    """regenera body.tex y appendix.tex, y reporta el recuento."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fuente", default=FUENTE)
    ap.add_argument("--destino", default=DESTINO)
    args = ap.parse_args()

    lineas = pathlib.Path(args.fuente).read_text(encoding="utf-8").split("\n")
    cuerpo = parte(lineas, "\\begin{abstract}", "\\section*{Code and data")
    apendice = parte(lineas, "\\appendix", "\\begin{thebibliography}",
                     incluir_inicio=False)

    cuerpo = a_estilo_casa(a_natbib(cuerpo))
    apendice = a_estilo_casa(a_natbib(apendice))

    d = pathlib.Path(args.destino)
    (d / "body.tex").write_text(cuerpo, encoding="utf-8")
    (d / "appendix.tex").write_text(apendice, encoding="utf-8")

    crudos = len(re.findall(r"\\cite\{", cuerpo + apendice))
    if crudos:
        print(f"[error] quedan {crudos} \\cite crudos", file=sys.stderr)
        sys.exit(1)

    # el corte por marcador excluye la sección de disponibilidad, pero
    # no lo que la precede: un comentario añadido justo encima del
    # \section* entra en el cuerpo y se lleva los identificadores
    # reales al envío ciego. ocurrió. el recuento no basta ---todo
    # cuadraba salvo el md5---, así que el ciego se comprueba por
    # contenido y sobre el texto completo, comentarios incluidos.
    fugas = sorted({m for m in re.findall(
        r"mmunozpl|ManPla|manpla\.net|zenodo|orcid|0009-0000-5714-912X",
        cuerpo + apendice, re.I)})
    if fugas:
        print(f"[error] el port desanonimiza: {fugas}", file=sys.stderr)
        sys.exit(1)
    print(f"body.tex {len(cuerpo.splitlines())} líneas | "
          f"citep {cuerpo.count('citep{')} citet {cuerpo.count('citet{')} "
          f"citealp {cuerpo.count('citealp{')}")
    print(f"appendix.tex {len(apendice.splitlines())} líneas")
    print("impacto amplio como subsubsection:",
          "sí" if "\\subsubsection*{Broader Impact" in cuerpo else "NO")
    print("availability fuera del port:",
          "sí" if "Code and data availability" not in cuerpo else "NO")


if __name__ == "__main__":
    main()
