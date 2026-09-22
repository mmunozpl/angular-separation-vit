"""genera el envío a SIMODS desde el derivado inglés.

gemelo de `port_tmlr.py` e inverso en anonimato: SIMODS revisa con
identidad simple, así que el autor, el ORCID y la sección de
disponibilidad ---con sus tres DOI y la demo--- van dentro. sin guard
de anonimato, con las mismas verificaciones de contenido: secciones
esperadas, citas cubiertas por la bibliografía, y salida con código
distinto de cero si el port pierde una pieza.

decisiones de clase, leídas de `siamart251216.cls` y no supuestas:
la clase carga hyperref y cleveref y define `proposition`,
`definition` y `proof` vía ntheorem ---el port no los redefine ni
carga amsthm---; los metadatos van en los entornos `keywords` y
`MSCcodes` (el nombre vigente del entorno de códigos; `AMS` es su
alias heredado).

uso:
    python scripts/port_simods.py [--verifica]
"""

import argparse
import pathlib
import re
import sys

FUENTE = "paper/angular_separation_paper_en.tex"
DESTINO = "paper_SIMODS"
BIB = "paper/refs.bib"
MSC = "68T07, 15A18, 15A23"
KEYWORDS = ("gauge non-identifiability, OV circuit, head pruning, "
            "certified radius, multi-head attention")

PREAMBULO = r"""% paquetes que el cuerpo usa y la clase no trae; amsthm,
% hyperref y
% cleveref NO se cargan: la clase los trae o usa ntheorem
\usepackage{amssymb}
\usepackage{bm}
\usepackage{microtype}
\usepackage{booktabs}
\usepackage{tabularx}
\newcolumntype{Y}{>{\centering\arraybackslash}X}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc,decorations.pathreplacing}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}

\newcommand{\Sph}{\mathbb{S}}
\newcommand{\Rset}{\mathbb{R}}
\usepackage[numbers,sort&compress]{natbib}
\providecommand{\acks}[1]{\section*{Acknowledgments}#1}
"""

MAIN = r"""% envio a SIMODS --- generado por scripts/port_simods.py;
% no editar a mano: regenerar desde el derivado ingles.
% la opcion review activa la numeracion de lineas que SIMODS exige
% durante la revision (peticion editorial del 19-08-2026); la
% version aceptada la retirara
\documentclass[review]{siamart251216}

__PREAMBULO__


\hypersetup{
  pdftitle={The dominant direction of W_O is not identifiable: gauge
    orbit, zero certified radius, and consequences for pruning},
  pdfauthor={Manuel Munoz Pla},
}

\title{The dominant direction of
  \texorpdfstring{$\bm{W}_O$}{W\_O} is not identifiable: gauge
  orbit, zero certified radius, and consequences for pruning}
\author{Manuel Mu\~noz Pl\'a\thanks{Independent researcher, Seville,
  Spain (\email{mmunozpl@uoc.edu}; ORCID 0009-0000-5714-912X). No external
  funding supported this work.}}
\headers{The dominant direction of
  \texorpdfstring{$\bm{W}_O$}{W\_O} is not identifiable}
  {M. Mu\~noz Pl\'a}

\begin{document}
\maketitle

\begin{abstract}
__ABSTRACT__
\end{abstract}

\begin{keywords}
__KEYWORDS__
\end{keywords}

\begin{MSCcodes}
__MSC__
\end{MSCcodes}

\input{body}

\bibliographystyle{siamplain}
\bibliography{main}
\end{document}
"""

SUPLEMENTO = r"""% suplemento SIMODS --- generado por scripts/port_simods.py
\documentclass[review,supplement]{siamart251216}

__PREAMBULO__


\hypersetup{
  pdftitle={Supplementary Materials: The dominant direction of W_O
    is not identifiable},
  pdfauthor={Manuel Munoz Pla},
}

\title{The dominant direction of
  \texorpdfstring{$\bm{W}_O$}{W\_O} is not identifiable}
\author{Manuel Mu\~noz Pl\'a\thanks{Independent researcher, Seville,
  Spain (\email{mmunozpl@uoc.edu}; ORCID 0009-0000-5714-912X).}}
\headers{Supplementary Materials}{M. Mu\~noz Pl\'a}

\begin{document}
\maketitle

__CUERPO__
\end{document}
"""


def extrae(txt: str) -> tuple[str, str, str]:
    """separa abstract, cuerpo principal y suplemento del EN.

    el apéndice correlacional (app:beyond) sale del cuerpo hacia el
    suplemento: es la palanca del límite de 700 líneas de SIMODS, y
    su subordinación explícita lo hace el candidato natural.

    Args:
        txt: el `.tex` completo del EN.

    Returns:
        tupla (abstract, cuerpo sin el apéndice B, apéndice B).
    """
    abst = txt.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
    # la cola de palabras clave del formato article no viaja: van en
    # su entorno propio de la clase
    abst = abst.split("\\\\[4pt]")[0].replace("\\noindent", "").strip()
    ini = txt.index("\\section{Introduction}")
    fin = txt.index("\\bibliographystyle{plainnat}")
    cuerpo = a_estilo_casa(txt[ini:fin].rstrip())
    corte = cuerpo.index("\\section{Beyond the value-output sector")
    principal = cuerpo[:corte].rstrip()
    supl = cuerpo[corte:].rstrip()
    principal, supl = resuelve_cruces(principal, supl)
    return abst, principal, supl


def a_estilo_casa(cuerpo: str) -> str:
    """aplica las desviaciones de formato que la clase SIAM pide.

    dos reglas, ambas de formato y no de contenido: (i) la clase
    añade su propio punto a los títulos de \\paragraph, así que el
    punto del texto produciría «Título..» ---se retira---; (ii) la
    configuración cleveref de la clase imprime las ecuaciones ya
    entre paréntesis, de modo que «(\\cref{eq:x})» rendería
    «((5.3))» y un \\labelcref suelto quedaría sin paréntesis ---se
    normaliza todo a \\eqref, la forma de la casa---.

    Args:
        cuerpo: el body extraído del derivado inglés.

    Returns:
        el cuerpo con el formato de la casa.
    """
    cuerpo = re.sub(
        r"(\\paragraph\{[^{}]*(?:\{[^{}]*\}[^{}]*)*)\.\}",
        r"\1}", cuerpo)
    cuerpo = re.sub(r"\(\\[cC]ref\{(eq:[^}]*)\}\)",
                    r"\\eqref{\1}", cuerpo)
    cuerpo = re.sub(r"\\labelcref\{(eq:[^}]*)\}",
                    r"\\eqref{\1}", cuerpo)
    cuerpo = re.sub(r"\\[cC]ref\{(eq:[^}]*)\}",
                    r"\\eqref{\1}", cuerpo)
    # las referencias parentéticas al material del suplemento van en
    # forma corta ---(SM1.1)---: el nombre completo con su enlace es
    # un bloque irrompible que desborda la línea
    cuerpo = re.sub(
        r"\(\\cref\{(sec:results:inversion|sec:results:residuo)\}\)",
        r"(\\labelcref{\1})", cuerpo)
    return cuerpo


def resuelve_cruces(principal: str, supl: str) -> tuple[str, str]:
    """sustituye las referencias entre documentos por texto literal.

    el xr-hyper del sistema y el cleveref de la clase componen las
    referencias externas con el nombre del fichero pegado al número
    ---«SM1.1supplement.pdf»--- en ambas direcciones. como el port
    genera los dos documentos y fija su estructura, los cruces se
    resuelven aquí con la numeración **asertada**, no supuesta: si el
    orden de secciones cambia, el assert para el port antes de
    escribir un número falso.

    Args:
        principal: cuerpo del manuscrito.
        supl: cuerpo del suplemento.

    Returns:
        tupla (principal, suplemento) con los cruces resueltos.
    """
    orden = re.findall(r"\\section\{([^}]*)", principal)
    assert orden[3].startswith("The gauge-invariant"), orden[3]
    assert orden[5].startswith("Experimental setup"), orden[5]
    assert orden[6].startswith("Verification"), orden[6]
    sub7 = re.findall(r"\\subsection\{([^}]*)",
                      principal.split("\\section{Verification}")[1])
    assert sub7[0].startswith("The orbit, measured"), sub7[0]

    # main -> suplemento (numeración SM fijada por el propio port)
    principal = principal.replace(
        "(\\labelcref{sec:results:inversion})", "(SM1.1)")
    principal = principal.replace(
        "(\\labelcref{sec:results:residuo})", "(SM1.2)")
    principal = principal.replace(
        "\\Cref{sec:results:residuo} verifies",
        "Subsection~SM1.2 of the supplementary materials verifies")
    principal = principal.replace(
        "are subordinated to \\cref{app:beyond}.",
        "are subordinated to the supplementary materials "
        "(section~SM1).")
    principal = principal.replace(
        "(\\cref{sec:results:wo,sec:results:inversion})",
        "(\\cref{sec:results:wo} and~SM1.1)")

    # suplemento -> main (numeración del main, asertada arriba)
    supl = supl.replace("\\cref{sec:gauge:firma}", "section~4")
    supl = supl.replace("\\cref{sec:setup}", "section~6")
    supl = supl.replace("\\cref{sec:results:gauge}",
                        "subsection~7.1")
    for lbl in ("app:beyond", "sec:results:inversion",
                "sec:results:residuo"):
        assert f"cref{{{lbl}}}" not in principal, lbl
    for lbl in ("sec:gauge:firma", "sec:setup", "sec:results:gauge"):
        assert f"cref{{{lbl}}}" not in supl, lbl
    return principal, supl


def verifica(cuerpo: str, bib: str) -> list[str]:
    """comprueba que el port no perdió piezas ni cita en el vacío.

    Args:
        cuerpo: el body.tex generado.
        bib: contenido de main.bib.

    Returns:
        lista de fallos; vacía si todo cuadra.
    """
    fallos = []
    esperadas = [
        "\\section{Introduction}", "\\section{Related work}",
        "\\section{The value-output gauge and the zero certified",
        "\\section{The gauge-invariant response signature}",
        "\\section{Three intervention regimes",
        "\\section{Experimental setup}", "\\section{Verification}",
        "\\section{Discussion and limitations}",
        "\\section{Conclusion}",
        "\\section*{Code and data availability}",
        "\\section*{Declaration on the use of AI tools}", "\\appendix",
        "label{prop:orbita}", "label{app:orbita}",
        "label{tab:decision}",
    ]
    for e in esperadas:
        if e not in cuerpo:
            fallos.append(f"pieza perdida: {e}")
    claves = set(re.findall(r"@\w+\{([^,]+),", bib))
    citas = sorted({k.strip() for m in
                    re.findall(r"\\[cC]ite[tp]?\{([^}]*)\}", cuerpo)
                    for k in m.split(",")})
    for c in citas:
        if c not in claves:
            fallos.append(f"cita sin registro bib: {c}")
    for marca in ("\\todo{", "\\dato{", "\\captionsetup"):
        if marca in cuerpo:
            fallos.append(f"marca de trabajo en el envío: {marca}")
    return fallos


def main() -> None:
    """genera main.tex, body.tex y main.bib, y verifica el conjunto."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fuente", default=FUENTE)
    ap.add_argument("--destino", default=DESTINO)
    args = ap.parse_args()

    txt = pathlib.Path(args.fuente).read_text(encoding="utf-8")
    abst, cuerpo, supl = extrae(txt)
    bib = pathlib.Path(BIB).read_text(encoding="utf-8")

    fallos = verifica(cuerpo, bib)
    # el suplemento lleva sus propias piezas, y el cuerpo no debe
    # conservarlas: la separación es exacta o no es
    for e in ("\\section{Beyond the value-output sector",
              "label{app:beyond}", "label{sec:results:inversion}",
              "label{sec:results:residuo}"):
        if e not in supl:
            fallos.append(f"pieza perdida en el suplemento: {e}")
        if e in cuerpo:
            fallos.append(f"duplicada en el cuerpo: {e}")
    if re.search(r"\\[cC]ite[tp]?\{", supl):
        fallos.append("el suplemento cita bibliografía y no lleva")
    if fallos:
        for f in fallos:
            print(f"[error] {f}", file=sys.stderr)
        sys.exit(1)

    d = pathlib.Path(args.destino)
    main_tex = (MAIN.replace("__PREAMBULO__", PREAMBULO)
                    .replace("__ABSTRACT__", abst)
                    .replace("__KEYWORDS__", KEYWORDS)
                    .replace("__MSC__", MSC))
    supl_tex = (SUPLEMENTO.replace("__PREAMBULO__", PREAMBULO)
                          .replace("__CUERPO__", supl))
    (d / "main.tex").write_text(main_tex, encoding="utf-8")
    (d / "body.tex").write_text(cuerpo + "\n", encoding="utf-8")
    (d / "supplement.tex").write_text(supl_tex, encoding="utf-8")
    (d / "main.bib").write_text(bib, encoding="utf-8")

    n_sec = len(re.findall(r"\\section\*?\{", cuerpo))
    n_cit = len(re.findall(r"\\[cC]ite[tp]?\{", cuerpo))
    print(f"body.tex {len(cuerpo.splitlines())} líneas | "
          f"{n_sec} secciones | {n_cit} citas")
    print(f"supplement.tex {len(supl.splitlines())} líneas")
    print("identidad dentro: sí (autor, ORCID, availability)")


if __name__ == "__main__":
    main()
