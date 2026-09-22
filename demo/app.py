"""demo interactiva de la órbita de gauge valor-salida.

tres pestañas: la órbita con el conmutador ortogonal/genérico, la
decisión de poda bajo gauge, y qué se mide. toda la aritmética la
ejecuta el código del paper ---`src/gauge_flip.py`,
`scripts/run_fase_G.py`, `scripts/decision_rota.py`--- sobre el
portador ligero de `demo/portador.py`; aquí no se reimplementa nada.
"""

import math
import sys
from pathlib import Path

import gradio as gr
import matplotlib
import torch
from tqdm import tqdm

matplotlib.use("Agg")               # sin servidor gráfico en el Space

from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo.portador import (carga, verifica_manifiesto,  # noqa: E402
                           verifica_relleno)
from src.firma_funcional import w_v_columnas  # noqa: E402
from src.gauge_flip import (aplica_gauge_ortogonal,  # noqa: E402
                            aplica_gauge_ov)
from src.nucleo_lectura import decisiones, firma_exacta  # noqa: E402

DOI = "https://doi.org/10.5281/zenodo.21630534"
# gr.Markdown no renderiza matemáticas salvo que se le declaren los
# delimitadores: sin esto la prosa enseña los dólares en crudo, justo
# al lado de una figura que matplotlib sí compone bien
LATEX = [{"left": "$$", "right": "$$", "display": True},
         {"left": "$", "right": "$", "display": False}]
# clave = escala_id (el mando real); el rótulo cambia con el idioma y
# con el locale del número, la clave no
FUERZAS = {128.0: ("muy débil (0,06)", "very weak (0.06)"),
           64.0: ("débil (0,13)", "weak (0.13)"),
           32.0: ("media (0,25)", "medium (0.25)"),
           16.0: ("fuerte (0,50)", "strong (0.50)"),
           8.0: ("saturada (1,00)", "saturated (1.00)"),
           2.0: ("muy fuerte (4,05)", "very strong (4.05)")}


def op_fuerza(idi: str):
    """opciones del desplegable de fuerza en un idioma.

    Args:
        idi: 'es' o 'en'.

    Returns:
        lista de pares (rótulo, valor) para gradio.
    """
    j = 0 if idi == "es" else 1
    return [(v[j], k) for k, v in FUERZAS.items()]
# por columna: (etiqueta del sector, k de poda, referencia de la tabla 3).
# el k es el 25 % relativo de cada una, y la referencia es la medida
# publicada de esa columna ---nunca la de la otra---
COLUMNAS = {
    "vitb": {"es": "ViT-B/16 (visión)", "en": "ViT-B/16 (vision)",
             "k": 3, "ref": ("92,7 % / 0,378", "92.7 % / 0.378")},
    "pythia": {"es": "Pythia-410M (lenguaje)",
               "en": "Pythia-410M (language)",
               "k": 4, "ref": ("90,0 % / 0,379", "90.0 % / 0.379")},
}


def op_tipo(idi: str):
    """opciones del mando de tipo de gauge en un idioma.

    el valor viaja en clave estable ('gen'/'ort') porque gradio valida
    la entrada en el servidor contra las opciones declaradas al
    construir el bloque, y `gr.update` no las cambia allí: si el valor
    fuese la palabra traducida, el mando quedaría roto en inglés.

    Args:
        idi: 'es' o 'en'.

    Returns:
        lista de pares (rótulo, clave) para gradio.
    """
    return [(T[idi]["gen"], "gen"), (T[idi]["ort"], "ort")]


def op_columna(idi: str):
    """opciones del desplegable de columna en un idioma.

    Args:
        idi: 'es' o 'en'.

    Returns:
        lista de pares (rótulo, clave) para gradio.
    """
    return [(v[idi], k) for k, v in COLUMNAS.items()]

# textos en las dos lenguas. el español es el canónico y el inglés su
# derivado, con la terminología del glosario del paper: soft probe,
# hard variant, response signature, static invariant, band, gauge.
# ojo al locale de los números: coma en es, punto en en
T = {
    "es": {
        "titulo": "# La órbita de gauge valor-salida\nDos modelos "
                  "idénticos en función, dos geometrías distintas.",
        "tab1": "Órbita", "tab2": "Decisión rota", "tab3": "Qué se mide",
        "figb": "El abismo, cabeza a cabeza",
        "figb_t": "Capa {capa}: lo que el gauge mueve y lo que no",
        "figb_1": r"$\Delta v_1(W_O)$  [grados]",
        "figb_2": r"$1-|\cos|$  del circuito OV",
        "figb_m": "media", "figb_0": "0 exacto",
        "figb_s": "suelo de arccos en doble precisión",
        "figb_n": "eje común en escala logarítmica: la distancia "
                  "horizontal entre los dos bloques es el abismo",
        "btn3": "Barrer fuerzas",
        "figs": "Barrido de fuerzas",
        "figs_t": "Capa {capa}, semilla {sem}: deriva contra la "
                  "fuerza del gauge",
        "figs_x": "desviación de R respecto a un múltiplo escalar",
        "fig": "Similitud entre cabezas",
        "fig_t": "Capa {capa}: |cos| entre las direcciones "
                 "$v_1(W_O)$ de cada par de cabezas",
        "fig_a": "Antes del gauge", "fig_d": "Después del gauge",
        "fig_e": "cabeza", "fig_p": "par elegido",
        "col": "Columna", "capa": "Capa", "sem": "Semilla del gauge",
        "fuerza": "Fuerza (desv. de R al escalar)", "tipo": "Tipo de gauge",
        "gen": "genérico", "ort": "ortogonal",
        "btn1": "Muestrear gauge", "btn2": "Decidir bajo gauge",
        "porcabeza": "Por cabeza",
        "cab": ["cabeza", "Δ v1(W_O) [°]", "1-|cos| circuito OV"],
        "cab_d": "**{col}, capa {capa}, semilla {sem}, k={k}.**\n",
        "res": "**{col}, capa {capa}, gauge {tipo}, semilla {sem}.**\n\n"
               "- Desplazamiento medio de $v_1(W_O)$: **{gp}°**\n"
               "- Desviación media del circuito OV (invariante), "
               "en la métrica que certifica el paper: "
               "$1-|\\cos|$ = **{gi}**\n\nMisma libertad de gauge, dos "
               "resultados: {ver}. El circuito OV no se mueve en ningún "
               "caso ---es el objeto identificable---.",
        "ver_ort": "el subgrupo ortogonal **no la mueve**: la lectura "
                   "por pesos sobrevive intacta",
        "ver_gen": "la parte no ortogonal **sí la mueve**, decenas de "
                   "grados",
        "cols_d": ["criterio", "par antes", "par después", "¿cambia?",
                   "solape"],
        "crit_p": "por pesos $v_1(W_O)$", "crit_i": "por el invariante",
        "si": "**sí**", "no": "no",
        "contador": "**En esta sesión**, la decisión por pesos ha "
                    "cambiado en {c} de {n} gauges ({pct:.0f} %). La "
                    "medida publicada para esta columna es {ref} (par / "
                    "solape); la decisión por el invariante no cambia "
                    "nunca.",
    },
    "en": {
        "titulo": "# The value-output gauge orbit\nTwo models identical "
                  "in function, two different geometries.",
        "tab1": "Orbit", "tab2": "Broken decision",
        "tab3": "What is measured",
        "figb": "The gap, head by head",
        "figb_t": "Layer {capa}: what the gauge moves and what it "
                  "does not",
        "figb_1": r"$\Delta v_1(W_O)$  [degrees]",
        "figb_2": r"$1-|\cos|$  of the OV circuit",
        "figb_m": "mean", "figb_0": "exactly 0",
        "figb_s": "arccos floor in double precision",
        "figb_n": "shared logarithmic axis: the horizontal distance "
                  "between the two blocks is the gap",
        "btn3": "Sweep strengths",
        "figs": "Strength sweep",
        "figs_t": "Layer {capa}, seed {sem}: drift against gauge "
                  "strength",
        "figs_x": "deviation of R from a scalar multiple",
        "fig": "Similarity across heads",
        "fig_t": "Layer {capa}: |cos| between the $v_1(W_O)$ "
                 "directions of every pair of heads",
        "fig_a": "Before the gauge", "fig_d": "After the gauge",
        "fig_e": "head", "fig_p": "selected pair",
        "col": "Column", "capa": "Layer", "sem": "Gauge seed",
        "fuerza": "Strength (deviation of R from a scalar)",
        "tipo": "Gauge type", "gen": "generic", "ort": "orthogonal",
        "btn1": "Sample a gauge", "btn2": "Decide under gauge",
        "porcabeza": "Per head",
        "cab": ["head", "Δ v1(W_O) [°]", "1-|cos| OV circuit"],
        "cab_d": "**{col}, layer {capa}, seed {sem}, k={k}.**\n",
        "res": "**{col}, layer {capa}, {tipo} gauge, seed {sem}.**\n\n"
               "- Mean displacement of $v_1(W_O)$: **{gp}°**\n"
               "- Mean deviation of the OV circuit (invariant), "
               "in the metric the paper certifies: "
               "$1-|\\cos|$ = **{gi}**\n\nThe same gauge freedom, "
               "two outcomes: "
               "{ver}. The OV circuit does not move in either case "
               "---it is the identifiable object---.",
        "ver_ort": "the orthogonal subgroup **does not move it**: the "
                   "weight reading survives intact",
        "ver_gen": "the non-orthogonal part **does move it**, by tens "
                   "of degrees",
        "cols_d": ["criterion", "pair before", "pair after", "changes?",
                   "overlap"],
        "crit_p": "by weights $v_1(W_O)$", "crit_i": "by the invariant",
        "si": "**yes**", "no": "no",
        "contador": "**In this session**, the decision by weights has "
                    "changed in {c} of {n} gauges ({pct:.0f} %). The "
                    "published measurement for this column is {ref} "
                    "(pair / overlap); the decision by the invariant "
                    "never changes.",
    },
}
# el inglés va primero: es el idioma de partida de la interfaz
IDIOMAS = {"English": "en", "Español": "es"}


def num(x: float, idi: str, dec: int = 3) -> str:
    """formatea un número con el separador decimal del idioma.

    Args:
        x: valor.
        idi: 'es' o 'en'.
        dec: decimales.

    Returns:
        el número como texto, con coma o punto según el idioma.
    """
    s = f"{x:.{dec}f}"
    return s.replace(".", ",") if idi == "es" else s


_CACHE: dict = {}

verifica_manifiesto()   # el código vendido, firmado contra su commit


def cientifico(x: float, idi: str) -> str:
    """notación científica con el separador decimal del idioma.

    Args:
        x: el valor.
        idi: 'es' o 'en'.

    Returns:
        la cadena formateada.
    """
    s = f"{x:.1e}"
    return s.replace(".", ",") if idi == "es" else s


def portador(col: str):
    """carga perezosa del portador de una columna, con su contrato.

    Args:
        col: etiqueta de columna de `COLUMNAS`.

    Returns:
        el portador ya verificado.
    """
    if col not in _CACHE:
        p = carga(col)
        verifica_relleno(p)         # el contrato, vigilado también aquí
        _CACHE[col] = p
    return _CACHE[col]


def capas_de(col: str):
    """actualiza el desplegable de capas al cambiar de columna.

    Args:
        col: etiqueta de columna.

    Returns:
        actualización de gradio con las capas de esa columna.
    """
    n = len(portador(col).blocks)
    return gr.update(choices=list(range(n)), value=min(5, n - 1))


def _v1_pesos(port, capa: int) -> torch.Tensor:
    """dirección dominante de w_o por cabeza.

    Args:
        port: portador del sector.
        capa: índice de capa.

    Returns:
        tensor [h, d] de direcciones unitarias.
    """
    # sin bajar a simple: el coseno entre dos direcciones
    # casi iguales satura en fp32 y el invariante aparece
    # moviéndose centésimas de grado que no existen
    return firma_exacta(port.w_o_por_cabeza(capa).double(),
                        "der")


def _v1_invariante(port, capa: int) -> torch.Tensor:
    """dirección dominante del circuito ov, invariante de gauge.

    Args:
        port: portador del sector.
        capa: índice de capa.

    Returns:
        tensor [h, d] de direcciones unitarias.
    """
    w_o = port.w_o_por_cabeza(capa).double()
    w_v = w_v_columnas(port, capa, port.n_cabezas,
                       port.dim_cabeza).double()
    r = torch.linalg.qr(w_v, mode="reduced")[1]
    return firma_exacta(r @ w_o, "der")


def _con_gauge(col: str, capa: int, semilla: int,
               fuerza: float, tipo: str):
    """devuelve una copia del portador con un gauge aplicado.

    Args:
        capa: capa sobre la que actuar.
        semilla: semilla del generador de R.
        fuerza: escala del término identidad de R (ignorada si
            el gauge es ortogonal, que no tiene intensidad).
        tipo: 'gen' u 'ort'.

    Returns:
        tupla (portador con el gauge aplicado, desviación media de R
        respecto a su mejor múltiplo escalar; 0 si es ortogonal, que
        no tiene intensidad).
    """
    q = portador(col).copia()
    if tipo == "ort":
        aplica_gauge_ortogonal(q, capa, q.n_cabezas, q.dim_cabeza,
                               semilla=semilla)
        return q, 0.0
    desv = aplica_gauge_ov(q, capa, q.n_cabezas, q.dim_cabeza,
                           semilla=semilla, escala_id=fuerza)
    return q, desv


def _mapa_cos(v: torch.Tensor):
    """|cos| entre todas las parejas de direcciones de una capa.

    Args:
        v: tensor [h, d] de direcciones unitarias.

    Returns:
        tupla (matriz [h, h] en numpy, par de máximo fuera de la
        diagonal).
    """
    g = (v @ v.t()).abs().clamp(max=1.0)
    m = g.clone()
    m.fill_diagonal_(-1.0)
    i = int(m.argmax())
    return g.numpy(), (i // g.shape[0], i % g.shape[0])


def figura_cos(antes: torch.Tensor, desp: torch.Tensor, capa: int,
               idi: str) -> Figure:
    """dibuja la matriz de similitud antes y después del gauge.

    es la misma lectura que produce la decisión de poda: la celda más
    brillante fuera de la diagonal es el par que se podaría. verla
    moverse entre los dos paneles es la afirmación del paper hecha
    imagen.

    Args:
        antes: direcciones [h, d] antes del gauge.
        desp: direcciones [h, d] después.
        capa: índice de capa, para el título.
        idi: 'es' o 'en'.

    Returns:
        la figura de matplotlib, lista para `gr.Plot`.
    """
    d = T[idi]
    fig = Figure(figsize=(9.2, 4.2), dpi=110)
    ejes = fig.subplots(1, 2)
    for ax, x, sub in ((ejes[0], antes, d["fig_a"]),
                       (ejes[1], desp, d["fig_d"])):
        g, par = _mapa_cos(x)
        im = ax.imshow(g, cmap="magma", vmin=0.0, vmax=1.0)
        for i, j in (par, par[::-1]):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor="#39d353",
                                   lw=1.8))
        ax.set_title(f"{sub} \u00b7 {d['fig_p']} "
                     f"{tuple(sorted(par))}",
                     fontsize=10)
        ax.set_xlabel(d["fig_e"], fontsize=9)
        ax.set_ylabel(d["fig_e"], fontsize=9)
        ax.tick_params(labelsize=8)
    fig.colorbar(im, ax=ejes, fraction=0.032, pad=0.02)
    fig.suptitle(d["fig_t"].format(capa=capa), fontsize=11)
    return fig


def figura_abismo(gp: torch.Tensor, gi: torch.Tensor, capa: int,
                  idi: str) -> Figure:
    """dibuja la separación entre lo que se mueve y lo que no.

    la tabla por cabeza dice que $v_1(W_O)$ deriva decenas de grados y
    que el circuito OV se queda en 1e-16, pero eso hay que leerlo. con
    los dos bloques de barras sobre un mismo eje logarítmico, el
    abismo ---dieciocho órdenes de magnitud--- es distancia en la
    pantalla. bajo el conmutador ortogonal las barras de arriba
    desaparecen y las de abajo no se inmutan: c1 en una imagen.

    Args:
        gp: desplazamiento de v1(w_o) por cabeza, en grados.
        gi: desviación 1-|cos| del circuito ov por cabeza.
        capa: índice de capa, para el título.
        idi: 'es' o 'en'.

    Returns:
        la figura de matplotlib, lista para `gr.Plot`.
    """
    d = T[idi]
    n = gp.shape[0]
    y = list(range(n))
    fig = Figure(figsize=(9.2, 0.9 + 0.42 * n), dpi=110)
    ejes = fig.subplots(2, 1, sharex=True)
    for ax, v, etq, color in ((ejes[0], gp, d["figb_1"], "#c4432b"),
                              (ejes[1], gi, d["figb_2"], "#2b6cc4")):
        ax.barh(y, v.numpy(), height=0.72, color=color)
        # un valor exactamente nulo no dibuja barra en escala
        # logarítmica y se leería como dato ausente; se rotula, que es
        # justo lo que ocurre con el gauge ortogonal en el panel de
        # arriba: cero exacto, no falta de medida
        for h in y:
            if float(v[h]) == 0.0:
                ax.text(2e-18, h, d["figb_0"], va="center",
                        ha="left", fontsize=7, color=color)
        ax.set_ylabel(d["fig_e"], fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels([str(h) for h in y], fontsize=7)
        ax.invert_yaxis()
        ax.grid(axis="x", ls=":", lw=0.6, alpha=0.5)
        ax.set_title(f"{etq}  ·  {d['figb_m']} "
                     f"{float(v.mean()):.2e}", fontsize=10)
    # el panel de grados tiene suelo: arccos cerca de 1 pierde la
    # mitad de los dígitos, así que sqrt(2*eps) rad es lo mínimo
    # medible. sin la línea, las barras del gauge ortogonal ---que
    # deben ser cero--- se leerían como una deriva de 1e-6 grados
    suelo = math.degrees(math.sqrt(2 * torch.finfo(torch.float64).eps))
    ejes[0].axvline(suelo, ls="--", lw=1.0, color="#555555")
    ejes[0].text(suelo * 1.4, n - 0.4, d["figb_s"],
                 fontsize=7, color="#555555", va="center")
    # el eje común es lo que convierte dos medidas en un abismo; el
    # suelo baja hasta la precisión de máquina para que el bloque de
    # abajo tenga dónde caber
    ejes[1].set_xscale("log")
    ejes[1].set_xlim(1e-18, 1e3)
    ejes[1].set_xlabel(d["figb_n"], fontsize=8)
    fig.suptitle(d["figb_t"].format(capa=capa), fontsize=11)
    fig.tight_layout()
    return fig


def barrido(idioma: str, col: str, capa: int, semilla: int) -> Figure:
    """recorre las seis fuerzas y dibuja la deriva contra cada una.

    es la tabla 2 del paper dibujándose en vivo: la deriva de
    $v_1(W_O)$ crece con la desviación de R respecto a un escalar,
    mientras el circuito OV se queda plano en el suelo de la máquina.
    cuesta seis gauges por pulsación, y por eso tiene botón propio.

    Args:
        idioma: rótulo del selector.
        col: clave de columna.
        capa: capa a interrogar.
        semilla: semilla del gauge.

    Returns:
        la figura de matplotlib, lista para `gr.Plot`.
    """
    idi = IDIOMAS[idioma]
    d = T[idi]
    p = portador(col)
    antes_p, antes_i = _v1_pesos(p, capa), _v1_invariante(p, capa)
    xs, ys_p, ys_i = [], [], []
    for escala in tqdm(sorted(FUERZAS, reverse=True),
                       desc=f"barrido {col} L{capa}", leave=False):
        q, desv = _con_gauge(col, capa, semilla, escala, "gen")
        cp = _cos_abs(antes_p, _v1_pesos(q, capa))
        ci = _cos_abs(antes_i, _v1_invariante(q, capa))
        xs.append(desv)
        ys_p.append(float(torch.rad2deg(torch.arccos(cp)).mean()))
        ys_i.append(float((1.0 - ci).mean()))
    fig = Figure(figsize=(8.4, 5.4), dpi=110)
    ejes = fig.subplots(2, 1, sharex=True)
    # dos paneles y no un eje común: sobre veintiuna décadas el
    # crecimiento de v1 ---de veinte a ochenta grados--- sería
    # invisible. arriba la subida, abajo la planitud, cada una en su
    # escala; el abismo entre ambas lo cuenta la otra figura
    ejes[0].plot(xs, ys_p, "o-", color="#c4432b")
    ejes[0].set_ylim(0.0, 92.0)
    ejes[0].set_ylabel(d["figb_1"], fontsize=9)
    ejes[1].plot(xs, ys_i, "s-", color="#2b6cc4")
    ejes[1].set_yscale("log")
    ejes[1].set_ylim(1e-18, 1e-12)
    ejes[1].set_ylabel(d["figb_2"], fontsize=9)
    ejes[1].set_xscale("log")
    ejes[1].set_xlabel(d["figs_x"], fontsize=9)
    for ax in ejes:
        ax.grid(ls=":", lw=0.6, alpha=0.5)
        ax.tick_params(labelsize=8)
    fig.suptitle(d["figs_t"].format(capa=capa, sem=semilla),
                 fontsize=11)
    fig.tight_layout()
    return fig


def _cos_abs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """|cos| entre direcciones emparejadas por cabeza.

    Args:
        a: direcciones [h, d].
        b: direcciones [h, d].

    Returns:
        tensor [h] con el coseno en valor absoluto, acotado a 1.
    """
    return (a * b).sum(1).abs().clamp(max=1.0)


def orbita(idioma: str, col: str, capa: int, semilla: int,
           fuerza: float, tipo: str):
    """mide cuánto mueve el gauge la lectura por pesos y el invariante.

    Args:
        capa: capa a interrogar.
        semilla: semilla del gauge.
        fuerza: escala del término identidad de R.
        tipo: 'gen' u 'ort'.

    Returns:
        tupla (resumen en markdown, tabla por cabeza, figura del
        abismo).
    """
    p = portador(col)
    antes_p, antes_i = _v1_pesos(p, capa), _v1_invariante(p, capa)
    q, _ = _con_gauge(col, capa, semilla, fuerza, tipo)
    desp_p, desp_i = _v1_pesos(q, capa), _v1_invariante(q, capa)
    # v1 en grados, que es la magnitud interpretable; el invariante en
    # 1-|cos|, porque arccos cerca de 1 pierde la mitad de los dígitos
    # y pondría un suelo de ~1e-6 grados donde no hay movimiento
    gp = torch.rad2deg(torch.arccos(_cos_abs(antes_p, desp_p)))
    gi = 1.0 - _cos_abs(antes_i, desp_i)
    d = T[IDIOMAS[idioma]]
    idi = IDIOMAS[idioma]
    filas = [[h, num(float(gp[h]), idi, 2),
              cientifico(float(gi[h]), idi)]
             for h in range(p.n_cabezas)]
    ver = d["ver_ort"] if tipo == "ort" else d["ver_gen"]
    cient = cientifico(float(gi.mean()), idi)
    resumen = d["res"].format(
        col=COLUMNAS[col][idi], capa=capa, tipo=d[tipo],
        sem=semilla,
        gp=num(float(gp.mean()), idi, 2),
        gi=cient, ver=ver)
    return resumen, filas, figura_abismo(gp, gi, capa, idi)


def decision(idioma: str, col: str, capa: int, semilla: int,
             fuerza: float, estado: dict):
    """materializa la decisión de poda antes y después del gauge.

    Args:
        capa: capa a interrogar.
        semilla: semilla del gauge.
        fuerza: escala del término identidad de R.
        estado: contador acumulado de la sesión.

    Returns:
        tupla (markdown, estado actualizado, mapa de similitud).
    """
    p = portador(col)

    idi = IDIOMAS[idioma]
    d = T[idi]
    k = COLUMNAS[col]["k"]
    ref = COLUMNAS[col]["ref"][0 if idi == "es" else 1]
    nombre_col = COLUMNAS[col][idi]
    q, _ = _con_gauge(col, capa, semilla, fuerza, "gen")
    lineas = [d["cab_d"].format(col=nombre_col, capa=capa,
                                sem=semilla, k=k),
              "| " + " | ".join(d["cols_d"]) + " |",
              "|---|---|---|---|---|"]
    for nombre, f in ((d["crit_p"], _v1_pesos),
                      (d["crit_i"], _v1_invariante)):
        par0, top0 = decisiones(f(p, capa), k)
        par1, top1 = decisiones(f(q, capa), k)
        sol = len(set(top0) & set(top1)) / k
        cambia = par1 != par0
        if nombre == d["crit_p"]:
            estado["n"] = estado.get("n", 0) + 1
            estado["c"] = estado.get("c", 0) + int(cambia)
        lineas.append(f"| {nombre} | {par0} | {par1} | "
                      f"{d['si'] if cambia else d['no']} | "
                      f"{num(sol, idi, 2)} |")
    pct = 100 * estado["c"] / estado["n"]
    lineas.append("")
    lineas.append(d["contador"].format(c=estado["c"],
                                       n=estado["n"], pct=pct,
                                       ref=ref))
    # el mapa vive aquí y no en la órbita: es la lectura de la que
    # sale esta tabla, y verlo debajo convierte la fila «cambia: sí»
    # en algo comprobable en vez de en un acto de fe
    mapa = figura_cos(_v1_pesos(p, capa), _v1_pesos(q, capa),
                      capa, idi)
    return "\n".join(lineas), estado, mapa


QSM = {"es": f"""
### Qué se mide aquí

Una cabeza de atención escribe en el flujo residual a través de dos
matrices, $W_v$ y $W_O$. Esa factorización **no es única**: para
cualquier $R$ invertible, sustituir $W_v \\to W_v R$, $b_v \\to b_v R$ y
$W_O \\to R^{{-1}} W_O$ deja la función del modelo exactamente igual,
porque el producto $W_v W_O$ ---el circuito OV--- y el término de
sesgo $b_v W_O$ no cambian. Es la transformación que este código
aplica.

La dirección dominante $v_1(W_O)$ **no** es función de ese producto,
así que se mueve con $R$ mientras el modelo calcula lo mismo: su
órbita bajo el gauge es la esfera unitaria completa de su espacio
fila, y por tanto **ningún umbral de similitud sobre esa dirección
admite radio certificado positivo**.

Un matiz que la primera pestaña enseña: el subgrupo **ortogonal** deja
$v_1(W_O)$ quieta. Es la parte **no ortogonal** del gauge la que la
mueve, y está presente en cualquier reparametrización genérica.

Los pesos son los de un ViT-B/16 afinado en ImageNet-100 (semilla 42)
y los de Pythia-410M sin entrenar nada. Cada gauge muestreado es
reproducible: basta repetir la semilla.

Código, datos y certificación de la órbita: <{DOI}>
""", "en": f"""
### What is measured here

An attention head writes into the residual stream through two
matrices, $W_v$ and $W_O$. That factorization is **not unique**: for
any invertible $R$, substituting $W_v \\to W_v R$, $b_v \\to b_v R$ and
$W_O \\to R^{{-1}} W_O$ leaves the model's function exactly as it was,
because the product $W_v W_O$ ---the OV circuit--- and the bias term
$b_v W_O$ do not change. That is the transformation this code
applies.

The dominant direction $v_1(W_O)$ is **not** a function of that
product, so it moves with $R$ while the model computes the same
thing: its orbit under the gauge is the full unit sphere of its row
space, and therefore **no similarity threshold on that direction
admits a positive certified radius**.

One nuance the first tab shows: the **orthogonal** subgroup leaves
$v_1(W_O)$ still. It is the **non-orthogonal** part of the gauge that
moves it, and it is present in any generic reparametrization.

The weights are those of a ViT-B/16 fine-tuned on ImageNet-100
(seed 42) and of Pythia-410M with no training at all. Every sampled
gauge is reproducible: just repeat the seed.

Code, data and the orbit certification: <{DOI}>
"""}


def cambia_idioma(idioma: str):
    """rehace etiquetas y textos estáticos en el idioma elegido.

    Args:
        idioma: clave de `IDIOMAS`.

    Returns:
        tupla de actualizaciones de gradio, en el orden de los
        componentes que dependen del idioma (rótulos de pestaña
        incluidos, al final).
    """
    idi = IDIOMAS[idioma]
    d = T[idi]
    col = gr.update(choices=op_columna(idi), label=d["col"])
    fue = gr.update(choices=op_fuerza(idi), value=8.0, label=d["fuerza"])
    return (gr.update(value=d["titulo"]),
            col, gr.update(label=d["capa"]),
            gr.update(label=d["sem"]), fue,
            gr.update(choices=op_tipo(idi), value="gen",
                      label=d["tipo"]),
            gr.update(value=d["btn1"]), gr.update(value=d["btn3"]),
            gr.update(label=d["figs"]),
            gr.update(label=d["figb"]),
            gr.update(headers=d["cab"], label=d["porcabeza"]),
            col, gr.update(label=d["capa"]),
            gr.update(label=d["sem"]), fue,
            gr.update(value=d["btn2"]),
            gr.update(label=d["fig"]),
            gr.update(value=QSM[idi]),
            gr.update(label=d["tab1"]), gr.update(label=d["tab2"]),
            gr.update(label=d["tab3"]))


TITULO = "The value-output gauge orbit - La órbita de gauge "\
         "valor-salida"

with gr.Blocks(title=TITULO) as demo:
    idi = gr.Radio(list(IDIOMAS), value="English", label="Idioma / Language")
    cab = gr.Markdown(T["en"]["titulo"], latex_delimiters=LATEX)
    with gr.Tab(T["en"]["tab1"]) as pes1:
        with gr.Row():
            col_o = gr.Dropdown(op_columna("en"), value="vitb",
                                label=T["en"]["col"])
            capa_o = gr.Dropdown(list(range(12)), value=5,
                                 label=T["en"]["capa"])
            sem_o = gr.Number(value=0, precision=0, label=T["en"]["sem"])
            fue_o = gr.Dropdown(op_fuerza("en"), value=8.0,
                                label=T["en"]["fuerza"])
            tip_o = gr.Radio(op_tipo("en"), value="gen",
                             label=T["en"]["tipo"])
        with gr.Row():
            btn_o = gr.Button(T["en"]["btn1"], variant="primary")
            btn_s = gr.Button(T["en"]["btn3"])
        # el barrido va pegado a su botón y en un desplegable que se
        # abre solo: con la salida al final de la pestaña, debajo de
        # la tabla por cabeza, quedaba fuera de pantalla y el botón
        # parecía no hacer nada
        with gr.Accordion(T["en"]["figs"], open=False) as ple_s:
            fig_s = gr.Plot(show_label=False)
        res_o = gr.Markdown(latex_delimiters=LATEX)
        fig_o = gr.Plot(label=T["en"]["figb"])
        tab_o = gr.Dataframe(headers=T["en"]["cab"],
                             label=T["en"]["porcabeza"])
        col_o.change(capas_de, col_o, capa_o)
        btn_o.click(orbita, [idi, col_o, capa_o, sem_o, fue_o, tip_o],
                    [res_o, tab_o, fig_o])
        btn_s.click(barrido, [idi, col_o, capa_o, sem_o], fig_s).then(
            lambda: gr.update(open=True), None, ple_s)
    with gr.Tab(T["en"]["tab2"]) as pes2:
        with gr.Row():
            col_d = gr.Dropdown(op_columna("en"), value="vitb",
                                label=T["en"]["col"])
            capa_d = gr.Dropdown(list(range(12)), value=5,
                                 label=T["en"]["capa"])
            sem_d = gr.Number(value=0, precision=0, label=T["en"]["sem"])
            fue_d = gr.Dropdown(op_fuerza("en"), value=8.0,
                                label=T["en"]["fuerza"])
        btn_d = gr.Button(T["en"]["btn2"], variant="primary")
        res_d = gr.Markdown(latex_delimiters=LATEX)
        fig_m = gr.Plot(label=T["en"]["fig"])
        est = gr.State({})
        col_d.change(capas_de, col_d, capa_d)
        btn_d.click(decision, [idi, col_d, capa_d, sem_d, fue_d, est],
                    [res_d, est, fig_m])
    with gr.Tab(T["en"]["tab3"]) as pes3:
        qsm = gr.Markdown(QSM["en"], latex_delimiters=LATEX)
    idi.change(cambia_idioma, idi,
               [cab, col_o, capa_o, sem_o, fue_o, tip_o, btn_o, btn_s,
                ple_s, fig_o, tab_o, col_d, capa_d, sem_d, fue_d,
                btn_d, fig_m, qsm, pes1, pes2, pes3])

if __name__ == "__main__":
    demo.launch()
