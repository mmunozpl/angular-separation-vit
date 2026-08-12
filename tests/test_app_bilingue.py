"""la capa bilingüe de la demo no desalinea la interfaz.

`cambia_idioma` devuelve una tupla posicional que gradio reparte entre
los componentes cableados como salida. añadir un componente a la
interfaz y olvidarlo en la lista ---o al revés--- desplaza todas las
actualizaciones a partir de ahí, y el fallo no aparece hasta que
alguien pulsa el selector de idioma en producción. estas pruebas
convierten esa alineación en algo que se rompe en el banco.

el módulo pide gradio y matplotlib, que no viven en el entorno de
análisis; donde falten, se salta.
"""

import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

gr = pytest.importorskip("gradio",
                         reason="la app solo corre en el venv del Space")
if not hasattr(gr, "Blocks"):
    # desinstalar gradio deja sus stubs .pyi: el paquete se importa,
    # está vacío, y sin esto el módulo fallaría al recolectar
    pytest.skip("gradio incompleto (solo restos de una desinstalación)",
                allow_module_level=True)
pytest.importorskip("matplotlib", reason="la figura pide matplotlib")

SECTOR = "artifacts/demo/sector_vo_vitb.safetensors"
pytestmark = pytest.mark.skipif(
    not os.path.exists(SECTOR),
    reason="falta el sector; correr extraer_sector_vo.py")

import demo.app as app  # noqa: E402


def _evento_de_idioma():
    """localiza el evento del selector de idioma en el bloque.

    Returns:
        el objeto de evento de gradio.

    Raises:
        AssertionError: si el bloque ya no lo cablea.
    """
    for fn in app.demo.fns.values():
        if getattr(fn, "fn", None) is app.cambia_idioma:
            return fn
    raise AssertionError("el selector de idioma no está cableado")


def test_una_actualizacion_por_componente() -> None:
    """la tupla devuelta cubre exactamente las salidas cableadas."""
    fn = _evento_de_idioma()
    for idioma in app.IDIOMAS:
        assert len(app.cambia_idioma(idioma)) == len(fn.outputs), (
            f"{idioma}: la tupla y la lista de salidas se han "
            f"desalineado")


def test_cada_actualizacion_cabe_en_su_componente() -> None:
    """ninguna actualización lleva claves ajenas a su destino.

    es la comprobación que atrapa el desplazamiento aunque los dos
    recuentos cuadren por casualidad: pasarle `headers` a un
    desplegable solo falla al pulsar el selector.
    """
    fn = _evento_de_idioma()
    for idioma in app.IDIOMAS:
        for upd, comp in zip(app.cambia_idioma(idioma), fn.outputs):
            acepta = inspect.signature(
                type(comp).__init__).parameters
            for clave in upd:
                if clave == "__type__":
                    continue
                assert clave in acepta, (
                    f"{idioma}: '{clave}' no lo admite "
                    f"{type(comp).__name__}; las salidas van corridas")


def test_los_markdown_declaran_los_delimitadores_de_latex() -> None:
    """la prosa con matemáticas no puede salir con los dólares crudos.

    el defecto de `gr.Markdown` trae solo `$$` de display, y toda la
    matemática de la demo ---$v_1(W_O)$, $1-|\\cos|$--- es en línea,
    con un solo dólar. comprobar que la lista no está vacía no vale:
    nunca lo está. hay que exigir el delimitador de línea, que es el
    que faltaba cuando la prosa salía con los dólares crudos.
    """
    import gradio as gr

    md = [b for b in app.demo.blocks.values()
          if isinstance(b, gr.Markdown)]
    assert md, "no hay componentes Markdown que comprobar"
    sin = [m for m in md
           if not any(x.get("left") == "$" and not x.get("display")
                      for x in (m.latex_delimiters or []))]
    assert not sin, (f"{len(sin)} de {len(md)} componentes Markdown "
                     f"sin delimitador de matemática en línea")


def test_las_dos_lenguas_tienen_las_mismas_claves() -> None:
    """añadir un texto en una lengua y olvidarlo en la otra falla aquí.

    el fallo natural sería un `KeyError` en producción, y solo en el
    idioma descuidado.
    """
    es, en = set(app.T["es"]), set(app.T["en"])
    assert es == en, (f"solo en es: {sorted(es - en)}; "
                      f"solo en en: {sorted(en - es)}")


def _visibles(fuente) -> list[str]:
    """textos que el visitante llega a leer, de componente o de update.

    tolera cualquier tipo en los tres campos: el valor de un
    desplegable puede ser un float y las cabeceras una lista.

    Args:
        fuente: componente de gradio, o el dict de una actualización.

    Returns:
        lista de cadenas visibles.
    """
    lee = (fuente.get if isinstance(fuente, dict)
           else lambda k: getattr(fuente, k, None))
    out = []
    for campo in ("label", "value", "headers"):
        v = lee(campo)
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, (list, tuple)):
            out += [x for x in v if isinstance(x, str)]
    return out


def test_ningun_rotulo_sobrevive_en_el_idioma_equivocado() -> None:
    """tras cambiar de idioma no queda texto de la otra lengua.

    el banco vigila estructura y paridad de claves; esto vigila
    **cobertura**. un componente que se añade a la interfaz con su
    rótulo y se olvida en la lista de salidas conserva el idioma de
    construcción y sobrevive traducido a medias ---exactamente lo que
    pasó al mudar la figura del barrido dentro de un acordeón---.
    se simula el cambio aplicando las actualizaciones sobre los
    rótulos de construcción, que es lo que hace el navegador.
    """
    fn = _evento_de_idioma()
    for destino, otro in (("es", "en"), ("en", "es")):
        idioma = [k for k, v in app.IDIOMAS.items() if v == destino][0]
        # cadenas que solo existen en la lengua que hay que abandonar
        ajenas = set()
        for v in app.T[otro].values():
            if isinstance(v, str):
                ajenas.add(v)
            elif isinstance(v, list):
                ajenas.update(x for x in v if isinstance(x, str))
        ajenas -= {x for k, v in app.T[destino].items()
                   for x in ([v] if isinstance(v, str) else
                             v if isinstance(v, list) else [])}
        # se parte de **todos** los componentes, no solo de los
        # cableados: el que se olvida en la lista es precisamente el
        # que hay que cazar, y mirando solo las salidas quedaría fuera
        # del examen
        vivos = {id(c): _visibles(c)
                 for c in app.demo.blocks.values()}
        for upd, comp in zip(app.cambia_idioma(idioma), fn.outputs):
            vivos[id(comp)] = _visibles(upd)
        restos = sorted({x for xs in vivos.values() for x in xs}
                        & ajenas)
        assert not restos, (f"al pasar a '{destino}' sobreviven "
                            f"rótulos de '{otro}': {restos}")


def test_el_valor_de_los_mandos_no_depende_del_idioma() -> None:
    """columna, fuerza y tipo viajan en clave estable.

    gradio valida la entrada en el servidor contra las opciones con
    las que se construyó el bloque, y `gr.update` no las cambia allí:
    si el valor fuese la palabra traducida, el mando quedaría roto en
    el idioma que no es el de partida.
    """
    for f in (app.op_columna, app.op_fuerza, app.op_tipo):
        valores = {tuple(v for _, v in f(i)) for i in ("es", "en")}
        assert len(valores) == 1, (
            f"{f.__name__} cambia de valor con el idioma")
