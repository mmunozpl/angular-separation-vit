"""contrato de formas del sector valor-salida.

las dos convenciones ---[h, dh, d] y [h, d, dh]--- conviven porque el
aparato las necesita en ambas. un cambio silencioso entre ellas no
rompe nada visible: produce ángulos plausibles y falsos. estos asserts
lo convierten en fallo ruidoso.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.firma_funcional import (w_o_por_cabeza, w_v_columnas,  # noqa: E402
                                 w_v_por_cabeza)

SECTOR = "artifacts/demo/sector_vo_vitb.safetensors"
pytestmark = pytest.mark.skipif(
    not os.path.exists(SECTOR),
    reason="falta el sector; correr extraer_sector_vo.py")


@pytest.fixture
def portador():
    """portador del sector valor-salida de vit-b."""
    from demo.portador import carga
    return carga("vitb")


def test_formas_de_las_tres_lecturas(portador) -> None:
    """cada lectura devuelve la forma que su nombre promete."""
    nh, dh = portador.n_cabezas, portador.dim_cabeza
    d = portador.blocks[0].attn.qkv.weight.shape[1]
    assert w_o_por_cabeza(portador, 0, nh, dh).shape == (nh, dh, d)
    assert w_v_por_cabeza(portador, 0, nh, dh).shape == (nh, dh, d)
    assert w_v_columnas(portador, 0, nh, dh).shape == (nh, d, dh)


def test_columnas_es_la_traspuesta(portador) -> None:
    """las dos convenciones son la misma matriz, no dos matrices."""
    nh, dh = portador.n_cabezas, portador.dim_cabeza
    filas = w_v_por_cabeza(portador, 0, nh, dh)
    cols = w_v_columnas(portador, 0, nh, dh)
    assert torch.equal(filas.transpose(1, 2), cols)


def test_el_circuito_ov_exige_columnas(portador) -> None:
    """la factorización qr solo cierra con [h, d, dh]."""
    nh, dh = portador.n_cabezas, portador.dim_cabeza
    w_o = w_o_por_cabeza(portador, 0, nh, dh).double()
    r = torch.linalg.qr(w_v_columnas(portador, 0, nh, dh).double(),
                        mode="reduced")[1]
    assert (r @ w_o).shape == (nh, dh, w_o.shape[2])
    with pytest.raises(RuntimeError):
        mal = torch.linalg.qr(w_v_por_cabeza(portador, 0, nh, dh).double(),
                              mode="reduced")[1]
        _ = mal @ w_o
