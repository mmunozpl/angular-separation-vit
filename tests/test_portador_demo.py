"""el portador de la demo respeta el contrato con src/gauge_flip."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.portador import carga, verifica_relleno  # noqa: E402

SECTOR = "artifacts/demo/sector_vo_vitb.safetensors"


@pytest.mark.skipif(not os.path.exists(SECTOR),
                    reason="falta el sector; correr extraer_sector_vo.py")
def test_gauge_no_toca_el_relleno() -> None:
    """consulta y clave quedan intactas tras aplicar el gauge."""
    verifica_relleno(carga("vitb"), capa=0)


@pytest.mark.skipif(not os.path.exists(SECTOR),
                    reason="falta el sector; correr extraer_sector_vo.py")
def test_el_gauge_si_mueve_el_sector_de_valor() -> None:
    """control positivo: el valor y la salida sí cambian."""
    from src.gauge_flip import aplica_gauge_ov

    p = carga("vitb")
    q = p.copia()
    d = q.blocks[0].attn.qkv.weight.shape[1]
    antes_v = q.blocks[0].attn.qkv.weight[2 * d:, :].clone()
    antes_o = q.blocks[0].attn.proj.weight.clone()
    aplica_gauge_ov(q, 0, q.n_cabezas, q.dim_cabeza, semilla=0,
                    escala_id=8.0)
    assert not torch.equal(q.blocks[0].attn.qkv.weight[2 * d:, :], antes_v)
    assert not torch.equal(q.blocks[0].attn.proj.weight, antes_o)
