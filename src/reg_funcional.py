"""fase 1 — regularizador angular funcional localizado por profundidad.

el término r_func empuja las firmas hacia el símplex sin signo (parche,
inerte) o con signo (cls), sumado solo sobre las capas de la máscara. se
llama tras el forward, con la captura poblada, de modo que el gradiente
fluya por a_h, v_h y w_o —la contribución entera, gauge-invariante—.
"""

import torch

from src.firma_funcional import (CapturaContexto, w_o_por_cabeza,
                                 w_v_por_cabeza, w_qk_por_cabeza,
                                 v1_circuito, firma_parche, firma_cls)

# máscaras de localización por profundidad (definidas sobre ViT-B,
# 12 capas); la máscara completa se deriva de la profundidad del
# modelo en el bucle de entrenamiento, no de una constante
PERCEPTUAL = list(range(0, 6))    # capas 0..5, antes de la costura
PROFUNDO = list(range(6, 12))     # capas 6..11


def penaliza_firmas(
    firmas: torch.Tensor,
    n_cabezas: int,
    con_signo: bool,
) -> torch.Tensor:
    """bisagra angular de una capa hacia el símplex correspondiente.

    los portadores singulares (parche, inerte) usan |cos| hacia el símplex
    sin signo 1/(h-1); el portador cls, vector llano, usa coseno con signo
    hacia el símplex con signo -1/(h-1).

    args:
        firmas: tensor [h, d] con las h firmas unitarias.
        n_cabezas: cabezas h.
        con_signo: true para cls; false para parche/inerte.

    returns:
        escalar con la penalización media por pares de la capa.
    """
    cos = firmas @ firmas.t()                              # [h, h]
    fuera = ~torch.eye(n_cabezas, dtype=torch.bool, device=cos.device)
    if con_signo:
        # baja hacia el suelo con signo theta*_pm = arccos(-1/11)=95,22
        objetivo = -1.0 / (n_cabezas - 1)
        exceso = (cos - objetivo).clamp(min=0.0)
    else:
        # baja hacia el suelo sin signo theta* = arccos(1/11)=84,78
        objetivo = 1.0 / (n_cabezas - 1)
        exceso = (cos.abs() - objetivo).clamp(min=0.0)
    return exceso[fuera].sum() / (n_cabezas * (n_cabezas - 1))


def reg_funcional_localizado(
    captura: CapturaContexto,
    modelo,
    mascara: list[int],
    portador: str,
    n_cabezas: int = 12,
    dim_cabeza: int = 64,
    iteraciones: int = 8,
) -> torch.Tensor:
    """suma el regularizador funcional sobre las capas enmascaradas.

    se llama tras el forward; para 'parche'/'cls' el gradiente fluye por
    la captura (a_h, v_h, w_o), y para 'ov'/'qk' por los pesos del
    circuito identificable, sin necesitar la captura.

    args:
        captura: contexto a_h v_h por capa (solo lo usan parche/cls).
        modelo: el vit, para leer los pesos por capa.
        mascara: índices de capa donde se aplica (PERCEPTUAL,
            PROFUNDO o lista explícita; la completa la deriva el
            bucle de la profundidad del modelo).
        portador: 'ov', 'qk', 'parche' o 'cls'.
        n_cabezas: cabezas h.
        dim_cabeza: d_h.
        iteraciones: pasos de iteración de potencia.

    returns:
        escalar con el regularizador localizado de la pasada.
    """
    disp = next(modelo.parameters()).device
    total = torch.zeros((), device=disp)
    con_signo = portador == "cls"
    for idx in mascara:
        if portador == "ov":
            w_v = w_v_por_cabeza(modelo, idx, n_cabezas, dim_cabeza)
            w_o = w_o_por_cabeza(modelo, idx, n_cabezas, dim_cabeza)
            firmas = v1_circuito(w_v, w_o, iteraciones)
        elif portador == "qk":
            w_q, w_k = w_qk_por_cabeza(modelo, idx, n_cabezas, dim_cabeza)
            firmas = v1_circuito(w_q, w_k, iteraciones)
        elif portador == "parche":
            w_o = w_o_por_cabeza(modelo, idx, n_cabezas, dim_cabeza)
            firmas = firma_parche(captura.contexto[idx], w_o, iteraciones)
        elif portador == "cls":
            w_o = w_o_por_cabeza(modelo, idx, n_cabezas, dim_cabeza)
            firmas = firma_cls(captura.contexto[idx], w_o)
        else:
            raise ValueError(f"portador desconocido: {portador}")
        total = total + penaliza_firmas(firmas, n_cabezas, con_signo)
    return total
