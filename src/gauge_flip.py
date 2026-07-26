"""fase g — gauge-flip valor-salida demostrado, sin reentrenar.

aplica una transformación de gauge (en convención del paper v5:
w_v <- w_v r, w_o <- r^{-1} w_o, r en gl(d_h)) que deja la salida intacta
a precisión de máquina, y muestra que el ranking de redundancia por
v1(w_o) ---primer vector singular derecho de w_o, en r^768--- cambia
mientras la firma del circuito ov (w_v w_o) queda estable. convierte la
crítica de la premisa en una vulnerabilidad exhibida: leer la dirección
de w_o es leer ruido de gauge.
"""

import torch

from src.firma_funcional import w_o_por_cabeza, w_v_por_cabeza


@torch.no_grad()
def aplica_gauge_ov(
    modelo,
    capa: int,
    n_cabezas: int,
    dim_cabeza: int,
    semilla: int = 0,
    escala_id: float | None = None,
) -> float:
    """aplica un gauge valor-salida por cabeza, in situ.

    sustituye w_v <- r^t w_v, b_v <- r^t b_v y w_o <- w_o r^{-t} con r en
    gl(d_h) bien condicionada; deja la salida intacta a precisión de
    máquina y denota la libertad de gauge del sector valor-salida. el
    sesgo de valor entra en la transformación porque v = x w_v^t + b_v y
    la compensación de w_o exige que b_v gire igual que la weight; en una
    columna con qkv_bias omitirlo rompe la invariancia. clonar el modelo
    antes si se quiere conservar el original.

    args:
        modelo: el vit, modificado in situ.
        capa: índice de la capa intervenida.
        n_cabezas: cabezas h.
        dim_cabeza: d_h.
        semilla: semilla de la r aleatoria, para reproducir.
        escala_id: coeficiente de la identidad en r = randn + escala_id*i;
            menor valor aleja r de un múltiplo escalar de la identidad
            ---gauge más fuerte, v1(w_o) deriva más, el circuito ov no---.
            por defecto sqrt(d_h), el calibre de referencia.

    returns:
        desviación media de r respecto a su mejor múltiplo escalar de la
        identidad, sobre las cabezas; proxy de la fuerza del gauge.
    """
    if escala_id is None:
        escala_id = dim_cabeza ** 0.5
    g = torch.Generator(device="cpu").manual_seed(semilla)
    attn = modelo.blocks[capa].attn
    w_qkv = attn.qkv.weight                                # [3d, d]
    w_proj = attn.proj.weight                              # [d, d]
    base = 2 * w_qkv.shape[1]                              # inicio de v
    # r e inversa en float64: con r fuerte la inv en fp32 amplifica el
    # error por el número de condición y rompe la invariancia; en fp64 el
    # gauge es preciso sea cual sea la fuerza, y solo se pierde el último
    # casteo al dtype del peso.
    ident = torch.eye(dim_cabeza, device=w_qkv.device, dtype=torch.float64)
    desvs = []
    for h in range(n_cabezas):
        rt = torch.randn(dim_cabeza, dim_cabeza, generator=g,
                         dtype=torch.float64).to(w_qkv.device)
        r = rt + escala_id * ident
        # desviación de r respecto a su mejor múltiplo escalar de la id
        s = r.diagonal().mean()
        desvs.append(float((r - s * ident).norm()
                           / (s.abs() * dim_cabeza ** 0.5 + 1e-8)))
        r_inv_t = torch.linalg.inv(r).t()
        fil = slice(base + h * dim_cabeza, base + (h + 1) * dim_cabeza)
        col = slice(h * dim_cabeza, (h + 1) * dim_cabeza)
        w_qkv[fil, :] = (
            r.t() @ w_qkv[fil, :].double()).to(w_qkv.dtype)   # w_v
        if attn.qkv.bias is not None:
            attn.qkv.bias[fil] = (
                r.t() @ attn.qkv.bias[fil].double()).to(w_qkv.dtype)  # b_v
        w_proj[:, col] = (
            w_proj[:, col].double() @ r_inv_t).to(w_proj.dtype)  # w_o
    return float(sum(desvs) / len(desvs))


@torch.no_grad()
def cos_pares_v1_wo(
    modelo,
    capa: int,
    n_cabezas: int,
    dim_cabeza: int,
) -> torch.Tensor:
    """|cos| entre las direcciones v1(w_o) de las cabezas de una capa.

    v1(w_o) es el primer vector singular derecho de w_o (en r^768), la
    cantidad que la poda o interpretación por dirección de salida lee; el
    experimento muestra que cambia bajo el gauge.

    args:
        modelo: el vit.
        capa: índice de la capa.
        n_cabezas: cabezas h.
        dim_cabeza: d_h.

    returns:
        tensor [h, h] con |cos| entre direcciones dominantes de w_o.
    """
    w_o = w_o_por_cabeza(modelo, capa, n_cabezas, dim_cabeza)  # [h,dh,d]
    u = torch.stack([
        torch.linalg.svd(w_o[h], full_matrices=False).Vh[0]
        for h in range(n_cabezas)])                           # [h, d]
    return (u @ u.t()).abs()


@torch.no_grad()
def cos_pares_circuito_ov(
    modelo,
    capa: int,
    n_cabezas: int,
    dim_cabeza: int,
) -> torch.Tensor:
    """|cos| entre las firmas invariantes del circuito ov por cabeza.

    contrasta con cos_pares_v1_wo: no cambia bajo el gauge. el
    diagnóstico usa svd exacta del circuito ov (w_v w_o) ---no la
    iteración de potencia del regularizador---, de modo que la
    invariancia se ve a precisión de máquina y no arrastra el ruido de
    init de la potencia.

    args:
        modelo: el vit.
        capa: índice de la capa.
        n_cabezas: cabezas h.
        dim_cabeza: d_h.

    returns:
        tensor [h, h] con |cos| entre firmas del circuito ov.
    """
    w_v = w_v_por_cabeza(modelo, capa, n_cabezas, dim_cabeza)  # [h,dh,d]
    w_o = w_o_por_cabeza(modelo, capa, n_cabezas, dim_cabeza)  # [h,dh,d]
    # circuito ov por cabeza m = w_v w_o (rango <= dh, invariante de
    # gauge); su primer vector singular es la firma invariante. el einsum
    # contrae el eje de valor d_h compartido (w_v es [dh,d]=W_v^T)
    circ = torch.einsum("hed,hef->hdf", w_v, w_o)             # [h, d, d]
    r = torch.stack([
        torch.linalg.svd(circ[h], full_matrices=False).U[:, 0]
        for h in range(n_cabezas)])                           # [h, d]
    return (r @ r.t()).abs()
