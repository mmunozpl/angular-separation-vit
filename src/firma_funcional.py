"""núcleo compartido del portador funcional (B), gauge-invariante.

provee la captura del contexto a_h v_h por capa, la lectura de w_o por
cabeza y las firmas de respuesta —parche y cls— como primer vector
singular derecho de la contribución al residuo c_h^p = a_h v_h w_o.
lo importan la premisa (gauge-flip, computado/estático) y la sonda del
regularizador del paper v5.
"""

import torch
from tqdm import tqdm


class CapturaContexto:
    """captura la entrada a proj (contexto a_h v_h) por capa.

    registra un forward-pre-hook en cada proj del encoder y guarda la
    entrada sin desconectarla del grafo, de modo que el gradiente del
    regularizador fluya por a_h, v_h y w_o. la entrada a proj es ya el
    contexto concatenado, así que no hace falta materializar la atención.
    """

    def __init__(self, modelo, n_cabezas: int, dim_cabeza: int):
        self.h = n_cabezas
        self.dh = dim_cabeza
        self.contexto: dict[int, torch.Tensor] = {}
        self.handles = []
        for idx, bloque in enumerate(modelo.blocks):
            self.handles.append(
                bloque.attn.proj.register_forward_pre_hook(
                    self._gancho(idx)))

    def _gancho(self, idx: int):
        # se reordena la entrada (b, n, c) a (b, n, h, dh)
        def hook(modulo, entrada):
            x = entrada[0]
            b, n, _ = x.shape
            self.contexto[idx] = x.view(b, n, self.h, self.dh)
        return hook

    def limpiar(self) -> None:
        self.contexto = {}

    def quitar(self) -> None:
        for handle in self.handles:
            handle.remove()


def w_o_por_cabeza(
    modelo,
    capa: int,
    n_cabezas: int,
    dim_cabeza: int,
) -> torch.Tensor:
    """proyección de salida por cabeza w_o^(h) de una capa.

    args:
        modelo: el vit.
        capa: índice de la capa.
        n_cabezas: cabezas h.
        dim_cabeza: d_h.

    returns:
        tensor [h, dh, d] con w_o^(h) por cabeza.
    """
    w = modelo.blocks[capa].attn.proj.weight  # [d, d]
    return torch.stack([
        w[:, h * dim_cabeza:(h + 1) * dim_cabeza].t()
        for h in range(n_cabezas)])


def v1_desde_gram(
    gram: torch.Tensor,
    w_o: torch.Tensor,
    iteraciones: int = 8,
) -> torch.Tensor:
    """primer vector singular derecho de g @ w_o dado su gram en d_h.

    itera sobre el operador implícito (g w_o)^t (g w_o) sin materializar
    la matriz d x d; el gram en el espacio de valor d_h ya viene sumado.

    args:
        gram: tensor [h, dh, dh] con g^t g por cabeza.
        w_o: tensor [h, dh, d] con la proyección de salida por cabeza.
        iteraciones: pasos de iteración de potencia.

    returns:
        tensor [h, d] con la firma unitaria por cabeza, en el residuo.
    """
    h = gram.shape[0]
    d = w_o.shape[-1]
    v = torch.randn(h, d, device=gram.device, dtype=gram.dtype)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    for _ in range(iteraciones):
        a = torch.einsum("hed,hd->he", w_o, v)   # w_o v -> [h, dh]
        a = torch.einsum("hef,hf->he", gram, a)  # gram a -> [h, dh]
        v = torch.einsum("hed,he->hd", w_o, a)   # w_o^t a -> [h, d]
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return v


def firma_parche(
    contexto: torch.Tensor,
    w_o: torch.Tensor,
    iteraciones: int = 8,
) -> torch.Tensor:
    """firma de respuesta por parche, apilando el lote en el gram.

    args:
        contexto: tensor [b, n, h, dh] de la captura (n incluye cls).
        w_o: tensor [h, dh, d].
        iteraciones: pasos de iteración de potencia.

    returns:
        tensor [h, d] con la firma unitaria por cabeza.
    """
    # se descarta cls y se apila el lote en el eje de parches
    g = contexto[:, 1:, :, :].permute(2, 0, 1, 3)        # [h, b, p, dh]
    g = g.reshape(g.shape[0], -1, g.shape[-1])           # [h, b*p, dh]
    gram = torch.einsum("hnd,hne->hde", g, g)            # [h, dh, dh]
    return v1_desde_gram(gram, w_o, iteraciones)


def firma_cls(
    contexto: torch.Tensor,
    w_o: torch.Tensor,
) -> torch.Tensor:
    """escritura media de la cabeza al token cls, en el residuo.

    args:
        contexto: tensor [b, n, h, dh] de la captura.
        w_o: tensor [h, dh, d].

    returns:
        tensor [h, d] con la firma unitaria por cabeza (vector llano).
    """
    cls = contexto[:, 0, :, :]                            # [b, h, dh]
    contrib = torch.einsum("bhe,hed->bhd", cls, w_o)      # [b, h, d]
    firmas = contrib.mean(dim=0)                          # [h, d]
    return firmas / firmas.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def w_v_por_cabeza(
    modelo,
    capa: int,
    n_cabezas: int,
    dim_cabeza: int,
) -> torch.Tensor:
    """filas de valor w_v^(h) de una capa, desde el qkv fusionado.

    en timm el qkv concatena [q, k, v]; v ocupa el último tercio y se
    reordena por cabeza.

    args:
        modelo: el vit.
        capa: índice de la capa.
        n_cabezas: cabezas h.
        dim_cabeza: d_h.

    returns:
        tensor [h, dh, d] con w_v^(h) por cabeza.
    """
    w = modelo.blocks[capa].attn.qkv.weight                # [3d, d]
    base = 2 * w.shape[1]                                   # inicio de v
    return torch.stack([
        w[base + h * dim_cabeza:base + (h + 1) * dim_cabeza, :]
        for h in range(n_cabezas)])


def w_qk_por_cabeza(
    modelo,
    capa: int,
    n_cabezas: int,
    dim_cabeza: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """filas de consulta y clave w_q^(h), w_k^(h) de una capa.

    args:
        modelo: el vit.
        capa: índice de la capa.
        n_cabezas: cabezas h.
        dim_cabeza: d_h.

    returns:
        tupla (w_q [h, dh, d], w_k [h, dh, d]).
    """
    w = modelo.blocks[capa].attn.qkv.weight                # [3d, d]
    d = w.shape[1]
    w_q = torch.stack([
        w[h * dim_cabeza:(h + 1) * dim_cabeza, :]
        for h in range(n_cabezas)])
    w_k = torch.stack([
        w[d + h * dim_cabeza:d + (h + 1) * dim_cabeza, :]
        for h in range(n_cabezas)])
    return w_q, w_k


def v1_circuito(
    izq: torch.Tensor,
    der: torch.Tensor,
    iteraciones: int = 8,
) -> torch.Tensor:
    """primer vector singular del circuito compuesto izq·der.

    sirve para el circuito ov (w_v w_o) y el qk (w_q w_k^t); con los
    factores en convención de código (izq=[dh,d]=W^T) la composición
    materializa m=izq^t der = el circuito d x d, rango <= d_h e
    invariante de gauge. se itera sobre m m^t sin materializar la matriz
    d x d, vía los factores en d_h.

    args:
        izq: tensor [h, dh, d], el factor izquierdo.
        der: tensor [h, dh, d], el factor derecho.
        iteraciones: pasos de iteración de potencia.

    returns:
        tensor [h, d] con la firma unitaria por cabeza, invariante.
    """
    h = izq.shape[0]
    d = izq.shape[-1]
    v = torch.randn(h, d, device=izq.device, dtype=izq.dtype)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    # m = izq^t der ; m m^t = izq^t der der^t izq
    for _ in range(iteraciones):
        a = torch.einsum("hed,hd->he", izq, v)   # izq v -> [h, dh]
        b = torch.einsum("hed,he->hd", der, a)   # der^t a -> [h, d]
        c = torch.einsum("hed,hd->he", der, b)   # der b -> [h, dh]
        v = torch.einsum("hed,he->hd", izq, c)   # izq^t c -> [h, d]
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return v


@torch.no_grad()
def geometria_circuito_exacta(
    modelo,
    portador: str,
    n_cabezas: int = 12,
    dim_cabeza: int = 64,
) -> tuple[float, float]:
    """theta_min medio y redundancia media del circuito, por svd exacta.

    lectura para juzgar la separación del portador de circuito ---no la
    iteración de potencia del regularizador, que arrastra ruido de init---:
    por capa materializa el circuito (ov = w_v w_o, qk = w_q w_k^t), toma
    su primer vector singular exacto, y agrega entre capas el
    theta_min (peor par, sin signo) y la redundancia media (|cos| medio
    fuera de la diagonal). la redundancia acompaña al theta_min porque el
    peor par puede despegarse sin que el conjunto se separe.

    args:
        modelo: el vit (timm) que expone .blocks.
        portador: 'ov' o 'qk'.
        n_cabezas: cabezas h.
        dim_cabeza: d_h.

    returns:
        tupla (theta_min medio en grados, redundancia media).
    """
    n_capas = len(modelo.blocks)
    thetas, redund = [], []
    for l in range(n_capas):
        if portador == "ov":
            izq = w_v_por_cabeza(modelo, l, n_cabezas, dim_cabeza)
            der = w_o_por_cabeza(modelo, l, n_cabezas, dim_cabeza)
        else:
            izq, der = w_qk_por_cabeza(modelo, l, n_cabezas, dim_cabeza)
        circ = torch.einsum("hed,hef->hdf", izq, der)        # [h, d, d]
        r = torch.stack([
            torch.linalg.svd(circ[h], full_matrices=False).U[:, 0]
            for h in range(n_cabezas)])                       # [h, d]
        fuera = ~torch.eye(n_cabezas, dtype=torch.bool, device=r.device)
        cos = (r @ r.t()).abs().clamp(max=1.0)
        thetas.append(float(torch.rad2deg(
            torch.arccos(cos[fuera].max()))))
        redund.append(float(cos[fuera].mean()))
    return float(sum(thetas) / n_capas), float(sum(redund) / n_capas)


def contribuciones_por_cabeza(
    contexto: torch.Tensor,
    w_o: torch.Tensor,
) -> torch.Tensor:
    """contribución al residuo a_h v_h w_o por cabeza.

    args:
        contexto: tensor [b, n, h, dh] de CapturaContexto.
        w_o: tensor [h, dh, d].

    returns:
        tensor [h, b, n, d] con la contribución de cada cabeza al residuo.
    """
    return torch.einsum("bnhe,hed->hbnd", contexto, w_o)


def _cka_lineal(x: torch.Tensor, y: torch.Tensor) -> float:
    """cka lineal entre dos representaciones [n, p] y [n, q], centradas.

    huella no circular de diversidad de output (no usa el primer vector
    singular que el regularizador separa): 1 = idénticas, 0 = ortogonales.

    args:
        x: tensor [n, p].
        y: tensor [n, q].

    returns:
        cka lineal en [0, 1].
    """
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    num = (x.t() @ y).norm() ** 2
    den = (x.t() @ x).norm() * (y.t() @ y).norm()
    return float(num / den.clamp_min(1e-12))


@torch.no_grad()
def diversidad_output_cka(
    modelo,
    cargador,
    n_cabezas: int = 12,
    dim_cabeza: int = 64,
) -> tuple[float, list[float]]:
    """redundancia de output por capa: cka lineal entre cabezas sobre el
    flujo completo de contribuciones, sin promediar parches ni muestras.

    cada par (imagen, parche) es una muestra; para el par (h, h') se mide
    el cka lineal entre $c_h, c_{h'}\\in R^{n\\times d}$ ---la contribución
    por muestra, no su media---. es la huella que el regularizador OV no
    optimiza directamente (dos cabezas pueden compartir su primer vector
    singular y tener cka bajo si se encienden en muestras distintas), y
    por eso el pago no circular. se acumula el gram cruzado por capa para
    no materializar la matriz n x d completa. complementa s_func, que ve
    solo el lado q.k. el suelo del cka lineal escala con d/n, así que se
    reporta como delta contra el brazo base sobre el mismo conjunto.

    args:
        modelo: vit (timm) en eval.
        cargador: loader del conjunto congelado (mismas imágenes y orden
            para todos los brazos, para que el suelo común se reste).
        n_cabezas: cabezas h.
        dim_cabeza: d_h.

    returns:
        tupla (cka media entre capas, lista de cka por capa).
    """
    modelo.eval()
    disp = next(modelo.parameters()).device
    captura = CapturaContexto(modelo, n_cabezas, dim_cabeza)
    n_capas = len(modelo.blocks)
    d = w_o_por_cabeza(modelo, 0, n_cabezas, dim_cabeza).shape[-1]
    pares = [(i, j) for i in range(n_cabezas)
             for j in range(i, n_cabezas)]
    idx = {ij: k for k, ij in enumerate(pares)}
    gram = torch.zeros(n_capas, len(pares), d, d, device=disp)
    suma = torch.zeros(n_capas, n_cabezas, d, device=disp)
    total = 0
    for imgs, _ in tqdm(cargador, desc="cka output"):
        captura.limpiar()
        _ = modelo(imgs.to(disp))
        muestras = 0
        for l in range(n_capas):
            w_o = w_o_por_cabeza(modelo, l, n_cabezas, dim_cabeza)
            c = contribuciones_por_cabeza(captura.contexto[l], w_o)
            # se apilan (imagen, parche) como muestras, sin cls ni medias
            c = c[:, :, 1:, :].reshape(n_cabezas, -1, d)   # [h, b*p, d]
            muestras = c.shape[1]
            suma[l] = suma[l] + c.sum(dim=1)
            for (i, j) in pares:
                gram[l, idx[(i, j)]] += c[i].t() @ c[j]
        total += muestras
    captura.quitar()

    fuera = [(i, j) for i in range(n_cabezas)
             for j in range(i + 1, n_cabezas)]
    por_capa = []
    for l in range(n_capas):
        # gram centrado: gc_ij = g_ij - (sum c_i)(sum c_j)^t / n
        nrm = [
            (gram[l, idx[(i, i)]]
             - torch.outer(suma[l, i], suma[l, i]) / total).norm()
            for i in range(n_cabezas)]
        vals = []
        for (i, j) in fuera:
            gc = (gram[l, idx[(i, j)]]
                  - torch.outer(suma[l, i], suma[l, j]) / total)
            vals.append(float(
                gc.norm() ** 2 / (nrm[i] * nrm[j]).clamp_min(1e-12)))
        por_capa.append(sum(vals) / len(vals))
    return float(sum(por_capa) / n_capas), por_capa
