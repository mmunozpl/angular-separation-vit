"""núcleo de lectura: svd exacta y decisión de poda, sin dependencias.

`firma_exacta` y `decisiones` las comparten el aparato del paper y la
demo. vivían en `scripts/run_fase_G.py` y `scripts/decision_rota.py`,
que arrastran timm y pandas por su cadena de imports; aquí solo hace
falta torch, de modo que el Space puede ejecutar **el mismo código**
sin cargar el entorno de investigación entero. los cuerpos son los que
produjeron las tablas 2 y 3, movidos sin tocar una línea.
"""

import torch

K_PODA = 3          # top-k de poda del ancla vit-b (25 % de 12 cabezas)


@torch.no_grad()
def firma_exacta(matriz: torch.Tensor, lado: str) -> torch.Tensor:
    """vector singular dominante por svd exacta (no iteración).

    el diagnóstico de invariancia exige svd exacta: medirla con
    iteración de potencia introduce ruido de init que la enmascara.

    Args:
        matriz: tensor [h, m, n] con h matrices.
        lado: 'izq' devuelve u[:,0] (en R^m); 'der' devuelve vh[0]
            (en R^n).

    Returns:
        tensor [h, k] con la dirección unitaria por cabeza.
    """
    out = []
    for h in range(matriz.shape[0]):
        u, _, vh = torch.linalg.svd(matriz[h], full_matrices=False)
        out.append(u[:, 0] if lado == "izq" else vh[0])
    return torch.stack(out)


def decisiones(firmas: torch.Tensor,
               k: int) -> tuple[tuple[int, int], list[int]]:
    """par más redundante y top-k de poda desde unas firmas.

    Args:
        firmas: tensor [h, d] de direcciones unitarias por cabeza.
        k: tamaño del conjunto de poda. **sin valor por
            defecto a propósito**: un default de módulo hizo
            que la columna de lenguaje calculara conjuntos de
            3 y dividiera por 4 durante dos versiones del
            paper. cada llamante declara el suyo.

    Returns:
        tupla (par_top ordenado, top-k de cabezas por redundancia
        media descendente).
    """
    c = (firmas @ firmas.t()).abs().clamp(max=1.0)
    c.fill_diagonal_(0.0)
    h = c.shape[0]
    idx = int(torch.argmax(c).item())
    par = tuple(sorted((idx // h, idx % h)))
    red_media = c.sum(dim=1) / (h - 1)
    topk = torch.argsort(red_media, descending=True)[:k]
    return par, [int(i) for i in topk]
