"""portador ligero del sector valor-salida para la demo.

`src/gauge_flip.py` opera sobre `modelo.blocks[capa].attn.qkv` y
`.proj`, es decir sobre un vit de timm de 984 mb. la demo no puede
cargar eso, pero tampoco debe reimplementar el gauge: reimplementarlo
sería perder la propiedad que hace fiable a esta ruta ---que la demo
ejecuta el mismo código que produjo las tablas 2 y 3---.

la salida es un objeto mínimo que expone exactamente esa interfaz
sobre los tensores del safetensors. `aplica_gauge_ov` corre **sin una
sola modificación**; solo toca las filas de valor del qkv, su sesgo y
las columnas de la proyección de salida, que es lo que el portador
tiene. el resto del qkv es relleno que la función jamás lee.
"""

import pathlib

import torch
from safetensors.torch import load_file


class _Lineal:
    """imita `nn.Linear` en los dos atributos que el gauge usa."""

    def __init__(self, weight: torch.Tensor,
                 bias: torch.Tensor | None = None) -> None:
        """guarda peso y sesgo como tensores mutables.

        Args:
            weight: matriz de pesos.
            bias: vector de sesgo, o None.
        """
        self.weight = weight
        self.bias = bias


class _Atencion:
    """imita el módulo de atención de timm: `.qkv` y `.proj`."""

    def __init__(self, qkv: _Lineal, proj: _Lineal) -> None:
        """agrupa las dos proyecciones.

        Args:
            qkv: proyección fusionada consulta-clave-valor.
            proj: proyección de salida.
        """
        self.qkv = qkv
        self.proj = proj


class _Bloque:
    """imita un bloque de timm: solo expone `.attn`."""

    def __init__(self, attn: _Atencion) -> None:
        """envuelve la atención.

        Args:
            attn: el módulo de atención del bloque.
        """
        self.attn = attn


class PortadorVO:
    """vit de mentira con el sector valor-salida de verdad.

    Attributes:
        blocks: lista de bloques con la interfaz que el gauge espera.
        n_cabezas: cabezas por capa.
        dim_cabeza: d_h.
    """

    def __init__(self, ruta: str) -> None:
        """monta el portador desde el safetensors del sector.

        los tensores se elevan a float64. `aplica_gauge_ov` calcula r y
        su inversa en doble y devuelve el resultado al dtype del peso:
        con el portador en simple, ese último casteo deja un error de
        ~1e-7 relativo y el circuito ov ---que es exactamente
        invariante--- aparece moviéndose milésimas de grado, más que
        v1(w_o) bajo un gauge ortogonal. en doble no se pierde nada y
        la invariancia se lee al orden que certifica la fase g.

        Args:
            ruta: fichero generado por `scripts/extraer_sector_vo.py`.
        """
        s = load_file(ruta)
        capas = 1 + max(int(k.split(".")[0][1:]) for k in s)
        self.n_cabezas = 1 + max(int(k.split(".")[1][1:]) for k in s)
        self.dim_cabeza = s["L0.h0.w_v"].shape[0]
        d = s["L0.h0.w_v"].shape[1]
        nh, dh = self.n_cabezas, self.dim_cabeza
        self.blocks = []
        for c in range(capas):
            qkv_w = torch.zeros(3 * d, d, dtype=torch.float64)
            qkv_b = torch.zeros(3 * d, dtype=torch.float64)
            proj_w = torch.zeros(d, d, dtype=torch.float64)
            for h in range(nh):
                fil = slice(2 * d + h * dh, 2 * d + (h + 1) * dh)
                col = slice(h * dh, (h + 1) * dh)
                qkv_w[fil, :] = s[f"L{c}.h{h}.w_v"].double()
                qkv_b[fil] = s[f"L{c}.h{h}.b_v"].double()
                proj_w[:, col] = (
                    s[f"L{c}.h{h}.w_o"].t().double())
            self.blocks.append(
                _Bloque(_Atencion(_Lineal(qkv_w, qkv_b), _Lineal(proj_w))))

    def w_o_por_cabeza(self, capa: int) -> torch.Tensor:
        """proyección de salida por cabeza, tras el gauge que haya.

        Args:
            capa: índice de capa.

        Returns:
            tensor [h, dh, d].
        """
        w = self.blocks[capa].attn.proj.weight
        dh = self.dim_cabeza
        return torch.stack([w[:, h * dh:(h + 1) * dh].t()
                            for h in range(self.n_cabezas)])

    def copia(self) -> "PortadorVO":
        """duplica el portador para aplicarle un gauge sin destruirlo.

        Returns:
            un portador independiente con los mismos pesos.
        """
        otro = PortadorVO.__new__(PortadorVO)
        otro.n_cabezas, otro.dim_cabeza = self.n_cabezas, self.dim_cabeza
        otro.blocks = []
        for b in self.blocks:
            q, p = b.attn.qkv, b.attn.proj
            otro.blocks.append(_Bloque(_Atencion(
                _Lineal(q.weight.clone(), q.bias.clone()),
                _Lineal(p.weight.clone()))))
        return otro


def carga(col: str = "vitb", raiz: str = "artifacts/demo") -> PortadorVO:
    """carga el portador de una columna.

    Args:
        col: etiqueta de columna (vitb o pythia).
        raiz: directorio de los safetensors.

    Returns:
        el portador montado.

    Raises:
        FileNotFoundError: si falta el fichero del sector.
    """
    ruta = pathlib.Path(raiz) / f"sector_vo_{col}.safetensors"
    if not ruta.exists():
        raise FileNotFoundError(
            f"{ruta}: ejecuta antes scripts/extraer_sector_vo.py")
    return PortadorVO(str(ruta))


def verifica_relleno(p: PortadorVO, capa: int = 0,
                     escala_id: float = 8.0) -> None:
    """comprueba que el gauge no toca las regiones de relleno.

    el portador funciona porque `aplica_gauge_ov` solo lee y escribe
    las filas de valor del qkv, su sesgo y las columnas de la
    proyección de salida. eso es un contrato implícito con `src/`: si
    una versión futura del gauge tocara consulta o clave, el portador
    serviría derivas de un modelo que ya no es el del paper, y lo
    haría en silencio. este assert convierte el contrato en fallo
    ruidoso.

    Args:
        p: portador ya montado.
        capa: capa sobre la que probar.
        escala_id: fuerza del gauge de prueba.

    Raises:
        AssertionError: si el gauge escribe fuera del sector de valor.
    """
    from src.gauge_flip import aplica_gauge_ov

    q = p.copia()
    w = q.blocks[capa].attn.qkv.weight
    d = w.shape[1]
    antes_w = w[:2 * d, :].clone()
    antes_b = q.blocks[capa].attn.qkv.bias[:2 * d].clone()
    aplica_gauge_ov(q, capa, q.n_cabezas, q.dim_cabeza, semilla=0,
                    escala_id=escala_id)
    assert torch.equal(q.blocks[capa].attn.qkv.weight[:2 * d, :], antes_w), (
        "el gauge escribió en las filas de consulta o clave: el portador "
        "ya no representa al modelo del paper")
    assert torch.equal(q.blocks[capa].attn.qkv.bias[:2 * d], antes_b), (
        "el gauge escribió en el sesgo de consulta o clave")


def verifica_manifiesto(raiz: str = ".") -> None:
    """contrasta el sha256 de lo vendido contra `manifiesto.sha256`.

    el Space lleva copias del código del paper, y las copias derivan.
    esto es el patrón de identidad byte a byte de g4 aplicado al
    código: si un fichero cambió, el arranque falla en vez de servir
    derivas de algo que ya no es el aparato publicado. si no hay
    manifiesto ---ejecución local desde el repo--- no hay nada que
    contrastar y la función no hace nada.

    Args:
        raiz: directorio donde vive `manifiesto.sha256`.

    Raises:
        AssertionError: si algún fichero falta o su hash no cuadra.
    """
    import hashlib

    m = pathlib.Path(raiz) / "manifiesto.sha256"
    if not m.exists():
        return
    for linea in m.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        esperado, rel = linea.split("  ", 1)
        f = pathlib.Path(raiz) / rel
        assert f.exists(), f"manifiesto: falta {rel}"
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        assert h.hexdigest() == esperado, (
            f"manifiesto: {rel} ha derivado respecto al commit firmado; "
            f"el Space no sirve un código que no es el del paper")
