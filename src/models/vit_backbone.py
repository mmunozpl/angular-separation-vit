"""ViT-B/16 desde timm con ganchos sobre la proyección por cabeza."""

import contextlib

import timm
import torch
from torch import nn


def robust_svd(
    stack: torch.Tensor, full_matrices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SVD batched con escalada ante matrices mal condicionadas.

    Con λ=0 (o muy pequeño) W_O se vuelve mal condicionado tras varias
    épocas y la SVD batched en CUDA con gesdd (default) aborta con error
    64. Escalada: gesdd CUDA → gesvd CUDA (Jacobi, robusto a singulares
    casi repetidos) → CPU float64. Las ramas que no son la caliente se
    loguean para observabilidad de qué cabezas degeneran en producción.

    Args:
        stack: tensor (..., m, n) con las matrices a descomponer.
        full_matrices: igual que en `torch.linalg.svd`.

    Returns:
        (u, s, vh) en el dtype y device originales de `stack`.

    Raises:
        RuntimeError: si `stack` contiene NaN/Inf antes de la SVD.
    """
    if not torch.isfinite(stack).all():
        raise RuntimeError("robust_svd: la entrada contiene NaN/Inf")
    try:
        return torch.linalg.svd(stack, full_matrices=full_matrices)
    except torch._C._LinAlgError as e_gesdd:
        print(f"[svd-fallback] gesdd CUDA falló: {e_gesdd}", flush=True)
        try:
            if stack.is_cuda:
                out = torch.linalg.svd(
                    stack, full_matrices=full_matrices, driver="gesvd",
                )
                print("[svd-fallback] gesvd CUDA recuperó "
                      "(rama intermedia)", flush=True)
                return out
            raise torch._C._LinAlgError       # salta a CPU float64
        except torch._C._LinAlgError as e_gesvd:
            if stack.is_cuda:
                print(f"[svd-fallback] gesvd CUDA falló: {e_gesvd}",
                      flush=True)
            stack_cpu = stack.detach().cpu().double()
            u_c, s_c, vh_c = torch.linalg.svd(
                stack_cpu, full_matrices=full_matrices,
            )
            print("[svd-fallback] CPU float64 recuperó (rama última)",
                  flush=True)
            return (
                u_c.to(stack.dtype).to(stack.device),
                s_c.to(stack.dtype).to(stack.device),
                vh_c.to(stack.dtype).to(stack.device),
            )


class HeadProjections(nn.Module):
    """Envuelve un ViT y expone direcciones representativas de W_O.

    Para cada bloque del ViT se considera la matriz ``attn.proj.weight``
    como la concatenación por cabezas de la salida W_O; la dirección
    representativa de cada cabeza es ``v_1(W_O^(h))``, su primer vector
    singular derecho ---la dirección dominante de escritura en el
    residuo, en R^768---, calculada por SVD y canonizada de signo.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 1000,
        img_size: int | None = None,
    ) -> None:
        """Inicializa la columna y cachea metadatos de cabezas.

        Args:
            model_name: identificador timm de la columna.
            pretrained: si carga pesos preentrenados de timm.
            num_classes: tamaño de la cabeza de clasificación; con 0
                la cabeza es identidad y ``forward`` devuelve features.
            img_size: resolución de entrada (lado en píxeles); si se
                pasa, timm interpola las posiciones para esa
                resolución. Necesario cuando se usa ``patch16_224``
                con imágenes a otra escala (256 px en Fase B).
        """
        super().__init__()
        create_kwargs: dict = {
            "pretrained": pretrained,
            "num_classes": num_classes,
        }
        if img_size is not None:
            create_kwargs["img_size"] = int(img_size)
        self.model = timm.create_model(model_name, **create_kwargs)
        first = self.model.blocks[0].attn
        self.num_heads = int(first.num_heads)
        self.embed_dim = int(self.model.embed_dim)
        self.head_dim = self.embed_dim // self.num_heads
        self.num_layers = len(self.model.blocks)

    def head_directions(self) -> torch.Tensor:
        """Primer vector singular derecho de W_O^(h), canonizado.

        ``r_h = v_1(W_O^(h))`` ---la dirección dominante de escritura en
        el residuo, en R^768---. Con ``proj.weight`` dispuesto como
        ``(d, h, hd)``, esa dirección es el vector singular izquierdo de
        la submatriz ``(d, hd)`` por cabeza, que equivale al singular
        derecho de ``W_O^(h)=(hd, d)``. El singular está definido salvo
        signo; aquí se canoniza de forma determinista para que el coseno
        con signo entre direcciones sea coherente entre llamadas y
        consistente con otras métricas que también usen coseno con signo.

        Convención: para cada vector, el componente de mayor valor
        absoluto se elige con signo positivo.

        Returns:
            Tensor (L, H, D) con direcciones unitarias y signo
            canónico.
        """
        d = self.embed_dim
        h = self.num_heads
        hd = self.head_dim
        stack = torch.stack([
            blk.attn.proj.weight.view(d, h, hd).permute(1, 0, 2)
            for blk in self.model.blocks
        ])  # (L, H, D, hd)
        # SVD robusto: con λ pequeño W_O degenera y gesdd CUDA aborta;
        # la cascada gesdd→gesvd→CPU float64 lo cubre (ver robust_svd).
        u, _, _ = robust_svd(stack, full_matrices=False)
        u_top = u[..., :, 0]  # (L, H, D)
        # canonización del signo: el componente de mayor |valor| positivo
        argmax_idx = u_top.abs().argmax(dim=-1, keepdim=True)
        pivot = torch.gather(u_top, dim=-1, index=argmax_idx)
        sign = torch.where(
            pivot >= 0,
            torch.ones_like(pivot),
            -torch.ones_like(pivot),
        )
        return (u_top * sign).contiguous()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pase de extracción de características (sin cabeza)."""
        return self.model.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pase adelante estándar (con cabeza)."""
        return self.model(x)


@contextlib.contextmanager
def capture_attention(model: nn.Module):
    """Captura los mapas de atención de cada bloque ViT timm.

    Se desactiva el camino fusionado (``fused_attn``) durante el bloque
    ``with`` para que la atención pase explícitamente por
    ``attn_drop`` y los ganchos puedan registrarla.

    Yields:
        Lista en la que se acumulan los mapas durante el bloque
        ``with``.
    """
    maps: list[torch.Tensor] = []
    hooks: list = []
    saved_fused: list[bool] = []

    def make_hook(idx: int):
        def hook(module, inputs, output):
            # inputs[0] es la atención justo tras softmax
            maps.append(inputs[0].detach())
        return hook

    for i, blk in enumerate(model.blocks):
        saved_fused.append(bool(getattr(blk.attn, "fused_attn", False)))
        if hasattr(blk.attn, "fused_attn"):
            blk.attn.fused_attn = False
        h = blk.attn.attn_drop.register_forward_hook(make_hook(i))
        hooks.append(h)
    try:
        yield maps
    finally:
        for h in hooks:
            h.remove()
        for i, blk in enumerate(model.blocks):
            if hasattr(blk.attn, "fused_attn"):
                blk.attn.fused_attn = saved_fused[i]
