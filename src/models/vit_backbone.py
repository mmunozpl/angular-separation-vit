"""ViT-B/16 desde timm con ganchos sobre la proyección por cabeza."""

import contextlib

import timm
import torch
from torch import nn


class HeadProjections(nn.Module):
    """Envuelve un ViT y expone direcciones representativas de W_O.

    Para cada bloque del ViT se considera la matriz ``attn.proj.weight``
    como la concatenación por cabezas de la salida W_O; la dirección
    representativa de cada cabeza es la media de sus columnas,
    normalizada.
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
        """Primer vector singular izquierdo de W_O^(h), canonizado.

        Paper §628-643 eq.~3: ``r_h = u_1(W_O^(h))``. El primer
        vector singular está definido salvo signo; aquí se canoniza
        de forma determinista para que el coseno con signo entre
        direcciones sea coherente entre llamadas y consistente con
        otras métricas que también usen coseno con signo.

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
        u, _, _ = torch.linalg.svd(stack, full_matrices=False)
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
