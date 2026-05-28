"""Contribución A: ViT con regularización angular sobre cabezas."""

import torch
from torch import nn

from src.models.vit_backbone import HeadProjections


def _rotate_align_columns(
    w_sub: torch.Tensor,
    u1: torch.Tensor,
    c: torch.Tensor,
    eps: float = 1.0e-9,
) -> torch.Tensor:
    """Rotación ortogonal en R^D que lleva ``u1`` a ``c``, aplicada
    a las columnas de ``w_sub``.

    Caso no batched (D, hd). Para uso batched ver
    ``_rotate_align_columns_batched``.
    """
    dot = (u1 * c).sum().clamp(-1.0, 1.0)
    if dot.item() > 1.0 - eps:
        return w_sub
    e1 = u1
    e2 = c - dot * u1
    e2_norm = e2.norm()
    if e2_norm < eps:
        rand = torch.randn_like(e1)
        e2 = rand - (rand * e1).sum() * e1
        e2 = e2 / e2.norm().clamp_min(eps)
        cos_th = w_sub.new_tensor(-1.0)
        sin_th = w_sub.new_tensor(0.0)
    else:
        e2 = e2 / e2_norm
        cos_th = dot
        sin_th = (1.0 - dot * dot).clamp_min(0.0).sqrt()
    a1 = e1 @ w_sub
    a2 = e2 @ w_sub
    delta = (
        sin_th * (torch.outer(e2, a1) - torch.outer(e1, a2))
        + (cos_th - 1.0) * (torch.outer(e1, a1) + torch.outer(e2, a2))
    )
    return w_sub + delta


def _rotate_align_columns_batched(
    w_sub: torch.Tensor,
    u1: torch.Tensor,
    c: torch.Tensor,
    eps: float = 1.0e-9,
) -> torch.Tensor:
    """Versión batched de la rotación que alinea ``u1[h]`` con ``c[h]``.

    Aplica una rotación ortogonal en R^D por cabeza, todo en una
    pasada sin loop Python. Reduce el coste de
    ``_apply_hard_code`` ~H veces frente al loop por cabeza.

    Args:
        w_sub: tensor (H, D, hd) — W_O por cabeza.
        u1: tensor (H, D) — primer vector singular por cabeza.
        c: tensor (H, D) — direcciones objetivo (signo ya elegido).
        eps: tolerancia numérica.

    Returns:
        Tensor (H, D, hd) con las columnas rotadas.
    """
    dot = (u1 * c).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)  # (H, 1)
    e1 = u1                                                    # (H, D)
    e2 = c - dot * u1                                          # (H, D)
    e2_norm = e2.norm(dim=-1, keepdim=True)                    # (H, 1)
    # máscara: 1 si rotación necesaria, 0 si ya alineado
    needs = (e2_norm > eps).to(w_sub.dtype)                    # (H, 1)
    e2 = e2 / e2_norm.clamp_min(eps)
    cos_th = dot
    sin_th = (1.0 - dot * dot).clamp_min(0.0).sqrt()
    # proyecciones sobre el plano (e1, e2): (H, hd)
    a1 = torch.einsum("hd,hde->he", e1, w_sub)
    a2 = torch.einsum("hd,hde->he", e2, w_sub)
    # productos exteriores batched: (H, D, hd)
    out_e2_a1 = e2.unsqueeze(-1) * a1.unsqueeze(-2)
    out_e1_a2 = e1.unsqueeze(-1) * a2.unsqueeze(-2)
    out_e1_a1 = e1.unsqueeze(-1) * a1.unsqueeze(-2)
    out_e2_a2 = e2.unsqueeze(-1) * a2.unsqueeze(-2)
    delta = (
        sin_th.unsqueeze(-1) * (out_e2_a1 - out_e1_a2)
        + (cos_th - 1.0).unsqueeze(-1)
        * (out_e1_a1 + out_e2_a2)
    )
    delta = delta * needs.unsqueeze(-1)
    return w_sub + delta


class AttnDiverseViT(nn.Module):
    """ViT-B/16 con regulador R_div y modo duro de fijación al código.

    En modo blando, ``head_directions`` se usa para calcular ``R_div``
    durante el entrenamiento. En modo duro, las medias por cabeza se
    inicializan al código esférico y se **reproyectan tras cada paso
    del optimizador** (paper §486) llamando a ``reproject_to_code()``;
    el bucle de entrenamiento la invoca después de
    ``optimizer.step()``.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 1000,
        hard_code: torch.Tensor | None = None,
        img_size: int | None = None,
    ) -> None:
        """Inicializa la columna.

        Args:
            model_name: identificador timm.
            pretrained: si carga pesos preentrenados.
            num_classes: clases de salida.
            hard_code: tensor (L, H, D) opcional para fijar direcciones
                en modo duro. Se cachea como buffer para reproyectar.
            img_size: resolución de entrada en píxeles; se propaga a
                timm para que interpole las posiciones cuando se usa
                ``patch16_224`` a otra escala.
        """
        super().__init__()
        self.backbone = HeadProjections(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size,
        )
        self.hard_code_applied = False
        if hard_code is not None:
            # se cachea el código para reproyectar tras cada step
            self.register_buffer(
                "_hard_code", hard_code.clone().detach(),
            )
            self._apply_hard_code(hard_code)
            self.hard_code_applied = True

    def _apply_hard_code(self, code: torch.Tensor) -> None:
        """Alinea el primer vector singular de W_O^(h) con el código.

        **Full-batched sobre las L×H = 144 cabezas a la vez**:
        apila las 144 matrices ``W_O^(h)`` en un único tensor de
        forma ``(L*H, D, hd)``, hace una sola llamada a
        ``torch.linalg.svd``, una sola elección de signos y una sola
        rotación ortogonal batched. Equivalente bit-a-bit al loop
        por cabeza, solo más rápido.

        ``W_O^(h) <- Q_h @ W_O^(h)``, donde ``Q_h`` rota en el
        plano que generan ``u_1[h]`` y ``c_h`` y preserva el resto
        del espectro.
        """
        m = self.backbone.model
        L = len(m.blocks)
        d = self.backbone.embed_dim
        h = self.backbone.num_heads
        hd = self.backbone.head_dim
        assert code.shape == (L, h, d), (
            f"código {tuple(code.shape)} incompatible con "
            f"(L={L}, H={h}, D={d})"
        )
        with torch.no_grad():
            # apila las 144 W_O^(h) en (L, H, D, hd) y luego (L*H, D, hd)
            wo_stack = torch.stack([
                blk.attn.proj.weight.view(d, h, hd).permute(1, 0, 2)
                for blk in m.blocks
            ]).contiguous()  # (L, H, D, hd)
            wo_flat = wo_stack.reshape(L * h, d, hd)  # (LH, D, hd)
            # SVD batched de las 144 matrices en una sola llamada
            u, _, _ = torch.linalg.svd(
                wo_flat, full_matrices=False,
            )
            u1 = u[:, :, 0]  # (LH, D)
            # código aplanado a (LH, D) en mismo device/dtype
            c_flat = code.reshape(L * h, d).to(
                u1.device, dtype=u1.dtype,
            )
            # signo por cabeza que minimiza la rotación
            dot = (u1 * c_flat).sum(dim=-1, keepdim=True)
            c_signed = torch.where(dot < 0, -c_flat, c_flat)
            # rotación batched de las 144 cabezas a la vez
            new_wo_flat = _rotate_align_columns_batched(
                wo_flat, u1, c_signed,
            )
            # reescribe en cada capa
            new_wo_stack = new_wo_flat.reshape(L, h, d, hd)
            for li, blk in enumerate(m.blocks):
                blk.attn.proj.weight.view(d, h, hd).copy_(
                    new_wo_stack[li].permute(1, 0, 2)
                )

    def reproject_to_code(self) -> None:
        """Reproyecta W_O sobre el código cacheado (modo duro).

        Se invoca tras ``optimizer.step()`` para mantener las
        direcciones representativas alineadas al código esférico
        independientemente del paso de gradiente. Es no-op en modo
        blando.
        """
        if self.hard_code_applied:
            self._apply_hard_code(self._hard_code)

    def head_directions(self) -> torch.Tensor:
        """Direcciones representativas por cabeza y capa."""
        return self.backbone.head_directions()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pase de extracción de características (sin cabeza)."""
        return self.backbone.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pase adelante estándar."""
        return self.backbone(x)
