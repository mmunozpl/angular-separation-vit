"""Descenso de gradiente proyectado sobre la esfera unidad."""

import torch
from tqdm import tqdm

from src.codes.riesz import riesz_energy
from src.codes.validate import min_angle_deg


def generate_code(
    k: int,
    d: int,
    s: float = 1.0,
    steps: int = 5000,
    lr: float = 0.05,
    log_every: int = 500,
    device: str = "cuda",
    seed: int | None = None,
    init: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[tuple[int, float, float]]]:
    """Genera un código esférico minimizando la energía de Riesz.

    Se usa Adam sobre la versión tangente del gradiente (se elimina la
    componente radial antes del paso) y se reproyecta a la esfera tras
    cada actualización. Esto suele converger antes que el descenso
    estocástico ingenuo en alta dimensión.

    Args:
        k: número de puntos del código.
        d: dimensión ambiente.
        s: exponente de la energía de Riesz.
        steps: número de pasos del descenso.
        lr: tasa de aprendizaje del optimizador.
        log_every: frecuencia (en pasos) del registro de progreso.
        device: dispositivo torch ('cuda' o 'cpu').
        seed: semilla local opcional para la inicialización aleatoria.
        init: tensor (k, d) opcional para arrancar desde una
            configuración concreta (p. ej. la canónica del kissing).
            Si se pasa, ignora ``seed``.

    Returns:
        Par formado por:
        - tensor (K, d) en la esfera unidad,
        - traza [(paso, energía, theta_min_grados)] del registro.
    """
    if init is not None:
        assert init.shape == (k, d), (
            f"init con forma {tuple(init.shape)} != ({k}, {d})"
        )
        x = init.to(device).clone()
    elif seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(k, d, generator=gen, device=device)
    else:
        x = torch.randn(k, d, device=device)
    # se normaliza la inicialización a la esfera unidad
    x = x / x.norm(dim=1, keepdim=True)
    x.requires_grad_(True)

    opt = torch.optim.Adam([x], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr * 1.0e-2,
    )
    trace: list[tuple[int, float, float]] = []

    pbar = tqdm(range(steps), desc=f"riesz K={k} d={d}", leave=False)
    for step in pbar:
        opt.zero_grad(set_to_none=True)
        e = riesz_energy(x, s=s)
        e.backward()
        # se proyecta el gradiente al espacio tangente a S^{d-1}
        with torch.no_grad():
            radial = (x.grad * x).sum(dim=1, keepdim=True)
            x.grad.sub_(radial * x)
        opt.step()
        sched.step()
        # se reproyecta cada fila a la esfera unidad
        with torch.no_grad():
            x.div_(x.norm(dim=1, keepdim=True))
        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                th = min_angle_deg(x.detach())
            trace.append((step, float(e.item()), th))
            pbar.set_postfix(
                E=f"{e.item():.3e}", theta=f"{th:.2f}",
            )
    return x.detach(), trace
