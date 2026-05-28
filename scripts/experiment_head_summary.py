"""Experimento: qué resumen de W_O capta la diversidad funcional.

Sobre el ViT-B/16 afinado (pretrained timm de fábrica o checkpoint
opcional), para cada par de cabezas en cada capa calcula:

- s_attn: similitud entre los mapas de atención promediados sobre
  un lote de imágenes de validación (coseno de los mapas aplanados);
- s_A:    coseno entre direcciones representativas con resumen A
  = media de columnas de W_O (eq. 3 del paper);
- s_B:    coseno entre direcciones representativas con resumen B
  = primer vector singular izquierdo de W_O^(h).

Reporta la correlación de Pearson entre s_attn y cada uno, por capa
y agregado. Reporta también 15 pares aleatorios con las tres
similitudes para inspección directa. No entrena nada.
"""

import argparse
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.vit_backbone import HeadProjections, capture_attention


def head_dirs_mean(model: HeadProjections) -> torch.Tensor:
    """Direcciones por la media de columnas de W_O. (L, H, D)."""
    return model.head_directions()


def head_dirs_svd(model: HeadProjections) -> torch.Tensor:
    """Direcciones por el primer vector singular izquierdo de W_O^(h).

    Para cada capa l y cabeza h, toma la submatriz W_O[:, h*hd:(h+1)*hd]
    de tamaño (D, hd) y calcula su primer left singular vector como
    representante.
    """
    out = []
    bb = model
    d, h, hd = bb.embed_dim, bb.num_heads, bb.head_dim
    for blk in bb.model.blocks:
        wo = blk.attn.proj.weight.view(d, h, hd)  # (D, H, hd)
        per_layer = []
        for hi in range(h):
            u, _, _ = torch.linalg.svd(
                wo[:, hi, :], full_matrices=False,
            )
            per_layer.append(u[:, 0])
        v = torch.stack(per_layer)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        out.append(v)
    return torch.stack(out)


def _pair_idx(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Índices de los pares ordenados h < h'."""
    return torch.triu_indices(n, n, offset=1)


def attn_similarity(maps: list[torch.Tensor]) -> torch.Tensor:
    """Coseno entre mapas de atención por par de cabezas y capa.

    Args:
        maps: lista de L tensores (B, H, T, T).

    Returns:
        Tensor (L, n_pares) con cos en [-1, 1]. Los mapas son
        distribuciones (filas sum=1, valores ≥ 0), así que en la
        práctica los cosenos son no negativos.
    """
    L = len(maps)
    H = maps[0].shape[1]
    idx_i, idx_j = _pair_idx(H)
    out = torch.zeros(L, idx_i.numel())
    for li, m in enumerate(maps):
        m_avg = m.float().mean(dim=0)  # (H, T, T)
        flat = m_avg.flatten(1)
        flat = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        sim = flat @ flat.t()
        out[li] = sim[idx_i, idx_j]
    return out


def dir_similarity(dirs: torch.Tensor) -> torch.Tensor:
    """Coseno entre direcciones por par de cabezas y capa.

    Args:
        dirs: tensor (L, H, D) unitario.

    Returns:
        Tensor (L, n_pares) con cos en [-1, 1] (signo conservado).
    """
    L, H, _ = dirs.shape
    idx_i, idx_j = _pair_idx(H)
    out = torch.zeros(L, idx_i.numel())
    for li in range(L):
        g = dirs[li] @ dirs[li].t()
        out[li] = g[idx_i, idx_j]
    return out


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson r entre dos tensores aplanados."""
    x = x.flatten().float()
    y = y.flatten().float()
    xm = x - x.mean()
    ym = y - y.mean()
    den = float(xm.norm() * ym.norm())
    if den < 1.0e-12:
        return float("nan")
    return float((xm * ym).sum().item() / den)


def main() -> None:
    """Punto de entrada."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="/media/manpla/ST2TB/Imagenet",
        help="raíz de ImageNet (espera val_blurred/ dentro)",
    )
    parser.add_argument(
        "--ckpt", default=None,
        help="checkpoint .pt opcional (carga model state_dict)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("modelo: ViT-B/16 timm preentrenado")
    model = HeadProjections(
        model_name="vit_base_patch16_224",
        pretrained=True, num_classes=1000,
    ).cuda().eval()
    if args.ckpt is not None:
        blob = torch.load(
            args.ckpt, map_location="cuda", weights_only=False,
        )
        # tolera distintos formatos
        if "model" in blob:
            sd = blob["model"]
            # strip prefijo backbone. si lo hay
            sd = {
                k.replace("backbone.", "", 1): v for k, v in sd.items()
            }
        else:
            sd = blob
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(
            f"checkpoint cargado: {args.ckpt} "
            f"(missing={len(miss)} unexpected={len(unexp)})"
        )

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
        ),
    ])
    ds = datasets.ImageFolder(
        f"{args.root}/val_blurred", transform=tfm,
    )
    g = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, generator=g,
    )
    imgs, _ = next(iter(loader))
    imgs = imgs.cuda()
    print(f"lote de validación: {tuple(imgs.shape)}")

    with torch.no_grad():
        with capture_attention(model.model) as maps:
            _ = model(imgs)
    print(
        f"capturadas {len(maps)} capas, mapa[0] shape="
        f"{tuple(maps[0].shape)}"
    )

    with torch.no_grad():
        dirs_A = head_dirs_mean(model).detach().cpu()
        dirs_B = head_dirs_svd(model).detach().cpu()
    s_attn = attn_similarity(maps).cpu()
    s_A = dir_similarity(dirs_A)
    s_B = dir_similarity(dirs_B)

    # también |cos| por si la diversidad funcional ignora el signo
    s_A_abs = s_A.abs()
    s_B_abs = s_B.abs()

    L = s_attn.shape[0]
    H = dirs_A.shape[1]
    n_pairs = s_attn.shape[1]

    print(f"\n=== correlación de Pearson por capa ===")
    print(
        f"{'capa':>4} | "
        f"{'r(attn, A.cos)':>14} | {'r(attn, A.|cos|)':>16} | "
        f"{'r(attn, B.cos)':>14} | {'r(attn, B.|cos|)':>16} | "
        f"{'ganador':>8}"
    )
    rs = []
    for li in range(L):
        rAc = pearson(s_attn[li], s_A[li])
        rAa = pearson(s_attn[li], s_A_abs[li])
        rBc = pearson(s_attn[li], s_B[li])
        rBa = pearson(s_attn[li], s_B_abs[li])
        win = "A" if max(rAc, rAa) > max(rBc, rBa) else "B"
        rs.append((rAc, rAa, rBc, rBa))
        print(
            f"{li:>4} | "
            f"{rAc:>14.4f} | {rAa:>16.4f} | "
            f"{rBc:>14.4f} | {rBa:>16.4f} | "
            f"{win:>8}"
        )

    # agregado: correlación sobre TODOS los pares × capas
    print(f"\n=== correlación agregada (sobre las 12 capas × 66 pares = {L*n_pairs} puntos) ===")
    print(
        f"  r(s_attn, A.cos)   = {pearson(s_attn, s_A):.4f}"
    )
    print(
        f"  r(s_attn, A.|cos|) = {pearson(s_attn, s_A_abs):.4f}"
    )
    print(
        f"  r(s_attn, B.cos)   = {pearson(s_attn, s_B):.4f}"
    )
    print(
        f"  r(s_attn, B.|cos|) = {pearson(s_attn, s_B_abs):.4f}"
    )

    # 15 pares aleatorios para inspección directa
    rng = random.Random(args.seed)
    sample = []
    while len(sample) < 15:
        li = rng.randrange(L)
        pi = rng.randrange(n_pairs)
        sample.append((li, pi))
    idx_i, idx_j = _pair_idx(H)
    print(
        f"\n=== 15 pares aleatorios (capa, h, h') con las tres similitudes ==="
    )
    hp_lbl = "h'"
    print(
        f"{'capa':>4} {'h':>3} {hp_lbl:>3} | "
        f"{'s_attn':>8} {'s_A cos':>9} {'s_A |cos|':>10} "
        f"{'s_B cos':>9} {'s_B |cos|':>10}"
    )
    for li, pi in sample:
        h = int(idx_i[pi])
        hp = int(idx_j[pi])
        print(
            f"{li:>4} {h:>3} {hp:>3} | "
            f"{s_attn[li, pi].item():>8.4f} "
            f"{s_A[li, pi].item():>9.4f} {s_A_abs[li, pi].item():>10.4f} "
            f"{s_B[li, pi].item():>9.4f} {s_B_abs[li, pi].item():>10.4f}"
        )


if __name__ == "__main__":
    main()
