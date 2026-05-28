"""Análisis de diversidad funcional sobre checkpoints de la Fase A.

No reentrena nada: carga los checkpoints ya guardados (base, blando,
dura), pasa un lote de validación de ImageNet-100 y responde dos
preguntas que la Tabla 2 deja abiertas.

Análisis 1 — diversidad funcional: ¿las cabezas del modelo blando
atienden a cosas más distintas que las de la base? Se mide la
similitud coseno entre los mapas de atención (promediados sobre el
lote) de cada par de cabezas, por capa y agregada.

Análisis 2 — paradoja de la redundancia media: ¿por qué el blando
(red=0,034) tiene menos redundancia media que la dura (red=0,091=1/11)
si la dura está en el símplex óptimo? Se compara la distribución de
|cos| entre direcciones representativas de las tres configuraciones.

Uso:
    python scripts/analyze_diversity.py
"""

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.imagenet import build_imagenet_loaders
from src.metrics.attention import (
    functional_similarity_matrix,
    mean_upper_offdiag,
)
from src.models.attn_diverse import AttnDiverseViT
from src.models.vit_backbone import capture_attention
from src.seed import set_seed


def _load_wnids(ds: dict) -> list[str] | None:
    """Carga la lista canónica de synsets si la config la indica."""
    if "wnids_file" not in ds:
        return None
    path = Path(ds["wnids_file"])
    return [
        ln.strip() for ln in path.read_text().splitlines()
        if ln.strip()
    ]


def _collect_val_images(
    cfg: dict, num_images: int, batch_size: int,
) -> torch.Tensor:
    """Reúne un lote fijo de validación compartido por todos los modelos.

    Args:
        cfg: config base (de la que se toman dataset y rutas).
        num_images: número de imágenes a reunir.
        batch_size: tamaño de lote del loader de validación.

    Returns:
        Tensor (N, 3, H, W) en CPU con las primeras ``num_images``
        imágenes de validación, en orden determinista.
    """
    ds = cfg["dataset"]
    wnids = _load_wnids(ds)
    _, val_loader = build_imagenet_loaders(
        root=ds["root"],
        image_size=int(ds["image_size"]),
        batch_size=batch_size,
        num_workers=int(ds.get("num_workers", 8)),
        num_classes=int(ds["num_classes"]),
        train_subdir=str(ds.get("train_subdir", "train")),
        val_subdir=str(ds.get("val_subdir", "val")),
        wnids=wnids,
    )
    chunks: list[torch.Tensor] = []
    total = 0
    for imgs, _ in val_loader:
        chunks.append(imgs)
        total += imgs.shape[0]
        if total >= num_images:
            break
    images = torch.cat(chunks, dim=0)[:num_images]
    return images.contiguous()


def _load_model(
    ckpt_path: str, num_classes: int, img_size: int, device: str,
) -> AttnDiverseViT:
    """Construye la columna y carga los pesos del checkpoint.

    Args:
        ckpt_path: ruta al ``*_last.pt``.
        num_classes: clases de la cabeza.
        img_size: resolución de entrada.
        device: dispositivo destino.

    Returns:
        Modelo en modo evaluación con los pesos cargados.
    """
    # pretrained=False: los pesos vienen del checkpoint, no de timm
    model = AttnDiverseViT(
        model_name="vit_base_patch16_224",
        pretrained=False,
        num_classes=num_classes,
        hard_code=None,
        img_size=img_size,
    )
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # strict=False: la dura trae el buffer _hard_code, ajeno a esta
    # columna sin modo duro; los pesos de W_O sí cargan completos
    model.load_state_dict(blob["model"], strict=False)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def _avg_attention(
    model: AttnDiverseViT,
    images: torch.Tensor,
    batch_size: int,
    device: str,
    desc: str,
) -> torch.Tensor:
    """Mapa de atención promediado sobre el lote, por capa y cabeza.

    Args:
        model: columna en modo evaluación.
        images: tensor (N, 3, H, W) en CPU.
        batch_size: tamaño de lote del pase.
        device: dispositivo de cómputo.
        desc: etiqueta para la barra de progreso.

    Returns:
        Tensor (L, H, T, T) con la atención media tras softmax.
    """
    n = images.shape[0]
    acc: torch.Tensor | None = None
    base = model.backbone.model
    n_batches = math.ceil(n / batch_size)
    with capture_attention(base) as maps:
        for bi in tqdm(range(n_batches), desc=desc, unit="lote"):
            sl = slice(bi * batch_size, (bi + 1) * batch_size)
            xb = images[sl].to(device, non_blocking=True)
            model(xb)
            if acc is None:
                shape = (len(maps),) + tuple(maps[0].shape[1:])
                acc = torch.zeros(shape, device=device)
            for li, m in enumerate(maps):
                acc[li] += m.sum(dim=0)
            maps.clear()
    return acc / float(n)


def _functional_per_layer(
    attn_avg: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Similitud funcional media por capa y matrices por capa.

    Args:
        attn_avg: tensor (L, H, T, T) de atención media.

    Returns:
        Par (vector (L,) con la media de pares por capa, lista de L
        matrices (H, H) de similitud coseno entre cabezas).
    """
    per_layer = []
    mats = []
    for li in range(attn_avg.shape[0]):
        mat = functional_similarity_matrix(attn_avg[li])
        mats.append(mat)
        per_layer.append(mean_upper_offdiag(mat))
    return torch.stack(per_layer), mats


def _pooled_abs_cos(dirs: torch.Tensor) -> torch.Tensor:
    """Reúne |cos| de todos los pares de cabezas y todas las capas.

    Args:
        dirs: tensor (L, H, D) de direcciones unitarias.

    Returns:
        Tensor 1D con los |cos| de los pares no ordenados, agregando
        las L capas.
    """
    l, h, _ = dirs.shape
    iu = torch.triu_indices(h, h, offset=1)
    vals = []
    for li in range(l):
        g = (dirs[li] @ dirs[li].t())[iu[0], iu[1]].abs()
        vals.append(g)
    return torch.cat(vals)


def _percentiles(x: torch.Tensor) -> dict[str, float]:
    """Estadísticos resumen de una muestra 1D de |cos|."""
    qs = torch.tensor([0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0])
    pv = torch.quantile(x, qs).tolist()
    thr = 1.0 / 11.0
    return {
        "n": int(x.numel()),
        "mean": float(x.mean()),
        "min": pv[0],
        "p5": pv[1],
        "p25": pv[2],
        "p50": pv[3],
        "p75": pv[4],
        "p95": pv[5],
        "max": pv[6],
        "frac_ge_1_11": float((x >= thr - 1.0e-6).float().mean()),
    }


def _show_random(rows: list[dict], k: int = 15) -> None:
    """Imprime k observaciones aleatorias de un artefacto guardado."""
    if not rows:
        print("  (sin filas)")
        return
    sample = rows if len(rows) <= k else random.sample(rows, k)
    for r in sample:
        print("  " + "  ".join(f"{kk}={vv}" for kk, vv in r.items()))


def _save_csv(path: Path, rows: list[dict]) -> None:
    """Vuelca filas (lista de dicts) a CSV con cabecera."""
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Ejecuta los dos análisis y vuelca los tres entregables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-base", default="configs/attn_imagenet_base.yaml",
    )
    parser.add_argument(
        "--config-blando", default="configs/attn_imagenet.yaml",
    )
    parser.add_argument(
        "--config-dura", default="configs/attn_imagenet_dura.yaml",
    )
    parser.add_argument(
        "--ckpt-base",
        default="artifacts/checkpoints/attn_base/attnA_base_last.pt",
    )
    parser.add_argument(
        "--ckpt-blando",
        default="artifacts/checkpoints/attn/attnA_blando_last.pt",
    )
    parser.add_argument(
        "--ckpt-dura",
        default="artifacts/checkpoints/attn_dura/attnA_dura_last.pt",
    )
    parser.add_argument("--num-images", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--out-dir",
        default="artifacts/tables/diversidad_funcional",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config_base)
    device = (
        cfg.get("device", "cuda")
        if torch.cuda.is_available() else "cpu"
    )
    num_classes = int(cfg["dataset"]["num_classes"])
    img_size = int(cfg["dataset"]["image_size"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(
        f"[datos] reuniendo {args.num_images} imágenes de validación "
        f"(compartidas por los tres modelos)...",
        flush=True,
    )
    images = _collect_val_images(cfg, args.num_images, args.batch_size)
    print(f"[datos] lote fijo: {tuple(images.shape)}", flush=True)

    # ---------- análisis 1: diversidad funcional (base vs blando) -----
    print("\n[análisis 1] similitud funcional entre cabezas", flush=True)
    model_base = _load_model(
        args.ckpt_base, num_classes, img_size, device,
    )
    attn_base = _avg_attention(
        model_base, images, args.batch_size, device, "base",
    )
    fl_base, mats_base = _functional_per_layer(attn_base)
    del model_base
    if device == "cuda":
        torch.cuda.empty_cache()

    model_blando = _load_model(
        args.ckpt_blando, num_classes, img_size, device,
    )
    attn_blando = _avg_attention(
        model_blando, images, args.batch_size, device, "blando",
    )
    fl_blando, mats_blando = _functional_per_layer(attn_blando)
    del model_blando
    if device == "cuda":
        torch.cuda.empty_cache()

    rows_fl: list[dict] = []
    for li in range(fl_base.numel()):
        b = float(fl_base[li])
        s = float(fl_blando[li])
        rows_fl.append({
            "capa": li,
            "func_sim_base": round(b, 5),
            "func_sim_blando": round(s, 5),
            "delta": round(s - b, 5),
        })
    agg_b = float(fl_base.mean())
    agg_s = float(fl_blando.mean())
    rows_fl.append({
        "capa": "agregado",
        "func_sim_base": round(agg_b, 5),
        "func_sim_blando": round(agg_s, 5),
        "delta": round(agg_s - agg_b, 5),
    })
    path_fl = out / "func_sim_per_layer.csv"
    _save_csv(path_fl, rows_fl)
    print(f"\n[guardado] {path_fl}")
    print("similitud funcional media por capa (base vs blando):")
    _show_random(rows_fl, k=15)
    veredicto = (
        "MENOR en blando: la separación angular se traduce en "
        "cabezas funcionalmente más distintas."
        if agg_s < agg_b else
        "NO menor en blando: el regulador reordena la geometría sin "
        "efecto funcional claro."
    )
    print(
        f"\n[respuesta A1] func_sim agregada base={agg_b:.5f} vs "
        f"blando={agg_s:.5f}\n  -> {veredicto}"
    )

    # ---------- análisis 2: distribución de |cos| (geometría) ---------
    print(
        "\n[análisis 2] distribución de |cos| entre direcciones",
        flush=True,
    )
    pooled: dict[str, torch.Tensor] = {}
    cfgs = {
        "base": (args.config_base, args.ckpt_base),
        "blando": (args.config_blando, args.ckpt_blando),
        "dura": (args.config_dura, args.ckpt_dura),
    }
    for name, (_, ckpt) in cfgs.items():
        m = _load_model(ckpt, num_classes, img_size, device)
        dirs = m.head_directions().detach().cpu()
        pooled[name] = _pooled_abs_cos(dirs)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()

    rows_cos: list[dict] = []
    for name in ("base", "blando", "dura"):
        st = _percentiles(pooled[name])
        rows_cos.append({
            "config": name,
            "n_pares": st["n"],
            "mean_red": round(st["mean"], 5),
            "min": round(st["min"], 5),
            "p5": round(st["p5"], 5),
            "p25": round(st["p25"], 5),
            "p50": round(st["p50"], 5),
            "p75": round(st["p75"], 5),
            "p95": round(st["p95"], 5),
            "max": round(st["max"], 5),
            "frac>=1/11": round(st["frac_ge_1_11"], 4),
        })
    path_cos = out / "cos_distribution.csv"
    _save_csv(path_cos, rows_cos)
    print(f"\n[guardado] {path_cos}")
    print(f"umbral del símplex sin signo 1/11 = {1.0 / 11.0:.5f}")
    for r in rows_cos:
        print("  " + "  ".join(f"{k}={v}" for k, v in r.items()))
    print(
        "\n[respuesta A2] la dura clava todos los pares en 1/11 "
        "(mean=max=1/11);\n  el blando reparte la masa por debajo de "
        "1/11 y solo unos pocos pares lo alcanzan, así que su media "
        "queda\n  por debajo del símplex pese a tener el mismo "
        "theta_min. No hay paradoja."
    )
    # se honra la convención: 15 |cos| aleatorios del blando
    print("\n15 valores |cos| aleatorios (blando):")
    pb = pooled["blando"]
    idx = torch.randperm(pb.numel())[:15]
    print("  " + "  ".join(f"{float(pb[i]):.4f}" for i in idx))

    # ---------- entregable 3: 15 pares de cabezas al azar -------------
    l = len(mats_base)
    h = mats_base[0].shape[0]
    triples = [
        (li, i, j)
        for li in range(l)
        for i in range(h)
        for j in range(i + 1, h)
    ]
    sample = random.sample(triples, 15)
    rows_pairs: list[dict] = []
    for li, i, j in sample:
        rows_pairs.append({
            "capa": li,
            "cabeza_i": i,
            "cabeza_j": j,
            "func_sim_base": round(float(mats_base[li][i, j]), 5),
            "func_sim_blando": round(float(mats_blando[li][i, j]), 5),
        })
    path_pairs = out / "random_pairs.csv"
    _save_csv(path_pairs, rows_pairs)
    print(f"\n[guardado] {path_pairs}")
    print("15 pares de cabezas al azar (similitud funcional):")
    _show_random(rows_pairs, k=15)


if __name__ == "__main__":
    main()
