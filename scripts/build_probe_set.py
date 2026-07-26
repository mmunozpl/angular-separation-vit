"""Construye el probe set congelado compartido por D, E y G.

Probe set: 1000 imágenes de val (10 por clase × 100 clases CMC-100),
estratificado por clase, semilla fija 42. Se serializa la lista de
paths ordenada por clase canónica (orden del `imagenet100_cmc.txt`)
con la clase remapeada a [0, 99] en ese mismo orden.

El fichero resultante se carga desde las tres sondas para garantizar
que las correlaciones cruzadas (s_func vs sink_score vs s_subespacio
vs s_sem) se computan sobre la misma población y no se contaminan con
ruido de muestreo distinto por sonda. Ver `pipeline_v3.md` §4 (corr.
E4) y la versión revisada de §4.E.

Salida:
    artifacts/probe_set/imagenet100_val_1k.pt
        dict con claves:
            paths      lista[str] (n=1000) ordenada por clase
            labels     tensor[int64] (n,) con la clase en [0, 99]
            wnids      lista[str] (100,) en orden canónico CMC
            seed       42
"""

import argparse
import random
import sys
from pathlib import Path

import torch


def _load_wnids(path: Path) -> list[str]:
    """Lee la lista canónica de 100 wnids (orden CMC)."""
    return [
        ln.strip() for ln in path.read_text().splitlines() if ln.strip()
    ]


def _images_under(class_dir: Path) -> list[Path]:
    """Lista todos los .JPEG bajo una carpeta de clase."""
    return sorted(
        p for p in class_dir.iterdir()
        if p.suffix.upper() in {".JPEG", ".JPG"}
    )


def build(
    val_root: Path,
    wnids_file: Path,
    n_per_class: int,
    seed: int,
    out_path: Path,
) -> None:
    """Construye el probe set y lo serializa.

    Args:
        val_root: raíz que contiene una carpeta por wnid.
        wnids_file: lista canónica de wnids (100 líneas).
        n_per_class: imágenes por clase a muestrear (10 por defecto).
        seed: semilla para el muestreo determinista.
        out_path: dónde guardar el .pt.
    """
    wnids = _load_wnids(wnids_file)
    if len(wnids) != 100:
        raise ValueError(
            f"se esperan 100 wnids; se encontraron {len(wnids)}"
        )

    rng = random.Random(seed)
    paths: list[str] = []
    labels: list[int] = []
    for label, wnid in enumerate(wnids):
        cls_dir = val_root / wnid
        if not cls_dir.is_dir():
            raise FileNotFoundError(
                f"clase {wnid} ausente bajo {val_root}"
            )
        imgs = _images_under(cls_dir)
        if len(imgs) < n_per_class:
            raise ValueError(
                f"clase {wnid} solo tiene {len(imgs)} imágenes "
                f"(se piden {n_per_class})"
            )
        picked = rng.sample(imgs, n_per_class)
        # se ordenan dentro de la clase tras el sampling, así una
        # implementación que iterara sobre el csv en orden lee
        # primero todo de la clase 0, luego clase 1, etc.
        for p in sorted(picked):
            paths.append(str(p))
            labels.append(label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "paths": paths,
        "labels": torch.tensor(labels, dtype=torch.int64),
        "wnids": wnids,
        "seed": int(seed),
        "n_per_class": int(n_per_class),
    }
    torch.save(blob, out_path)
    print(
        f"[probe_set] guardado en {out_path}\n"
        f"  total: {len(paths)} imgs ({n_per_class}/clase × 100)\n"
        f"  semilla: {seed}",
        flush=True,
    )

    # se muestran 15 entradas aleatorias para inspección visual rápida.
    idx_show = random.Random(seed).sample(range(len(paths)), 15)
    print("\n=== 15 muestras aleatorias del probe set ===")
    for i in sorted(idx_show):
        wnid = wnids[int(labels[i])]
        rel = Path(paths[i]).name
        print(f"  [{i:>4}] clase={labels[i]:>3} {wnid}  {rel}")


def main() -> None:
    """Punto de entrada CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val-root",
        default="/media/manpla/Pruebas/Imagenet/val_blurred",
    )
    parser.add_argument(
        "--wnids-file",
        default="configs/imagenet100_cmc.txt",
    )
    parser.add_argument("--n-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default="artifacts/probe_set/imagenet100_val_1k.pt",
    )
    args = parser.parse_args()
    build(
        val_root=Path(args.val_root),
        wnids_file=Path(args.wnids_file),
        n_per_class=int(args.n_per_class),
        seed=int(args.seed),
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()
