"""Tests para los loaders sintéticos y para el script de agregación."""

import csv
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.genimage import build_logo_split


def _write_jpg(path: Path, color: tuple[int, int, int]) -> None:
    """Genera una imagen RGB sintética en la ruta dada."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (96, 96), color=color)
    img.save(path, format="JPEG")


def _fake_genimage_root(base: Path, gens: list[str]) -> None:
    """Crea una raíz GenImage sintética con 3 train + 2 val por carpeta."""
    for g in gens:
        for split in ("train", "val"):
            n = 3 if split == "train" else 2
            for i in range(n):
                _write_jpg(
                    base / g / split / "ai" / f"{i:03d}.jpg",
                    color=(20 * i % 255, 50, 90),
                )
                _write_jpg(
                    base / g / split / "nature" / f"{i:03d}.jpg",
                    color=(40, 30 * i % 255, 120),
                )


def test_genimage_logo_split_labels(tmp_path: Path) -> None:
    """build_logo_split excluye el generador indicado del train."""
    gens = ["gA", "gB", "gC"]
    _fake_genimage_root(tmp_path, gens)
    train_loader, val_loader, lmap = build_logo_split(
        root=str(tmp_path),
        generators=gens,
        held_out="gB",
        image_size=64,
        batch_size=4,
        num_workers=0,
    )
    assert lmap == {"real": 0, "gA": 1, "gB": 2, "gC": 3}
    # ningún lote de train contiene la etiqueta del excluido (2)
    seen = set()
    for _, lbls in train_loader:
        seen.update(lbls.tolist())
    assert 2 not in seen, f"el excluido apareció en train: {seen}"
    # validación sí incluye al excluido
    seen_val = set()
    for _, lbls in val_loader:
        seen_val.update(lbls.tolist())
    assert 2 in seen_val


def test_genimage_logo_split_shapes(tmp_path: Path) -> None:
    """Las imágenes salen con la forma (B, 3, image_size, image_size)."""
    gens = ["gA", "gB"]
    _fake_genimage_root(tmp_path, gens)
    train_loader, _, _ = build_logo_split(
        root=str(tmp_path),
        generators=gens,
        held_out="gA",
        image_size=64,
        batch_size=2,
        num_workers=0,
    )
    imgs, lbls = next(iter(train_loader))
    assert imgs.shape == (2, 3, 64, 64)
    assert lbls.shape == (2,)


def _write_csv(path: Path, header: list[str],
               rows: list[list]) -> None:
    """Escribe un CSV con cabecera y filas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        for r in rows:
            wr.writerow(r)


def test_aggregate_end_to_end(tmp_path: Path) -> None:
    """El script aggregate.py corre sobre logs sintéticos."""
    logs = tmp_path / "logs"
    codes = tmp_path / "codes"
    ckpts = tmp_path / "ckpts"
    tables = tmp_path / "tables"
    figs = tmp_path / "figs"

    # csv de códigos
    _write_csv(
        logs / "codes.csv",
        [
            "name", "d", "K", "theta_min_deg", "theta_max_deg",
            "theta_median_deg", "theta_mean_deg",
            "energy_riesz_s1", "energy_canonical",
            "kissing_ok", "n_pairs",
        ],
        [
            ["k2", 2, 6, 60.0, 120.0, 120.0, 90.0,
             10.0, 10.0, 1, 15],
            ["k4", 4, 24, 60.0, 180.0, 90.0, 88.0,
             208.3, 208.3, 1, 276],
        ],
    )
    # csv principal de A
    _write_csv(
        logs / "attn" / "run.csv",
        [
            "epoch", "lr", "train_loss", "ce", "div",
            "redundancy_mean", "theta_min_mean_deg",
            "val_top1", "val_top5",
        ],
        [
            [1, 1e-4, 2.0, 1.9, 0.1, 0.3, 70.0, 0.4, 0.7],
            [2, 8e-5, 1.5, 1.4, 0.1, 0.2, 75.0, 0.5, 0.8],
            [3, 5e-5, 1.0, 0.9, 0.1, 0.1, 80.0, 0.6, 0.9],
        ],
    )
    # csv por capa de A
    _write_csv(
        logs / "attn" / "layerwise.csv",
        ["epoch", "layer", "redundancy", "theta_min_deg"],
        [
            [3, 0, 0.20, 70.0],
            [3, 1, 0.15, 75.0],
        ],
    )
    # csvs LOGO
    _write_csv(
        logs / "detect" / "gA.csv",
        [
            "epoch", "loss", "val_auroc_realfake", "val_oscr",
            "val_closed_top1", "val_acc_realfake",
        ],
        [
            [1, 0.7, 0.85, 0.65, 0.80, 0.82],
            [2, 0.5, 0.90, 0.70, 0.85, 0.88],
        ],
    )
    _write_csv(
        logs / "detect" / "gB.csv",
        [
            "epoch", "loss", "val_auroc_realfake", "val_oscr",
            "val_closed_top1", "val_acc_realfake",
        ],
        [
            [1, 0.6, 0.88, 0.68, 0.83, 0.85],
            [2, 0.4, 0.92, 0.72, 0.87, 0.90],
        ],
    )
    # códigos vacíos (no requeridos para tablas pero el script
    # los escanea)
    codes.mkdir()

    script = (
        Path(__file__).resolve().parents[1] / "scripts"
        / "aggregate.py"
    )
    result = subprocess.run(
        [
            sys.executable, str(script),
            "--logs", str(logs),
            "--codes", str(codes),
            "--ckpts", str(ckpts),
            "--tables", str(tables),
            "--figs", str(figs),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"aggregate falló: {result.stderr}"
    )
    # las tres tablas existen y contienen un cuerpo tabularx
    for name in (
        "tabla_codigos.tex",
        "tabla_atencion.tex",
        "tabla_logo.tex",
    ):
        p = tables / name
        assert p.exists(), f"{name} no se generó"
        body = p.read_text()
        assert "tabularx" in body
        assert r"\toprule" in body
        assert r"\bottomrule" in body
    # la tabla LOGO debe contener la línea de media
    assert "media" in (tables / "tabla_logo.tex").read_text()
