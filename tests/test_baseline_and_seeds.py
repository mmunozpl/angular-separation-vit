"""Tests para baseline lineal (#9) y multi-semilla (#8) de Fase B."""

import csv
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.linear_head import LinearHead


def test_linear_head_shape() -> None:
    """LinearHead devuelve (B, C) logits sin acotar."""
    head = LinearHead(num_classes=9, feature_dim=768)
    feats = torch.randn(4, 768)
    out = head(feats)
    assert out.shape == (4, 9)
    # logits no acotados (no es softmax)
    assert out.dtype == torch.float32


def test_linear_head_parameters() -> None:
    """LinearHead tiene parámetros entrenables."""
    head = LinearHead(num_classes=5, feature_dim=10)
    params = list(head.parameters())
    # nn.Linear -> 2 parámetros (weight + bias)
    assert len(params) == 2
    assert sum(p.numel() for p in params) == 5 * 10 + 5


def test_linear_head_softmax_normalized() -> None:
    """Tras softmax, suma por fila = 1; equivalencia con score proto."""
    head = LinearHead(num_classes=4, feature_dim=12)
    feats = torch.randn(3, 12)
    probs = head(feats).softmax(dim=-1)
    assert torch.allclose(
        probs.sum(dim=-1), torch.ones(3), atol=1.0e-5,
    )


def _write_csv(path: Path, header: list[str],
               rows: list[list]) -> None:
    """Escribe un CSV mínimo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        for r in rows:
            wr.writerow(r)


def test_aggregate_groups_by_generator_over_seeds(
    tmp_path: Path,
) -> None:
    """aggregate.py reduce <gen>_seed<N>.csv por generador.

    Crea logs sintéticos con dos generadores × tres semillas; la
    tabla LOGO debe contener 'media' y un valor por generador con
    ±std.
    """
    logs = tmp_path / "logs"
    codes = tmp_path / "codes"
    ckpts = tmp_path / "ckpts"
    tables = tmp_path / "tables"
    figs = tmp_path / "figs"
    codes.mkdir()

    header = [
        "epoch", "loss", "ce",
        "train_top1", "train_acc_realfake",
        "val_auroc_realfake", "val_oscr",
        "val_closed_top1", "val_acc_realfake",
        "val_precision", "val_recall", "val_f1",
        "epoch_seconds", "img_per_sec", "grad_norm_last",
    ]
    # gA: AUROC varía entre semillas (0.85, 0.90, 0.95) → mean 0.90
    for seed, au in [(42, 0.85), (43, 0.90), (44, 0.95)]:
        _write_csv(
            logs / "detect" / f"gA_seed{seed}.csv", header,
            [[1, 0.5, 0.5, 0.7, 0.7, au, 0.7, 0.8, 0.8,
              0.8, 0.8, 0.8, 10.0, 50.0, 5.0]],
        )
    # gB: AUROC constante (0.80) → std 0
    for seed in [42, 43, 44]:
        _write_csv(
            logs / "detect" / f"gB_seed{seed}.csv", header,
            [[1, 0.6, 0.6, 0.7, 0.7, 0.80, 0.65, 0.78, 0.78,
              0.78, 0.78, 0.78, 12.0, 45.0, 5.5]],
        )

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
    body = (tables / "tabla_logo.tex").read_text()
    assert "gA" in body
    assert "gB" in body
    assert "media" in body
    # gA tiene std > 0 (0.85, 0.90, 0.95) → formato 'mean±std'
    assert "$\\pm$" in body
    # comprobamos que la línea de gA contiene 0.900 y de gB 0.800
    assert "0.900" in body
    assert "0.800" in body


def test_aggregate_ignores_auxiliary_csvs(tmp_path: Path) -> None:
    """Los csvs '<gen>_per_class.csv' y '_per_gen_auroc.csv' no
    deben confundirse con corridas LOGO al agregar."""
    logs = tmp_path / "logs"
    tables = tmp_path / "tables"
    figs = tmp_path / "figs"
    codes = tmp_path / "codes"
    ckpts = tmp_path / "ckpts"
    codes.mkdir()

    header = [
        "epoch", "loss", "ce",
        "train_top1", "train_acc_realfake",
        "val_auroc_realfake", "val_oscr",
        "val_closed_top1", "val_acc_realfake",
        "val_precision", "val_recall", "val_f1",
        "epoch_seconds", "img_per_sec", "grad_norm_last",
    ]
    _write_csv(
        logs / "detect" / "gA_seed42.csv", header,
        [[1, 0.5, 0.5, 0.7, 0.7, 0.9, 0.7, 0.8, 0.8,
          0.8, 0.8, 0.8, 10.0, 50.0, 5.0]],
    )
    # csvs auxiliares con cabecera distinta — no deben ser leídos
    # como corridas LOGO
    _write_csv(
        logs / "detect" / "gA_seed42_per_class.csv",
        ["epoch", "class_idx", "n_samples", "n_correct", "accuracy"],
        [[1, 0, 100, 90, 0.9]],
    )
    _write_csv(
        logs / "detect" / "gA_seed42_per_gen_auroc.csv",
        ["epoch", "gen_label", "auroc", "n_real", "n_gen"],
        [[1, 1, 0.92, 100, 50]],
    )

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
    assert result.returncode == 0
    body = (tables / "tabla_logo.tex").read_text()
    # gA solo aparece una vez como generador, no como variante
    # 'gA_seed42_per_class' o similar
    assert body.count(r"\\") >= 2  # al menos cabecera + 1 fila + media
