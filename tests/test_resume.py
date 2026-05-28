"""Tests de la lógica resume-safe en train_attn y train_detect.

Verifica:
1. Sin checkpoint previo → entrena desde época 1 y crea CSVs con
   cabecera.
2. Con checkpoint válido → reanuda desde la época guardada, no
   trunca los CSVs, añade en modo append.
3. Con checkpoint incompatible (sin optimizer) → imprime aviso y
   arranca de cero.
"""

import csv
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.train_attn import train_attn


class _TinyAttnModel(nn.Module):
    """Modelo mínimo con la API que train_attn espera.

    Expone ``head_directions()`` que devuelve un tensor (L, H, D)
    derivado de un parámetro entrenable, para que R_div tenga un
    camino diferenciable hacia el optimizador.
    """

    def __init__(self, num_classes: int = 6, l: int = 2,
                 h: int = 3, d: int = 6) -> None:
        super().__init__()
        self.l = l
        self.h = h
        self.d = d
        self.dirs_raw = nn.Parameter(torch.randn(l, h, d) * 0.1)
        self.classifier = nn.Linear(d, num_classes)

    def head_directions(self) -> torch.Tensor:
        """Direcciones unitarias derivadas del parámetro entrenable."""
        return self.dirs_raw / self.dirs_raw.norm(
            dim=-1, keepdim=True,
        ).clamp_min(1.0e-12)

    @property
    def backbone(self):
        """API mínima para que `capture_attention` no se invoque (se
        invoca solo si val_loader proporciona algo; aquí lo dejamos
        compatible pero sin uso real)."""
        return self

    @property
    def model(self):
        """Para `capture_attention(model.model)`."""
        class _M:
            blocks: list = []
        return _M()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward sintético: media espacial + cabeza lineal."""
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        if x.shape[1] != self.d:
            # se proyecta a la dim D
            x = x[:, :self.d] if x.shape[1] > self.d else \
                torch.cat(
                    [x, x.new_zeros(x.shape[0], self.d - x.shape[1])],
                    dim=1,
                )
        return self.classifier(x)


def _make_loaders(num_classes: int = 6, n: int = 16, d: int = 6):
    """DataLoaders sintéticos pequeños."""
    x = torch.randn(n, d)
    y = torch.randint(0, num_classes, (n,))
    ds = TensorDataset(x, y)
    tr = DataLoader(ds, batch_size=8, shuffle=False)
    va = DataLoader(ds, batch_size=8, shuffle=False)
    return tr, va


def test_resume_fresh_creates_headers(tmp_path: Path) -> None:
    """Sin checkpoint previo arranca desde 0 y escribe cabeceras."""
    model = _TinyAttnModel()
    tr, va = _make_loaders()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2)

    train_attn(
        model=model, train_loader=tr, val_loader=va,
        epochs=2, optimizer=opt, scheduler=sched,
        lambda_div=0.0, theta_target_deg=90.0,
        label_smoothing=0.0,
        log_dir=str(tmp_path / "logs"),
        ckpt_dir=str(tmp_path / "ckpts"),
        device="cpu", run_name="tiny",
    )
    run_csv = tmp_path / "logs" / "run.csv"
    assert run_csv.exists()
    rows = run_csv.read_text().strip().splitlines()
    # cabecera + 2 épocas
    assert len(rows) == 3
    assert rows[0].startswith("epoch,lr,train_loss")
    assert rows[1].startswith("1,")
    assert rows[2].startswith("2,")


def test_resume_continues_from_checkpoint(tmp_path: Path) -> None:
    """Con un checkpoint válido, retoma desde la siguiente época."""
    log_dir = tmp_path / "logs"
    ckpt_dir = tmp_path / "ckpts"
    # primera tanda: 2 épocas
    model1 = _TinyAttnModel()
    tr, va = _make_loaders()
    opt1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=5)
    train_attn(
        model=model1, train_loader=tr, val_loader=va,
        epochs=2, optimizer=opt1, scheduler=sched1,
        lambda_div=0.0, theta_target_deg=90.0,
        label_smoothing=0.0, log_dir=str(log_dir),
        ckpt_dir=str(ckpt_dir), device="cpu", run_name="tiny",
    )
    run_csv = log_dir / "run.csv"
    rows_after_first = run_csv.read_text().strip().splitlines()
    assert len(rows_after_first) == 3

    # segunda tanda: total 5 épocas, debería reanudar desde la 3
    model2 = _TinyAttnModel()
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=5)
    train_attn(
        model=model2, train_loader=tr, val_loader=va,
        epochs=5, optimizer=opt2, scheduler=sched2,
        lambda_div=0.0, theta_target_deg=90.0,
        label_smoothing=0.0, log_dir=str(log_dir),
        ckpt_dir=str(ckpt_dir), device="cpu", run_name="tiny",
    )
    rows_after_second = run_csv.read_text().strip().splitlines()
    # cabecera (1) + 5 épocas (5) = 6 líneas
    assert len(rows_after_second) == 6
    # las primeras 3 líneas se preservan (cabecera + ép. 1 + ép. 2)
    assert rows_after_second[:3] == rows_after_first
    # las nuevas son 3, 4, 5
    assert rows_after_second[3].startswith("3,")
    assert rows_after_second[4].startswith("4,")
    assert rows_after_second[5].startswith("5,")


def test_resume_incompatible_checkpoint_starts_over(
    tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    """Un checkpoint sin 'optimizer' se descarta con aviso."""
    log_dir = tmp_path / "logs"
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir(parents=True)
    # checkpoint legacy: solo 'model', 'epoch', sin 'optimizer'
    legacy = {
        "model": _TinyAttnModel().state_dict(),
        "epoch": 6,
        "val_top1": 0.5,
    }
    torch.save(legacy, ckpt_dir / "tiny_last.pt")

    model = _TinyAttnModel()
    tr, va = _make_loaders()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2)
    train_attn(
        model=model, train_loader=tr, val_loader=va,
        epochs=2, optimizer=opt, scheduler=sched,
        lambda_div=0.0, theta_target_deg=90.0,
        label_smoothing=0.0, log_dir=str(log_dir),
        ckpt_dir=str(ckpt_dir), device="cpu", run_name="tiny",
    )
    out = capsys.readouterr().out
    assert "checkpoint incompatible" in out
    assert "optimizer" in out
    # arrancó de cero: 2 épocas + cabecera
    rows = (log_dir / "run.csv").read_text().strip().splitlines()
    assert len(rows) == 3
    assert rows[1].startswith("1,")
    assert rows[2].startswith("2,")
