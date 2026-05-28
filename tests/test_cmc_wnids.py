"""Tests para la selección por lista de wnids (CMC ImageNet-100)."""

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.imagenet import (
    _RemappedSubset,
    _select_subset,
    build_imagenet_loaders,
)


def _build_fake_imagenet(root: Path, wnids: list[str], n_per: int = 3):
    """Crea estructura tipo ImageFolder con synsets dadas."""
    for w in wnids:
        d = root / w
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_per):
            Image.new("RGB", (64, 64), (i * 30 % 255, 80, 130)).save(
                d / f"{w}_{i}.jpg", "JPEG",
            )


def test_remapped_subset_relabels_to_compact_range(
    tmp_path: Path,
) -> None:
    """Las etiquetas se remapean al orden de la lista wnids."""
    from torchvision import datasets, transforms

    all_wnids = ["n00000001", "n00000002", "n00000003",
                 "n00000004", "n00000005"]
    _build_fake_imagenet(tmp_path / "train", all_wnids, n_per=2)

    ds = datasets.ImageFolder(
        tmp_path / "train",
        transform=transforms.ToTensor(),
    )
    # selecciono 3 de 5, EN ORDEN INVERSO al alfabético, para forzar
    # que la lista determine el remapeo
    wnids_subset = ["n00000004", "n00000002", "n00000005"]
    sub = _select_subset(ds, num_classes=3, wnids=wnids_subset)
    assert isinstance(sub, _RemappedSubset)
    # cada wnid debe mapearse a su posición en wnids_subset
    assert sub.label_map[ds.class_to_idx["n00000004"]] == 0
    assert sub.label_map[ds.class_to_idx["n00000002"]] == 1
    assert sub.label_map[ds.class_to_idx["n00000005"]] == 2
    # iterando, las etiquetas que salen están en [0, 1, 2]
    labels = sorted({lbl for _, lbl in sub})
    assert labels == [0, 1, 2]


def test_select_subset_missing_wnid_raises(tmp_path: Path) -> None:
    """Pasar un wnid que no existe en el dataset levanta ValueError."""
    from torchvision import datasets, transforms

    _build_fake_imagenet(
        tmp_path / "train",
        ["n00000001", "n00000002"],
        n_per=2,
    )
    ds = datasets.ImageFolder(
        tmp_path / "train",
        transform=transforms.ToTensor(),
    )
    with pytest.raises(ValueError, match="faltantes"):
        _select_subset(ds, num_classes=2,
                       wnids=["n00000001", "n00000999"])


def test_build_loaders_with_wnids(tmp_path: Path) -> None:
    """build_imagenet_loaders pasa wnids correctamente."""
    wnids_all = [f"n0000{i:04d}" for i in range(5)]
    wnids_keep = [wnids_all[3], wnids_all[1]]  # 2 en orden invertido
    _build_fake_imagenet(tmp_path / "train", wnids_all, n_per=2)
    _build_fake_imagenet(tmp_path / "val", wnids_all, n_per=1)
    tr, va = build_imagenet_loaders(
        root=str(tmp_path),
        image_size=64,
        batch_size=2,
        num_workers=0,
        num_classes=2,
        train_subdir="train",
        val_subdir="val",
        wnids=wnids_keep,
    )
    # 2 wnids × 2 imgs train = 4 imgs total → 2 batches con bs=2
    all_lbls_tr = []
    for _, lbls in tr:
        all_lbls_tr.extend(lbls.tolist())
    all_lbls_va = []
    for _, lbls in va:
        all_lbls_va.extend(lbls.tolist())
    # solo etiquetas {0, 1}, según orden de wnids_keep
    assert set(all_lbls_tr) == {0, 1}
    assert set(all_lbls_va) == {0, 1}


def test_cmc_file_loads_and_has_100(tmp_path: Path) -> None:
    """El fichero CMC existe en configs/ y contiene 100 synsets."""
    cmc = (
        Path(__file__).resolve().parents[1]
        / "configs" / "imagenet100_cmc.txt"
    )
    assert cmc.exists(), "configs/imagenet100_cmc.txt no presente"
    lines = [ln.strip() for ln in cmc.read_text().splitlines()
             if ln.strip()]
    assert len(lines) == 100, f"esperadas 100 synsets, hay {len(lines)}"
    for w in lines:
        assert w.startswith("n") and len(w) == 9, (
            f"wnid malformado: {w!r}"
        )
