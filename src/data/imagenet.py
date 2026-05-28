"""Loaders para ImageNet-100 / ImageNet-1k con estructura ImageFolder."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


class _RemappedSubset(Dataset):
    """Subset que además remapea etiquetas a un rango compacto.

    ImageFolder asigna las etiquetas en orden alfabético de las
    carpetas (0..999 para ImageNet-1k). Cuando se filtra a una lista
    arbitraria de synsets (p. ej. la canónica CMC para ImageNet-100),
    las etiquetas resultantes serían dispersas en [0, 999] y rompen
    cross_entropy. Esta clase remapea a un rango compacto [0..K-1]
    según el orden de la lista pasada al loader.
    """

    def __init__(
        self,
        base: Dataset,
        indices: list[int],
        label_map: dict[int, int],
    ) -> None:
        """Inicializa la envoltura.

        Args:
            base: dataset original (ImageFolder).
            indices: índices de muestras a conservar.
            label_map: dict ``orig_label -> new_label``.
        """
        self.base = base
        self.indices = indices
        self.label_map = label_map

    def __len__(self) -> int:
        """Tamaño efectivo del subset."""
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        """Devuelve (imagen, etiqueta remapeada)."""
        img, lbl = self.base[self.indices[i]]
        return img, self.label_map[int(lbl)]


def _build_transforms(image_size: int, train: bool):
    """Transformaciones estándar (ImageNet mean/std)."""
    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        norm,
    ])


def _select_subset(
    ds: datasets.ImageFolder,
    num_classes: int,
    wnids: list[str] | None = None,
    max_per_class: int | None = None,
    subsample_seed: int = 42,
) -> Subset | _RemappedSubset | datasets.ImageFolder:
    """Restringe el ImageFolder a un subconjunto de clases.

    Modos:

    - Si ``wnids`` es una lista, filtra a esas synsets y remapea
      etiquetas al orden de la lista (0..len(wnids)-1).
      Pensado para ImageNet-100 canónico de CMC.
    - Si ``wnids`` es None y ``num_classes < total``, filtra a las
      primeras ``num_classes`` synsets alfabéticas; las etiquetas
      coinciden con los índices del ImageFolder (no requiere remap).
    - Si ``num_classes`` coincide con el total, devuelve ``ds``
      sin tocar.

    Args:
        ds: dataset ImageFolder original.
        num_classes: clases a conservar (fallback si no hay wnids).
        wnids: lista opcional de synsets en el orden deseado.

    Returns:
        Subset (con o sin remapeo) o el dataset original.

    Raises:
        ValueError: si algún wnid de la lista no está en ds.classes.
    """
    if wnids is not None:
        missing = [w for w in wnids if w not in ds.class_to_idx]
        if missing:
            raise ValueError(
                f"synsets faltantes en el dataset: "
                f"{missing[:5]}... ({len(missing)} en total)"
            )
        label_map = {
            ds.class_to_idx[w]: new for new, w in enumerate(wnids)
        }
        keep_cls = set(label_map)
        # se agrupan por clase para poder submuestrear
        by_cls: dict[int, list[int]] = {}
        for i, (_, c) in enumerate(ds.samples):
            if c in keep_cls:
                by_cls.setdefault(c, []).append(i)
        keep_idx: list[int] = []
        if max_per_class is not None:
            import random
            rng = random.Random(subsample_seed)
            for c, idxs in by_cls.items():
                rng.shuffle(idxs)
                keep_idx.extend(idxs[:max_per_class])
        else:
            for idxs in by_cls.values():
                keep_idx.extend(idxs)
        return _RemappedSubset(ds, keep_idx, label_map)
    if num_classes >= len(ds.classes):
        return ds
    keep = set(range(num_classes))
    by_cls = {}
    for i, (_, c) in enumerate(ds.samples):
        if c in keep:
            by_cls.setdefault(c, []).append(i)
    if max_per_class is not None:
        import random
        rng = random.Random(subsample_seed)
        keep_idx = []
        for c, idxs in by_cls.items():
            rng.shuffle(idxs)
            keep_idx.extend(idxs[:max_per_class])
    else:
        keep_idx = [i for ixs in by_cls.values() for i in ixs]
    return Subset(ds, keep_idx)


def build_imagenet_loaders(
    root: str,
    image_size: int = 224,
    batch_size: int = 128,
    num_workers: int = 8,
    num_classes: int = 1000,
    train_subdir: str = "train",
    val_subdir: str = "val",
    wnids: list[str] | None = None,
    max_per_class: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Construye DataLoaders de train y val.

    Espera ``root/train_subdir`` y ``root/val_subdir`` con estructura
    ImageFolder.

    - Si ``wnids`` se pasa, se queda con esas synsets y remapea las
      etiquetas a [0..len(wnids)-1] según el orden de la lista
      (recomendado para ImageNet-100 canónico).
    - Si no, fallback al modo "primeras ``num_classes`` synsets
      alfabéticas".

    Args:
        root: carpeta raíz.
        image_size: lado de imagen tras crop.
        batch_size: tamaño de lote.
        num_workers: workers de DataLoader.
        num_classes: número de clases a usar (fallback sin wnids).
        train_subdir: nombre del subdirectorio de train.
        val_subdir: nombre del subdirectorio de val.
        wnids: lista opcional de synsets (sobreescribe ``num_classes``).

    Returns:
        Par (train_loader, val_loader).
    """
    base = Path(root)
    tr = datasets.ImageFolder(
        base / train_subdir,
        transform=_build_transforms(image_size, True),
    )
    va = datasets.ImageFolder(
        base / val_subdir,
        transform=_build_transforms(image_size, False),
    )
    tr_sub = _select_subset(
        tr, num_classes, wnids=wnids, max_per_class=max_per_class,
    )
    va_sub = _select_subset(
        va, num_classes, wnids=wnids, max_per_class=max_per_class,
    )
    return (
        DataLoader(
            tr_sub,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0,
        ),
        DataLoader(
            va_sub,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        ),
    )


def build_imagenet_sanity_loaders(
    root: str,
    val_subdir: str = "val_blurred",
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 8,
    num_classes: int = 100,
    val_fraction: float = 0.1,
    seed: int = 42,
    wnids: list[str] | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Reparte un único split (val_blurred) en train/val para sanity.

    Mientras no esté disponible ``train_blurred``, este loader permite
    comprobar que el pipeline arranca usando un porcentaje
    ``val_fraction`` de cada clase como validación y el resto como
    entrenamiento. No sustituye a la corrida real: el modelo verá las
    mismas muestras en train que vio antes en val.

    Args:
        root: carpeta raíz del dataset.
        val_subdir: subcarpeta con la estructura ImageFolder.
        image_size: lado de imagen tras crop.
        batch_size: tamaño de lote.
        num_workers: workers de DataLoader.
        num_classes: número de clases (primeras N synsets).
        val_fraction: fracción reservada a validación por clase.
        seed: semilla de la partición estratificada.

    Returns:
        Par (train_loader, val_loader).
    """
    base = Path(root)
    tr_full = datasets.ImageFolder(
        base / val_subdir,
        transform=_build_transforms(image_size, True),
    )
    va_full = datasets.ImageFolder(
        base / val_subdir,
        transform=_build_transforms(image_size, False),
    )
    # se filtran las synsets según wnids (con remap) o las primeras N
    if wnids is not None:
        missing = [w for w in wnids if w not in tr_full.class_to_idx]
        if missing:
            raise ValueError(
                f"synsets faltantes: {missing[:5]}..."
            )
        label_map = {
            tr_full.class_to_idx[w]: new
            for new, w in enumerate(wnids)
        }
        keep = set(label_map)
    else:
        label_map = None
        keep = set(range(num_classes))

    by_cls: dict[int, list[int]] = {}
    for i, (_, c) in enumerate(tr_full.samples):
        if c in keep:
            by_cls.setdefault(c, []).append(i)

    gen = torch.Generator().manual_seed(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for c, idxs in by_cls.items():
        perm = torch.randperm(len(idxs), generator=gen).tolist()
        n_val = max(1, int(round(len(idxs) * val_fraction)))
        for p in perm[:n_val]:
            val_idx.append(idxs[p])
        for p in perm[n_val:]:
            train_idx.append(idxs[p])

    if label_map is not None:
        tr_sub = _RemappedSubset(tr_full, train_idx, label_map)
        va_sub = _RemappedSubset(va_full, val_idx, label_map)
    else:
        tr_sub = Subset(tr_full, train_idx)
        va_sub = Subset(va_full, val_idx)

    return (
        DataLoader(
            tr_sub,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0,
        ),
        DataLoader(
            va_sub,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        ),
    )
