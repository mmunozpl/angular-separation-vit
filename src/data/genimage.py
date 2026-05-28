"""Loader de GenImage con split leave-one-generator-out."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms


_IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _tfm(size: int, train: bool):
    """Transformación estándar para imágenes sintéticas."""
    norm = transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
    )
    pad = size + 32
    if train:
        return transforms.Compose([
            transforms.Resize(pad),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize(pad),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        norm,
    ])


def _collect_images(root: Path) -> list[Path]:
    """Recorre recursivamente y filtra extensiones de imagen."""
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.suffix.lower() in _IMG_EXT]


def count_images(root: str | Path) -> int:
    """Cuenta imágenes bajo una carpeta (recursivo).

    Args:
        root: carpeta a contar.

    Returns:
        Número de ficheros con extensión de imagen; 0 si no existe.
    """
    return len(_collect_images(Path(root)))


def verify_structure(
    root: str,
    generators: list[str],
    real_subdir: str = "nature",
    train_split: str = "train",
    val_split: str = "val",
) -> dict[str, dict[str, int]]:
    """Cuenta ai/nature por generador y split (chequeo previo B0).

    Recorre la estructura esperada
    ``<root>/<gen>/<train|val>/<ai|nature>`` y devuelve los conteos.
    No aborta: el llamante decide qué hacer con las carpetas a 0.

    Args:
        root: raíz de GenImage.
        generators: lista de generadores a comprobar.
        real_subdir: nombre de la carpeta de reales.
        train_split: subcarpeta de train.
        val_split: subcarpeta de val.

    Returns:
        Dict ``gen -> {train_ai, train_real, val_ai, val_real}``.
    """
    base = Path(root)
    out: dict[str, dict[str, int]] = {}
    for g in generators:
        out[g] = {
            "train_ai": count_images(base / g / train_split / "ai"),
            "train_real": count_images(
                base / g / train_split / real_subdir
            ),
            "val_ai": count_images(base / g / val_split / "ai"),
            "val_real": count_images(
                base / g / val_split / real_subdir
            ),
        }
    return out


def _folder_nonempty(d: Path) -> bool:
    """True si la carpeta existe y contiene al menos un fichero."""
    return d.exists() and any(d.iterdir())


def assert_splits_extracted(
    root: str,
    generators: list[str],
    held_out: str,
    real_subdir: str = "nature",
    train_split: str = "train",
    val_split: str = "val",
) -> None:
    """Verifica en disco que están las carpetas que el fold necesita.

    Los loaders leen de disco (no del CSV); antes de construirlos se
    confirma que cada generador requerido por este fold está extraído,
    y se aborta con un mensaje claro si falta alguno (en vez de fallar
    a mitad de época). El generador excluido no necesita su ``train/ai``
    pero sí el resto (su ``val/ai`` se usa como OOD y sus reales como
    clase real).

    Args:
        root: raíz de GenImage.
        generators: lista de generadores del barrido.
        held_out: generador excluido de este fold.
        real_subdir: carpeta de reales.
        train_split: subcarpeta de train.
        val_split: subcarpeta de val.

    Raises:
        ValueError: si falta alguna carpeta requerida (no extraída).
    """
    base = Path(root)
    required: list[tuple[str, str, str]] = []
    for g in generators:
        if g != held_out:
            required.append((g, train_split, "ai"))
        required.append((g, train_split, real_subdir))
        required.append((g, val_split, "ai"))
        required.append((g, val_split, real_subdir))
    missing = [
        f"{g}/{s}/{k}" for (g, s, k) in required
        if not _folder_nonempty(base / g / s / k)
    ]
    if missing:
        shown = ", ".join(missing[:12])
        extra = " …" if len(missing) > 12 else ""
        raise ValueError(
            f"GenImage incompleto para el fold held_out={held_out}: "
            f"faltan {len(missing)} carpetas extraídas "
            f"[{shown}{extra}]. Espera a que termine la extracción."
        )


class GenImageFolder(Dataset):
    """Imágenes de una carpeta con una etiqueta común.

    Es la unidad mínima del LOGO: una carpeta de imágenes (ai o
    nature) con su etiqueta asociada.
    """

    def __init__(
        self,
        path: str | Path,
        label: int,
        image_size: int,
        train: bool,
        max_per_folder: int | None = None,
        seed: int = 42,
    ) -> None:
        """Inicializa el dataset.

        Args:
            path: carpeta con imágenes.
            label: etiqueta común a todas.
            image_size: lado de imagen.
            train: si aplica augmentation.
            max_per_folder: si se indica, submuestrea a ese número de
                imágenes (para smoke); muestreo determinista por
                ``seed`` sobre la lista ordenada.
            seed: semilla del submuestreo.
        """
        self.path = Path(path)
        files = sorted(_collect_images(self.path))
        if max_per_folder is not None and len(files) > max_per_folder:
            import random
            files = random.Random(seed).sample(files, max_per_folder)
        self.files = files
        self.label = int(label)
        self.tfm = _tfm(image_size, train)

    def __len__(self) -> int:
        """Número de muestras."""
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Devuelve la imagen y su etiqueta."""
        img = Image.open(self.files[idx]).convert("RGB")
        return self.tfm(img), self.label


def build_logo_split(
    root: str,
    generators: list[str],
    held_out: str,
    image_size: int,
    batch_size: int,
    num_workers: int = 8,
    real_subdir: str = "nature",
    train_split: str = "train",
    val_split: str = "val",
    n_train_per_gen: int | None = None,
    n_val_per_gen: int | None = None,
    sample_seed: int = 1234,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    """Construye loaders en régimen leave-one-generator-out.

    La estructura esperada es ``<root>/<gen>/<train|val>/<ai|nature>``.
    El entrenamiento concatena los generadores no excluidos + reales.
    La validación incluye también el generador excluido para evaluar
    transferencia.

    Sobre los reales (auditoría #7, Fase B0): la sospecha era que las
    ``nature`` estuvieran duplicadas entre generadores, lo que al
    concatenarlas inflaría la clase real ×M. La verificación sobre
    ``genimage_metadata.csv`` (ver ``scripts/verify_genimage_b0.py``)
    lo descarta: cada imagen real lleva su nombre ImageNet original y
    aparece en un solo generador —los conjuntos son disjuntos—. Por
    eso se concatenan todos: el balance train queda en ~0,99:1
    (nature:ai), equilibrado. Deduplicar a un generador dejaría las
    reales en ~1:M.

    Args:
        root: raíz de GenImage.
        generators: lista de M generadores.
        held_out: el generador excluido del entrenamiento.
        image_size: lado de imagen.
        batch_size: tamaño de lote.
        num_workers: workers torch.
        real_subdir: subcarpeta de imágenes reales (``nature`` o
            ``real``).
        train_split: subcarpeta de train por generador.
        val_split: subcarpeta de val por generador.
        n_train_per_gen: límite de imágenes por carpeta de TRAIN
            (protocolo eval B, cambio 2); None usa todas.
        n_val_per_gen: límite por carpeta de VAL; por defecto None =
            val estándar completa (cambio 1). Solo se acota para
            acelerar smokes, nunca en la evaluación real.
        sample_seed: semilla del submuestreo, FIJA e independiente de
            la semilla de entrenamiento, para que todas las semillas
            de entrenamiento vean el mismo subconjunto.

    Returns:
        Tupla (train_loader, val_loader, label_map). El mapa
        asocia ``real`` -> 0 y cada generador -> 1..M.
    """
    assert held_out in generators, (
        f"{held_out} no está en la lista de generadores"
    )
    # los loaders leen de disco: se verifica que lo que el fold
    # necesita está extraído antes de construirlos
    assert_splits_extracted(
        root, generators, held_out,
        real_subdir=real_subdir,
        train_split=train_split, val_split=val_split,
    )
    base = Path(root)
    label_map: dict[str, int] = {"real": 0}
    train_sets: list[Dataset] = []
    for i, g in enumerate(generators, start=1):
        label_map[g] = i
        if g == held_out:
            continue
        train_sets.append(GenImageFolder(
            base / g / train_split / "ai",
            label=i, image_size=image_size, train=True,
            max_per_folder=n_train_per_gen, seed=sample_seed,
        ))
    # reales: disjuntas entre generadores (verificado), se concatenan
    for g in generators:
        d = base / g / train_split / real_subdir
        if d.exists():
            train_sets.append(GenImageFolder(
                d, label=0, image_size=image_size, train=True,
                max_per_folder=n_train_per_gen, seed=sample_seed,
            ))
    train_ds = ConcatDataset(train_sets)

    # val: estándar completa por defecto (cambio 1); n_val_per_gen
    # solo para acelerar smokes
    val_sets: list[Dataset] = []
    for g in generators:
        val_sets.append(GenImageFolder(
            base / g / val_split / "ai",
            label=label_map[g],
            image_size=image_size, train=False,
            max_per_folder=n_val_per_gen, seed=sample_seed,
        ))
    for g in generators:
        d = base / g / val_split / real_subdir
        if d.exists():
            val_sets.append(GenImageFolder(
                d, label=0, image_size=image_size, train=False,
                max_per_folder=n_val_per_gen, seed=sample_seed,
            ))
    val_ds = ConcatDataset(val_sets)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, label_map
