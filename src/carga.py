"""carga de modelos base afinados y del conjunto de probing congelado.

helpers compartidos por los scripts de diagnóstico (fase g, fase 0):
construye la columna vit-b/16, le inyecta los pesos de un checkpoint de
contribución a y arma el loader sobre el probe set de 1000 imágenes.
porta los loaders probados del v3 (probing_lineal) a un módulo kept.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.models.vit_backbone import HeadProjections

_NORM = transforms.Normalize(
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
)

# arquitecturas soportadas por los scripts de señal; la etiqueta casa
# con el subárbol de checkpoints (<arch>_clean) y la columna arch del
# csv. img_size = resolución NATIVA del config del modelo (dinov2 no
# se fuerza a 224); crop_pct/interp siguen su config timm. probe =
# conjunto congelado propio de la columna.
ARQUITECTURAS = {
    "vitb": {
        "model_name": "vit_base_patch16_224",
        "n_cabezas": 12,
        "dim_cabeza": 64,
        "img_size": 224,
        "crop_pct": 0.875,
        "interp": "bilinear",
        "probe": "artifacts/probe_set/imagenet100_val_1k.pt",
    },
    "vitl": {
        "model_name": "vit_large_patch16_224",
        "n_cabezas": 16,
        "dim_cabeza": 64,
        "img_size": 224,
        "crop_pct": 0.875,
        "interp": "bilinear",
        "probe": "artifacts/probe_set/imagenet100_val_1k.pt",
    },
    # ssl congelado a escala vit-l (tab:arch: varia el REGIMEN a escala
    # constante frente a vitl); columna congelada, n=1 de facto
    "dinov2": {
        "model_name": "vit_large_patch14_dinov2.lvd142m",
        "n_cabezas": 16,
        "dim_cabeza": 64,
        "img_size": 518,
        "crop_pct": 1.0,
        "interp": "bicubic",
        "probe": "artifacts/probe_set/imagenet100_val_1k_dinov2.pt",
    },
}


class _ConjuntoRutas(Dataset):
    """dataset mínimo (ruta, etiqueta) -> tensor normalizado."""

    def __init__(
        self,
        rutas: list[str],
        etiquetas: list[int],
        img_size: int = 224,
        crop_pct: float = 0.875,
        interp: str = "bilinear",
    ) -> None:
        """guarda rutas y etiquetas y fija el transform de validación.

        Args:
            rutas: rutas de las imágenes.
            etiquetas: etiqueta entera por imagen.
            img_size: lado del crop final (resolución del modelo).
            crop_pct: fracción del resize previa al crop (config timm).
            interp: interpolación del resize (bilinear o bicubic).
        """
        self.rutas = rutas
        self.etiquetas = etiquetas
        modo = {
            "bilinear": transforms.InterpolationMode.BILINEAR,
            "bicubic": transforms.InterpolationMode.BICUBIC,
        }[interp]
        self.tfm = transforms.Compose([
            transforms.Resize(round(img_size / crop_pct),
                              interpolation=modo),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            _NORM,
        ])

    def __len__(self) -> int:
        """tamaño del conjunto."""
        return len(self.rutas)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """(imagen normalizada, etiqueta) del índice."""
        img = Image.open(self.rutas[idx]).convert("RGB")
        return self.tfm(img), int(self.etiquetas[idx])


def cargar_probe_loader(
    probe_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: int = 224,
    crop_pct: float = 0.875,
    interp: str = "bilinear",
) -> tuple[DataLoader, np.ndarray]:
    """loader sobre el conjunto de probing congelado (1000 imgs val).

    args:
        probe_path: ruta al .pt del probe set (paths, labels).
        batch_size: tamaño de lote.
        num_workers: procesos de carga.
        img_size: resolución del modelo (224 o la nativa del config).
        crop_pct: fracción del resize previa al crop.
        interp: interpolación del resize.

    returns:
        tupla (loader sin barajar, etiquetas como array).
    """
    blob = torch.load(probe_path, weights_only=False)
    ds = _ConjuntoRutas(blob["paths"], blob["labels"].tolist(),
                        img_size=img_size, crop_pct=crop_pct,
                        interp=interp)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return loader, blob["labels"].numpy()


def cargar_probe_tensor(
    probe_path: str,
    max_imgs: int | None = None,
    num_workers: int = 4,
    img_size: int = 224,
    crop_pct: float = 0.875,
    interp: str = "bilinear",
) -> torch.Tensor:
    """materializa las imágenes del probe set en un tensor de lote.

    el probe set guarda rutas, no imágenes; este helper las decodifica y
    normaliza una vez, para los scripts que esperan un tensor de lote.

    args:
        probe_path: ruta al .pt del probe set (paths, labels).
        max_imgs: si se da, limita a las primeras n imágenes.
        num_workers: procesos de carga.
        img_size: resolución del modelo (224 o la nativa del config).
        crop_pct: fracción del resize previa al crop.
        interp: interpolación del resize.

    returns:
        tensor [n, 3, img_size, img_size] en cpu, normalizado.
    """
    blob = torch.load(probe_path, weights_only=False)
    rutas = blob["paths"]
    etiq = blob["labels"].tolist()
    if max_imgs is not None:
        rutas, etiq = rutas[:max_imgs], etiq[:max_imgs]
    ds = _ConjuntoRutas(rutas, etiq, img_size=img_size,
                        crop_pct=crop_pct, interp=interp)
    loader = DataLoader(
        ds, batch_size=16, shuffle=False, num_workers=num_workers)
    return torch.cat([x for x, _ in loader])


def cargar_vit_base(
    ckpt_path: str,
    device: str = "cuda",
    num_classes: int = 100,
    model_name: str = "vit_base_patch16_224",
    img_size: int = 224,
) -> torch.nn.Module:
    """carga la columna vit con los pesos de un checkpoint de a.

    devuelve el vit interno de timm ---el que expone .blocks---, listo
    para los ganchos y los diagnósticos de circuito. los pesos del
    checkpoint llevan el prefijo 'backbone.'; se retira una vez para
    casar con la state_dict de la columna.

    args:
        ckpt_path: ruta al checkpoint (.pt con clave 'model').
        device: cuda o cpu.
        num_classes: clases de la cabeza (imagenet-100 -> 100).
        model_name: identificador timm de la columna.

    returns:
        el vit interno de timm en modo eval, sobre device.
    """
    hp = HeadProjections(
        model_name=model_name,
        pretrained=False, num_classes=num_classes, img_size=img_size,
    ).to(device).eval()
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = blob.get("model", blob)
    sd = {k.replace("backbone.", "", 1): v for k, v in sd.items()}
    ms = hp.state_dict()
    sd = {k: v for k, v in sd.items()
          if not (k in ms and ms[k].shape != v.shape)}
    miss, unexp = hp.load_state_dict(sd, strict=False)
    print(f"[modelo] {Path(ckpt_path).name} "
          f"(faltan={len(miss)} sobran={len(unexp)})", flush=True)
    return hp.model
