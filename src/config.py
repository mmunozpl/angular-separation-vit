"""Cargador único de configuraciones YAML."""

from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    """Lee un archivo YAML y devuelve su contenido.

    Args:
        path: ruta al fichero .yaml.

    Returns:
        Diccionario con los parámetros cargados.

    Raises:
        ValueError: si el contenido del fichero no es un diccionario.
    """
    # se centraliza la lectura para tener un único punto de cambio
    with Path(path).open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config inválida en {path}: se esperaba dict")
    return cfg
