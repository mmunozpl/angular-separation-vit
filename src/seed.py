"""Semilla global reproducible para todo el proyecto."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fija las semillas de aleatoriedad para reproducibilidad.

    Args:
        seed: entero no negativo aplicado a random, numpy y torch
            (cpu y cuda). Activa además modo determinista en cudnn.
    """
    # se cubren todas las fuentes habituales de aleatoriedad
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # modo determinista donde sea viable
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
