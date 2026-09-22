"""b2 — construye la sonda anidada de orden rotado (preregistro).

la rejilla P ∈ {250, 500, 1000, 2000, 5000} exige un orden único del
que cada P sea prefijo. el preregistro
(`Paper_X/prereg_B2_sonda.md`, 26-08, nota de diseño 27-08) lo fija:
rotación por clases ---las posiciones ciclan por las cien clases, con
la permutación de clases y el orden intra-clase sorteados con semilla
2026--- y las diez imágenes de la sonda publicada primero dentro de
cada clase, de modo que el prefijo de mil sea exactamente la sonda
existente y ningún P menor toque imágenes nuevas.

así todo prefijo hereda cuasi-estratificación por construcción: a
P=250, tres imágenes en cincuenta clases y dos en las otras
cincuenta ---qué cincuenta lo decide la semilla, no el script---; a
P=500, cinco por clase exactas; a P=2000 y 5000, veinte y cincuenta.

uso:
    python scripts/build_probe_anidado.py
"""

import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

ORIGEN = "artifacts/probe_set/imagenet100_val_1k.pt"
SALIDA = "artifacts/probe_set/imagenet100_val_5k_rotado.pt"
SEMILLA = 2026        # escrita antes de correr (preregistro)
POR_CLASE = 50        # el val de imagenet-100 tiene 50 por clase
EXTS = {".JPEG", ".JPG"}


def imagenes_de(carpeta: Path) -> list[str]:
    """lista ordenada de imágenes de una carpeta de clase.

    Args:
        carpeta: directorio de la clase.

    Returns:
        lista de rutas absolutas en orden determinista.
    """
    return sorted(str(p) for p in carpeta.iterdir()
                  if p.suffix.upper() in EXTS)


def main() -> None:
    """construye el orden rotado y lo serializa junto a su rejilla."""
    blob = torch.load(ORIGEN, weights_only=False)
    wnids, rutas0 = blob["wnids"], blob["paths"]
    etiq0 = blob["labels"].tolist()
    raiz = Path(rutas0[0]).parent.parent
    # se agrupan las originales por clase, en el orden en que se
    # publicaron; nada de lo que sigue las reordena entre clases
    orig: dict[int, list[str]] = {c: [] for c in range(len(wnids))}
    for ruta, c in zip(rutas0, etiq0):
        orig[c].append(ruta)

    rng = random.Random(SEMILLA)
    orden_clases = rng.sample(range(len(wnids)), len(wnids))
    por_clase: dict[int, list[str]] = {}
    for c, wnid in enumerate(wnids):
        todas = imagenes_de(raiz / wnid)
        if len(todas) < POR_CLASE:
            raise SystemExit(f"clase {wnid}: {len(todas)} imágenes")
        faltan = [r for r in todas if r not in set(orig[c])]
        if len(orig[c]) + len(faltan) < POR_CLASE:
            raise SystemExit(f"clase {wnid}: sin extensión suficiente")
        cabeza = orig[c][:]        # las diez publicadas, primero
        rng.shuffle(cabeza)
        cola = faltan[:]
        rng.shuffle(cola)
        por_clase[c] = (cabeza + cola)[:POR_CLASE]

    rutas: list[str] = []
    etiq: list[int] = []
    for k in range(POR_CLASE * len(wnids)):
        c = orden_clases[k % len(wnids)]
        rutas.append(por_clase[c][k // len(wnids)])
        etiq.append(c)

    # el prefijo de mil ha de ser la sonda publicada, sin excepción
    assert set(rutas[:1000]) == set(rutas0), "el prefijo no anida"
    assert len(set(rutas)) == len(rutas), "rutas repetidas"

    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    torch.save({"paths": rutas,
                "labels": torch.tensor(etiq, dtype=torch.int64),
                "wnids": wnids, "seed": SEMILLA,
                "n_per_class": POR_CLASE,
                "orden_clases": orden_clases,
                "origen": ORIGEN}, SALIDA)
    print(f"[guardado] {SALIDA} ({len(rutas)} rutas)")

    # composición por prefijo: se comprueba lo que el preregistro dijo
    print("\n    P | clases | imgs por clase (mín-máx)")
    for p in (250, 500, 1000, 2000, 5000):
        cuenta: dict[int, int] = {}
        for c in etiq[:p]:
            cuenta[c] = cuenta.get(c, 0) + 1
        print(f"{p:>5} | {len(cuenta):>6} | "
              f"{min(cuenta.values())}-{max(cuenta.values())}")

    # 15 observaciones aleatorias del artefacto
    for i in sorted(random.Random(0).sample(range(len(rutas)), 15)):
        print(f"  [{i:>4}] clase={etiq[i]:>3} {wnids[etiq[i]]}  "
              f"{Path(rutas[i]).name}")


if __name__ == "__main__":
    main()
