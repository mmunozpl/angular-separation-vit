"""prepara el árbol del Space y firma lo que se vende.

el Space se lleva copias de `gauge_flip`, `nucleo_lectura`,
`firma_funcional` y el portador, y las copias derivan. el manifiesto
aplica al código el patrón que g4 aplica a los pdf: sha256 de cada
fichero vendido, contrastado en el arranque. si una copia cambió, el
Space **no arranca** en vez de servir derivas de un código que ya no
es el del paper.

los ficheros se extraen del **respaldo commiteado**, no del directorio
de trabajo, para que la procedencia del hash sea un commit y no un
estado a medio editar.

uso:
    python scripts/desplegar_demo.py --salida /ruta/al/space
"""

import argparse
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys

RESPALDO = "/home/manpla/.respaldo_hiperesferas.git"
ARBOL = "/media/manpla/Pruebas/Hiperesferas"

# origen en el repo -> destino en el Space. app.py va a la raíz porque
# el sdk de gradio de hf lo exige ahí
VENDIDOS = {
    "demo/app.py": "app.py",
    "demo/portador.py": "demo/portador.py",
    "demo/requirements.txt": "requirements.txt",
    "src/gauge_flip.py": "src/gauge_flip.py",
    "src/nucleo_lectura.py": "src/nucleo_lectura.py",
    "src/firma_funcional.py": "src/firma_funcional.py",
}
SECTORES = ["artifacts/demo/sector_vo_vitb.safetensors",
            "artifacts/demo/sector_vo_pythia.safetensors"]

CARD = """---
title: The value-output gauge orbit
emoji: 🧭
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: false
---

# The value-output gauge orbit

Interactive demo. The dominant direction of a head's output projection
moves under a reparameterisation that leaves the model's function
untouched, and the pruning decision it yields moves with it. The
identifiable object ---the OV circuit--- does not move.

Two models, identical in function, two different prunings. English and
Spanish; ViT-B/16 and Pythia-410M.

DOI: https://doi.org/10.5281/zenodo.21630534

---

# La órbita de gauge valor-salida

Demo interactiva. La dirección dominante de la proyección de salida de
una cabeza se desplaza bajo una reparametrización que deja la función
del modelo intacta, y la decisión de poda que produce cambia con ella.
El objeto identificable ---el circuito OV--- no se mueve.
"""

GITATTRIBUTES = "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"


def sha256(ruta: str) -> str:
    """hash de un fichero, por bloques.

    Args:
        ruta: ruta del fichero.

    Returns:
        el sha256 en hexadecimal.
    """
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def desde_el_respaldo(rel: str, destino: pathlib.Path) -> None:
    """extrae un fichero de la punta del respaldo, no del árbol vivo.

    Args:
        rel: ruta relativa dentro del repositorio.
        destino: fichero de salida.

    Raises:
        SystemExit: si el fichero no está en el commit.
    """
    r = subprocess.run(["git", f"--git-dir={RESPALDO}", "show",
                        f"arbol-local:{rel}"], capture_output=True)
    if r.returncode != 0:
        sys.exit(f"[error] {rel} no está en la punta del respaldo: "
                 f"{r.stderr.decode()[:120]}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(r.stdout)


def main() -> None:
    """monta el árbol del Space y escribe el manifiesto."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--salida", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.salida)
    out.mkdir(parents=True, exist_ok=True)

    manifiesto = {}
    for origen, destino in VENDIDOS.items():
        d = out / destino
        desde_el_respaldo(origen, d)
        manifiesto[destino] = sha256(str(d))
        print(f"  {origen:32s} -> {destino}")

    # los sectores son binarios grandes: se copian del árbol, y su hash
    # entra igualmente en el manifiesto
    for s in SECTORES:
        d = out / s
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(os.path.join(ARBOL, s), d)
        manifiesto[s] = sha256(str(d))
        print(f"  {s:32s} -> {s} "
              f"({os.path.getsize(d) / 1e6:.0f} MB)")

    (out / "README.md").write_text(CARD, encoding="utf-8")
    (out / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8")
    lineas = [f"{h}  {f}" for f, h in sorted(manifiesto.items())]
    (out / "manifiesto.sha256").write_text("\n".join(lineas) + "\n",
                                           encoding="utf-8")
    commit = subprocess.run(
        ["git", f"--git-dir={RESPALDO}", "rev-parse", "--short",
         "arbol-local"], capture_output=True, text=True).stdout.strip()
    print(f"\n[manifiesto] {len(manifiesto)} ficheros, "
          f"procedencia commit {commit}")
    for l in lineas:
        print(f"  {l[:12]}…  {l.split('  ', 1)[1]}")


if __name__ == "__main__":
    main()
