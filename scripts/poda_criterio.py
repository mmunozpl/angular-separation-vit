"""benchmark mínimo del instrumento: poda por criterio, sin reentrenar.

valida el INSTRUMENTO, no propone un método de poda: con el mismo
presupuesto (k=2 cabezas por capa, 24 de 144), se poda el conjunto más
redundante según tres criterios ---pesos (|cos| medio entre v1(w_o)),
firma (|cos| medio entre firmas v1(c_h^p) sobre el probe congelado) y
aleatorio (3 sorteos con semillas 42/43/44, el suelo)--- y se mide la
caída de top-1 sobre el val limpio completo (5000). la poda anula la
contribución de la cabeza al residuo poniendo a cero sus columnas en
la proyección de salida (proj.weight[:, h*dh:(h+1)*dh] = 0; el sesgo
de proj es compartido y se conserva); el mecanismo es el MISMO para
los tres criterios.

uso:
    python scripts/poda_criterio.py [--sanity] [--seeds 42 ...]

sanity: k=0 (no se poda nada) -> caída 0,000 pp en los tres criterios;
y el enmascarado de una cabeza fija repetido dos veces debe dar la
misma caída (determinista).
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_fase_G import carga_modelo, firma_exacta
from scripts.run_fase_0 import firma_computada, gram_contexto
from src.carga import cargar_probe_tensor
from src.data.imagenet import build_imagenet_loaders
from src.firma_funcional import CapturaContexto, w_o_por_cabeza

PROBE = "artifacts/probe_set/imagenet100_val_1k.pt"
SALIDA = "artifacts/logs/poda_criterio/poda_criterio.csv"
ROOT = "/media/manpla/ST2TB/kaggle/in100"
WNIDS = "configs/imagenet100_cmc.txt"
N_CABEZAS, DIM_CABEZA = 12, 64
K_PODA = 2                      # cabezas por capa (24 de 144)
SORTEOS = [42, 43, 44]          # semillas del criterio aleatorio


def val_loader_completo() -> torch.utils.data.DataLoader:
    """loader del val limpio completo (5000 imágenes)."""
    wnids = [ln.strip() for ln in Path(WNIDS).read_text().splitlines()
             if ln.strip()]
    _, val = build_imagenet_loaders(
        root=ROOT, image_size=224, batch_size=128, num_workers=8,
        num_classes=100, train_subdir="train", val_subdir="val",
        wnids=wnids)
    return val


@torch.no_grad()
def top1(modelo, loader, disp: str) -> float:
    """exactitud top-1 del modelo sobre el loader."""
    aciertos, total = 0, 0
    for imgs, lbls in tqdm(loader, desc="eval", leave=False):
        logits = modelo(imgs.to(disp, non_blocking=True))
        pred = logits.argmax(dim=1).cpu()
        aciertos += int((pred == lbls).sum())
        total += int(lbls.numel())
    return aciertos / total


def seleccion_redundante(firmas_por_capa: list[torch.Tensor],
                         k: int) -> list[tuple[int, int]]:
    """las k cabezas más redundantes por capa según unas firmas.

    Args:
        firmas_por_capa: lista de tensores [h, d], una por capa.
        k: cabezas a seleccionar por capa.

    Returns:
        lista de pares (capa, cabeza).
    """
    sel: list[tuple[int, int]] = []
    for capa, f in enumerate(firmas_por_capa):
        c = (f @ f.t()).abs().clamp(max=1.0)
        c.fill_diagonal_(0.0)
        red = c.sum(dim=1) / (f.shape[0] - 1)
        top = torch.argsort(red, descending=True)[:k]
        sel.extend((capa, int(h)) for h in top)
    return sel


def seleccion_aleatoria(n_capas: int, k: int,
                        semilla: int) -> list[tuple[int, int]]:
    """k cabezas por capa al azar, con semilla fijada."""
    rng = random.Random(semilla)
    sel: list[tuple[int, int]] = []
    for capa in range(n_capas):
        sel.extend((capa, h)
                   for h in rng.sample(range(N_CABEZAS), k))
    return sel


def podar(modelo, seleccion: list[tuple[int, int]]) -> dict:
    """anula las columnas de proj de las cabezas seleccionadas.

    Args:
        modelo: el vit, modificado in situ.
        seleccion: pares (capa, cabeza) a podar.

    Returns:
        dict capa -> copia del proj.weight original, para restaurar.
    """
    backup: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for capa, h in seleccion:
            w = modelo.blocks[capa].attn.proj.weight
            if capa not in backup:
                backup[capa] = w.detach().clone()
            w[:, h * DIM_CABEZA:(h + 1) * DIM_CABEZA] = 0.0
    return backup


def restaurar(modelo, backup: dict) -> None:
    """repone los proj.weight originales tras una poda."""
    with torch.no_grad():
        for capa, w0 in backup.items():
            modelo.blocks[capa].attn.proj.weight.copy_(w0)


def main() -> None:
    """corre el benchmark de poda por criterio y guarda el csv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 43, 44, 45, 46])
    parser.add_argument("--sanity", action="store_true",
                        help="k=0 y determinismo de una cabeza")
    args = parser.parse_args()
    k = 0 if args.sanity else K_PODA

    disp = "cuda" if torch.cuda.is_available() else "cpu"
    imgs_probe = cargar_probe_tensor(PROBE)
    val = val_loader_completo()
    filas: list[dict] = []
    for seed in tqdm(args.seeds, desc="semillas"):
        ckpt = (f"artifacts/checkpoints/vitb_clean/"
                f"attnA_base_seed{seed}_last.pt")
        modelo = carga_modelo(ckpt).to(disp).eval()
        n_capas = len(modelo.blocks)

        # firmas de los dos criterios informados
        captura = CapturaContexto(modelo, N_CABEZAS, DIM_CABEZA)
        gram = gram_contexto(modelo, imgs_probe, captura, n_capas,
                             disp)
        captura.quitar()
        v1_wo, firma = [], []
        for capa in range(n_capas):
            w_o = w_o_por_cabeza(modelo, capa, N_CABEZAS, DIM_CABEZA)
            v1_wo.append(firma_exacta(w_o, "der"))
            firma.append(firma_computada(gram[capa], w_o))

        base_top1 = top1(modelo, val, disp)
        selecciones: list[tuple[str, int, list]] = [
            ("pesos", 0, seleccion_redundante(v1_wo, k)),
            ("firma", 0, seleccion_redundante(firma, k)),
        ]
        selecciones += [
            ("aleatorio", s, seleccion_aleatoria(n_capas, k, s))
            for s in SORTEOS
        ]
        if args.sanity:
            # determinismo: la misma cabeza fija, dos veces
            selecciones += [("sanity_det", i, [(0, 0)])
                            for i in (1, 2)]
        for crit, sorteo, sel in selecciones:
            backup = podar(modelo, sel)
            podado = top1(modelo, val, disp)
            restaurar(modelo, backup)
            filas.append({
                "seed": seed, "criterio": crit, "sorteo": sorteo,
                "top1_intacto": round(base_top1, 4),
                "top1_podado": round(podado, 4),
                "caida_pp": round(100 * (base_top1 - podado), 3),
                "podadas": str(sel)})
        del modelo
        torch.cuda.empty_cache()

    out = Path(SALIDA)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        wr.writeheader()
        wr.writerows(filas)
    print(f"[guardado] {out}  ({len(filas)} filas)")

    # 15 observaciones aleatorias del artefacto
    rng = random.Random(0)
    for fila in rng.sample(filas, min(15, len(filas))):
        print({key: fila[key] for key in fila if key != "podadas"})

    # agregado para tab:poda (media±banda entre semillas; los sorteos
    # aleatorios se promedian antes, dentro de cada semilla)
    import statistics
    print("\ncriterio | caida_pp media ± banda entre semillas")
    for crit in ("pesos", "firma", "aleatorio"):
        por_semilla = []
        for seed in args.seeds:
            vals = [f["caida_pp"] for f in filas
                    if f["seed"] == seed and f["criterio"] == crit]
            if vals:
                por_semilla.append(sum(vals) / len(vals))
        if len(por_semilla) > 1:
            print(f"{crit:>9} | {statistics.mean(por_semilla):.3f} "
                  f"± {statistics.stdev(por_semilla):.3f}")


if __name__ == "__main__":
    main()
