"""Confirmatorio F preregistrado: Δs_func en capa 11 (paired, una cola).

Recalcula ``s_func`` por capa sobre el **probe set congelado** (1000
imágenes) para las 5 semillas base y las 5 blanda, y ejecuta el test
preregistrado (pipeline_v3.md §4.A): paired t-test de una cola
(dirección Δ<0) en capa 11, con criterio conjunto p<0,05 **y**
|d_Cohen|>0,5 sobre la varianza de los Δ paired.

La métrica es idéntica a la preregistrada: ``mean_upper_offdiag`` de
``functional_similarity_matrix`` (similitud coseno entre mapas de
atención promedio por par de cabezas no ordenado, agregada por capa).
La única diferencia con el ``layerwise.csv`` de entrenamiento es la
población (probe set limpio vs capturas de val), que es el punto del
recálculo.

Aplica la tabla de decisión vinculante (pipeline_v3.md §4.A, fechada
2026-06-05) y NO interpreta el número: el veredicto sale de la regla.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import torch
from PIL import Image
from scipy import stats as scistats
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.attention import (
    functional_similarity_matrix,
    mean_upper_offdiag,
)
from src.models.vit_backbone import HeadProjections, capture_attention

_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
)


class _PathDataset(torch.utils.data.Dataset):
    """Lee paths del probe set y devuelve tensores normalizados."""

    def __init__(self, paths: list[str], image_size: int) -> None:
        """Inicializa con la lista de paths y el tamaño de imagen."""
        self.paths = paths
        self.tfm = transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            _NORMALIZE,
        ])

    def __len__(self) -> int:
        """Número de imágenes del probe set."""
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Carga y normaliza una imagen."""
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tfm(img)


def _avg_attention(
    model: HeadProjections, loader: DataLoader, device: str,
) -> torch.Tensor:
    """Mapa de atención promediado sobre todo el probe set.

    Returns:
        Tensor (L, H, T, T) con la atención media tras softmax.
    """
    acc: torch.Tensor | None = None
    n_imgs = 0
    base = model.model
    with capture_attention(base) as maps:
        for imgs in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.no_grad():
                _ = model(imgs)
            for li, m in enumerate(maps):
                if acc is None:
                    n_layers = len(maps)
                    shape = (n_layers,) + tuple(m.shape[1:])
                    acc = torch.zeros(shape, device=device)
                acc[li] += m.sum(dim=0)
            n_imgs += imgs.shape[0]
            maps.clear()
    return acc / float(n_imgs)


def _sfunc_per_layer(attn_avg: torch.Tensor) -> torch.Tensor:
    """s_func por capa = media de pares no ordenados de cos entre mapas.

    Args:
        attn_avg: tensor (L, H, T, T) promediado sobre el probe set.

    Returns:
        Tensor (L,) con s_func por capa.
    """
    n_layers = attn_avg.shape[0]
    return torch.stack([
        mean_upper_offdiag(functional_similarity_matrix(attn_avg[l]))
        for l in range(n_layers)
    ])


def _load_model(
    ckpt_path: Path, model: HeadProjections, device: str,
) -> None:
    """Carga los pesos de un ckpt de A sobre el modelo dado."""
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = blob.get("model", blob)
    sd = {k.replace("backbone.", "", 1): v for k, v in sd.items()}
    ms = model.state_dict()
    sd = {k: v for k, v in sd.items()
          if not (k in ms and ms[k].shape != v.shape)}
    model.load_state_dict(sd, strict=False)


def _paired_stats(
    deltas: list[float],
) -> tuple[float, float, float, float]:
    """Media, std muestral, Cohen's d pareado y t de los Δ paired.

    Args:
        deltas: lista de Δ = s_func_blanda − s_func_base por semilla.

    Returns:
        (media, std_muestral, d_paired, t_stat).
    """
    n = len(deltas)
    m = sum(deltas) / n
    sd = (sum((x - m) ** 2 for x in deltas) / (n - 1)) ** 0.5
    d = m / sd if sd > 0 else float("nan")
    t = m / (sd / math.sqrt(n)) if sd > 0 else float("nan")
    return m, sd, d, t


def main() -> None:
    """Ejecuta el confirmatorio y aplica la tabla de decisión."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-set",
        default="artifacts/probe_set/imagenet100_val_1k.pt",
    )
    parser.add_argument("--ckpt-dir", default="artifacts/checkpoints/A")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
    )
    parser.add_argument("--target-layer", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out-dir", default="artifacts/tables/confirmatory_sfunc",
    )
    args = parser.parse_args()

    # comprueba que existen los 10 ckpts antes de empezar
    needed = []
    for cfg in ("base", "blanda"):
        for s in args.seeds:
            p = Path(args.ckpt_dir) / f"attnA_{cfg}_seed{s}_last.pt"
            if not p.exists():
                needed.append(str(p))
    if needed:
        print("[abort] faltan ckpts:\n  " + "\n  ".join(needed),
              file=sys.stderr)
        sys.exit(1)

    probe = torch.load(args.probe_set, weights_only=False)
    paths = probe["paths"]
    print(f"[probe_set] n={len(paths)} imgs")
    ds = _PathDataset(paths, image_size=224)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    model = HeadProjections(
        model_name="vit_base_patch16_224",
        pretrained=False, num_classes=100, img_size=224,
    ).to(args.device).eval()

    # s_func[cfg][seed] = tensor(L)
    sfunc: dict[str, dict[int, torch.Tensor]] = {"base": {}, "blanda": {}}
    for cfg in ("base", "blanda"):
        for s in args.seeds:
            ckpt = Path(args.ckpt_dir) / f"attnA_{cfg}_seed{s}_last.pt"
            _load_model(ckpt, model, args.device)
            attn_avg = _avg_attention(model, loader, args.device)
            sfunc[cfg][s] = _sfunc_per_layer(attn_avg).cpu()
            print(f"  {cfg} seed{s}: s_func capa "
                  f"{args.target_layer} = "
                  f"{sfunc[cfg][s][args.target_layer]:.4f}", flush=True)

    # validación de instrumento POR PATRÓN, no por valor absoluto: el
    # valor puede shiftar por población (probe set vs capturas de val),
    # que es el punto del recálculo. lo que NO debe cambiar es el patrón
    # relativo del layerwise n=5: seed42 la más alta, seed46 la más
    # baja, std ~0,034. si el probe set rompe el orden o colapsa la
    # dispersión, sospechar del script, no del dato. mirar ESTO antes
    # del veredicto.
    lt = args.target_layer
    base_lt = {s: float(sfunc["base"][s][lt]) for s in args.seeds}
    vals = list(base_lt.values())
    mb_v = sum(vals) / len(vals)
    sd_v = (sum((x - mb_v) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
    smax = max(base_lt, key=base_lt.get)
    smin = min(base_lt, key=base_lt.get)
    print(f"\n[instrumento] capa {lt} base: " +
          ", ".join(f"s{s}={base_lt[s]:.4f}" for s in args.seeds))
    ok_max, ok_min = smax == 42, smin == 46
    ok_disp = 0.015 <= sd_v <= 0.055
    print(f"  patrón esperado (layerwise): max=seed42, min=seed46, "
          f"std~0,034")
    print(f"  observado: max=seed{smax} [{'OK' if ok_max else 'WARN'}]"
          f"  min=seed{smin} [{'OK' if ok_min else 'WARN'}]"
          f"  std={sd_v:.4f} [{'OK' if ok_disp else 'WARN'}]")
    if not (ok_max and ok_min and ok_disp):
        print("  [WARN] el patrón NO se reproduce: revisar el script "
              "antes de confiar en el veredicto.")
    else:
        print("  [OK] patrón reproducido; instrumento validado.")

    n_layers = sfunc["base"][args.seeds[0]].shape[0]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n=== s_func por capa (probe set congelado, n=5) ===")
    hdr = (f"{'capa':>4} {'base m±sd':>16} {'blanda m±sd':>16} "
           f"{'Δpair m±sd':>18} {'d_pair':>7} {'t':>7} {'p_1cola':>8}")
    print(hdr)
    rows = []
    conf = {}
    for cap in range(n_layers):
        b = [float(sfunc["base"][s][cap]) for s in args.seeds]
        bl = [float(sfunc["blanda"][s][cap]) for s in args.seeds]
        delt = [bl[i] - b[i] for i in range(len(args.seeds))]
        mb = sum(b) / len(b)
        sb = (sum((x - mb) ** 2 for x in b) / (len(b) - 1)) ** 0.5
        mbl = sum(bl) / len(bl)
        sbl = (sum((x - mbl) ** 2 for x in bl) / (len(bl) - 1)) ** 0.5
        md, sd, d, t = _paired_stats(delt)
        # p de una cola (dirección Δ<0) vía CDF de la t de Student:
        # rechazar H1 exige t suficientemente negativo (cola izquierda)
        p1 = float(scistats.t.cdf(t, df=len(args.seeds) - 1))
        mark = "  <-- CONF" if cap == args.target_layer else ""
        print(f"{cap:>4} {mb:7.4f}±{sb:.4f} {mbl:7.4f}±{sbl:.4f} "
              f"{md:+7.4f}±{sd:.4f} {d:+6.2f} {t:+6.2f} {p1:8.4f}{mark}")
        rows.append({
            "capa": cap, "base_mean": mb, "base_std": sb,
            "blanda_mean": mbl, "blanda_std": sbl,
            "delta_mean": md, "delta_std": sd,
            "d_paired": d, "t": t, "p_one_tailed": p1,
        })
        if cap == args.target_layer:
            conf = {"d": d, "p": p1, "delta_mean": md, "t": t,
                    "deltas": delt}

    with (out / "sfunc_per_layer.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- tabla de decisión vinculante (pipeline_v3.md §4.A, 06-05) ---
    # el test preregistrado es de UNA COLA (dirección Δ<0): tiene
    # exactamente DOS desenlaces, rechaza H1 o no. la "dirección
    # opuesta" (§10.1 c) NO es una rama de este test —es un flag
    # descriptivo sobre el signo de Δ; el agregado significativo lo
    # evaluaría el bloque exploratorio aparte, no este confirmatorio—.
    d, p, dm = conf["d"], conf["p"], conf["delta_mean"]
    rechaza_h1 = (p < 0.05) and (abs(d) > 0.5) and (dm < 0.0)
    print(f"\n=== veredicto capa {args.target_layer} (regla en ciego) ===")
    # cinco Δ individuales (signo por semilla) para la lectura
    # descriptiva: mismo signo = efecto consistente pequeño; repartido
    # = fluctuación sin dirección. no cambia el veredicto del test.
    print("Δ por semilla: " + ", ".join(
        f"s{args.seeds[i]}={conf['deltas'][i]:+.4f}"
        for i in range(len(args.seeds))))
    print(f"Δpair_media={dm:+.4f}  d_paired={d:+.3f}  p_1cola={p:.4f}  "
          f"(criterio: p<0,05 Y |d|>0,5 Y Δ<0)")
    if rechaza_h1:
        print("F RECHAZA H1 -> §10.1(a): reducción funcional "
              "localizada en capa 11 real; afirmación v2 confirmada, "
              "refuerza (no sustituye) el desacople-D.")
    else:
        print("F NO RECHAZA H1 -> la afirmación v2 de reducción "
              "localizada NO se sostiene, se RETIRA; el desacople de "
              "v3 se evalúa sobre D/E/G. NO se escribe 'F confirma "
              "desacople' (un test no confirma la nula).")
    # flag descriptivo de signo — NO una rama del test de una cola
    if dm > 0.0:
        print(f"  [nota descriptiva] Δ={dm:+.4f} > 0: la función SUBE "
              "levemente al regularizar, no baja. Si el exploratorio "
              "del agregado saliera significativo sería territorio "
              "§10.1(c); el confirmatorio de una cola Δ<0 no lo testa.")


if __name__ == "__main__":
    main()
