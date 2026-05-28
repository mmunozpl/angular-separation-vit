"""Fase B0: verificación de GenImage antes de entrenar (bloqueante).

Resuelve el punto 7 de la auditoría: ¿las imágenes reales (``nature``)
están duplicadas entre generadores? Si lo están, concatenarlas de los
M generadores infla la clase real ×M y sesga cualquier AUROC. Este
script:

1. comprueba la estructura ``<root>/<gen>/<train|val>/<ai|nature>`` y
   cuenta ai/nature por generador y split;
2. compara por hash de contenido las ``nature`` de pares de
   generadores para detectar duplicación;
3. reporta el balance real/sintético con y sin deduplicación, para
   justificar ``real_dedup`` en el loader.

No entrena nada. Debe correr DESPUÉS de extraer GenImage.

Uso:
    python scripts/verify_genimage_b0.py --config \
        configs/detect_genimage.yaml
"""

import argparse
import csv
import hashlib
import os
import random
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.genimage import verify_structure

_IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _list_image_names(folder: Path) -> dict[str, Path]:
    """Mapa basename -> ruta de las imágenes de una carpeta.

    Lista el primer nivel con ``os.scandir`` (rápido); si no hay
    imágenes ahí, recae en búsqueda recursiva por si hubiera
    subcarpetas de clase.

    Args:
        folder: carpeta a listar.

    Returns:
        Dict de nombre de fichero a ruta completa; vacío si no existe.
    """
    if not folder.exists():
        return {}
    out: dict[str, Path] = {}
    with os.scandir(folder) as it:
        for e in it:
            if e.is_file() and Path(e.name).suffix.lower() in _IMG_EXT:
                out[e.name] = Path(e.path)
    if not out:
        for p in folder.rglob("*"):
            if p.suffix.lower() in _IMG_EXT:
                out[p.name] = p
    return out


def _sha1(path: Path, chunk: int = 1 << 20) -> str:
    """Hash sha1 del contenido de un fichero."""
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _load_corrupted(path: str | None) -> set[str]:
    """Carga las rutas de ``corrupted_files.txt`` (formato del CSV).

    Args:
        path: ruta al fichero de corruptos; None devuelve conjunto
            vacío.

    Returns:
        Conjunto de rutas relativas (``GenImage/.../x.PNG``) a
        descontar de la contabilidad.
    """
    if not path or not Path(path).exists():
        return set()
    return {
        ln.strip() for ln in Path(path).read_text().splitlines()
        if ln.strip()
    }


def _hash_content_check(
    root: str,
    gens: list[str],
    n_samples: int,
    real_subdir: str = "nature",
    split: str = "train",
) -> bool:
    """B0 por hash de CONTENIDO entre generadores (no por nombre).

    Como los nombres de las ``nature`` son disjuntos entre generadores,
    la duplicación solo puede detectarse comparando el contenido. Se
    hashea (sha1) una muestra de cada generador y se buscan colisiones
    de hash entre generadores distintos.

    Args:
        root: raíz de GenImage (con los generadores extraídos).
        gens: generadores a comparar (deben estar en disco).
        n_samples: nº de imágenes a hashear por generador.
        real_subdir: carpeta de reales.
        split: split del que tomar las reales.

    Returns:
        True si se detecta alguna colisión de contenido entre
        generadores (duplicación); False si todas son distintas.
    """
    base = Path(root)
    rng = random.Random(0)
    hashes: dict[str, set[str]] = {}
    for g in gens:
        names = _list_image_names(base / g / split / real_subdir)
        keys = sorted(names)
        pick = keys if len(keys) <= n_samples else rng.sample(
            keys, n_samples,
        )
        hs: set[str] = set()
        for name in tqdm(
            pick, desc=f"sha1 {g}", unit="img", leave=False,
        ):
            hs.add(_sha1(names[name]))
        hashes[g] = hs
        print(f"  {g}: {len(hs)} hashes únicos de {len(pick)} muestras")
    any_dup = False
    print("[B0-hash] intersección de contenido entre pares")
    for i in range(len(gens)):
        for j in range(i + 1, len(gens)):
            a, b = gens[i], gens[j]
            inter = hashes[a] & hashes[b]
            print(
                f"  {a} ∩ {b}: {len(inter)} colisiones de contenido",
            )
            any_dup = any_dup or bool(inter)
    return any_dup


def _logo_balances(
    counts: dict[str, dict[str, int]],
    generators: list[str],
) -> list[dict]:
    """Balance del train de cada split leave-one-out, con/sin dedup.

    Args:
        counts: conteos por generador (train_ai, train_nature, …).
        generators: generadores a considerar.

    Returns:
        Una fila por generador excluido con el balance del train.
    """
    present = [g for g in generators if g in counts]
    real_all = sum(counts[g].get("train_nature", 0) for g in present)
    ref = present[0] if present else None
    real_dedup = counts[ref].get("train_nature", 0) if ref else 0
    rows: list[dict] = []
    for g in present:
        tr_ai = sum(
            counts[x].get("train_ai", 0) for x in present if x != g
        )
        rows.append({
            "held_out": g,
            "train_ai": tr_ai,
            "real_nodedup": real_all,
            "ratio_nodedup": round(real_all / max(tr_ai, 1), 2),
            "real_dedup": real_dedup,
            "ratio_dedup": round(real_dedup / max(tr_ai, 1), 2),
        })
    return rows


def _save_csv(path: Path, rows: list[dict]) -> None:
    """Vuelca filas (lista de dicts) a CSV con cabecera."""
    if not rows:
        return
    with path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def _show_random(rows: list[dict], k: int = 15) -> None:
    """Imprime k observaciones aleatorias del artefacto guardado."""
    if not rows:
        print("  (sin filas)")
        return
    sample = rows if len(rows) <= k else random.sample(rows, k)
    for r in sample:
        print("  " + "  ".join(f"{kk}={vv}" for kk, vv in r.items()))


def _verify_from_csv(
    meta_path: str,
    generators: list[str],
    out: Path,
    corrupted: set[str],
) -> None:
    """Verificación B0 a partir de ``genimage_metadata.csv``.

    No necesita la extracción: deriva generador/split/tipo del campo
    ``path`` y detecta duplicación de reales comparando el nombre de
    fichero (las ``nature`` llevan su nombre ImageNet original, único)
    entre generadores.

    Args:
        meta_path: ruta al CSV de metadatos.
        generators: generadores esperados según la config.
        out: carpeta de salida de los artefactos.
    """
    counts: dict[str, dict[str, int]] = {}
    first_gen: dict[str, str] = {}
    n_nat = 0
    dup = 0
    n_corrupt = 0
    dup_examples: list[dict] = []
    sample_nat: list[dict] = []
    rng = random.Random(0)
    seen_dirs: set[str] = set()

    with open(meta_path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # cabecera
        for row in tqdm(reader, desc="csv", unit="fila"):
            if not row:
                continue
            path = row[0]
            if path in corrupted:
                # se descuenta del conteo: no existe/está corrupto
                n_corrupt += 1
                continue
            a = path.split("/")
            if len(a) < 5:
                continue
            gen, mode, kind, base = a[1], a[2], a[3], a[-1]
            seen_dirs.add(gen)
            counts.setdefault(gen, {})
            key = f"{mode}_{kind}"
            counts[gen][key] = counts[gen].get(key, 0) + 1
            if kind == "nature":
                n_nat += 1
                prev = first_gen.get(base)
                if prev is None:
                    first_gen[base] = gen
                elif prev != gen and len(dup_examples) < 50:
                    dup += 1
                    dup_examples.append({
                        "basename": base, "gen_a": prev, "gen_b": gen,
                    })
                elif prev != gen:
                    dup += 1
                # reservoir de 1000 para mostrar 15 al azar
                if len(sample_nat) < 1000:
                    sample_nat.append(
                        {"basename": base, "generador": gen,
                         "mode": mode}
                    )
                elif rng.random() < 1000 / max(n_nat, 1):
                    sample_nat[rng.randrange(1000)] = {
                        "basename": base, "generador": gen,
                        "mode": mode,
                    }

    # nombres vs config
    cfg_set, found = set(generators), seen_dirs
    print(f"[B0-csv] generadores en CSV : {sorted(found)}")
    faltan = sorted(cfg_set - found)
    extra = sorted(found - cfg_set)
    if faltan:
        print(f"[B0-csv][aviso] en config sin datos: {faltan}")
    if extra:
        print(f"[B0-csv][aviso] en datos sin config: {extra}")

    # tabla de conteos
    rows_counts: list[dict] = []
    for g in sorted(found):
        c = counts[g]
        rows_counts.append({
            "generador": g,
            "train_ai": c.get("train_ai", 0),
            "train_nature": c.get("train_nature", 0),
            "val_ai": c.get("val_ai", 0),
            "val_nature": c.get("val_nature", 0),
        })
    path_counts = out / "counts_csv.csv"
    _save_csv(path_counts, rows_counts)
    print(f"\n[guardado] {path_counts}")
    _show_random(rows_counts, k=15)

    # muestra de 15 nombres nature (cada uno con su único generador)
    path_sample = out / "nature_sample.csv"
    _save_csv(path_sample, sample_nat)
    print(f"\n[guardado] {path_sample}")
    print("15 nombres nature al azar (basename -> generador):")
    _show_random(sample_nat, k=15)

    # balance y contabilidad de reales (con y sin dedup)
    tr_ai = sum(c.get("train_ai", 0) for c in counts.values())
    tr_na = sum(c.get("train_nature", 0) for c in counts.values())
    va_ai = sum(c.get("val_ai", 0) for c in counts.values())
    va_na = sum(c.get("val_nature", 0) for c in counts.values())
    one_gen = sorted(counts)[0] if counts else None
    real_one = counts[one_gen].get("train_nature", 0) if one_gen else 0
    print("\n[B0-csv] balance y reales con/sin dedup")
    if tr_ai:
        print(
            f"  train ai={tr_ai}  "
            f"nature SIN dedup (los {len(counts)} gen)={tr_na} -> "
            f"{tr_na/tr_ai:.2f}:1",
        )
        print(
            f"        nature CON dedup ('{one_gen}')={real_one} -> "
            f"{real_one/tr_ai:.2f}:1  (deja las reales en ~1:M)",
        )
    if va_ai:
        print(
            f"  val   ai={va_ai}  nature={va_na} -> "
            f"{va_na/va_ai:.2f}:1",
        )
    print(
        f"  (descontados {n_corrupt} ficheros de corrupted_files.txt)",
    )

    # balance de los 8 splits leave-one-out (train), con/sin dedup
    bal_rows = _logo_balances(counts, generators)
    path_bal = out / "logo_balances.csv"
    _save_csv(path_bal, bal_rows)
    print(f"\n[guardado] {path_bal}")
    print("balance train por fold leave-one-out:")
    for r in bal_rows:
        print(
            f"  held_out={r['held_out']:<24} ai={r['train_ai']:>8}  "
            f"real(sin dedup)={r['real_nodedup']:>8} "
            f"({r['ratio_nodedup']}:1)  "
            f"real(dedup)={r['real_dedup']:>7} "
            f"({r['ratio_dedup']}:1)",
        )

    # fold construible ahora con los 3 generadores completos en disco
    trio = [g for g in ("ADM", "BigGAN", "glide") if g in counts]
    if len(trio) == 3:
        print(
            "\n[B0-csv] fold construible con 3 gen completos "
            "(ADM, BigGAN, glide):",
        )
        for r in _logo_balances(counts, trio):
            print(
                f"  held_out={r['held_out']:<8} ai={r['train_ai']:>7}  "
                f"real={r['real_nodedup']:>7} "
                f"({r['ratio_nodedup']}:1, sin dedup)",
            )

    # veredicto duplicación
    print("\n[VEREDICTO B0 (csv)]")
    print(f"  nombres nature únicos: {len(first_gen)} de {n_nat} filas")
    if dup == 0:
        print(
            "  reales DISJUNTAS entre generadores (0 nombres "
            "compartidos) -> concatenar todas; NO deduplicar.",
        )
    else:
        frac = dup / max(n_nat, 1)
        _save_csv(out / "dup_examples.csv", dup_examples)
        print(
            f"  {dup} nombres nature aparecen en >1 generador "
            f"({frac:.4f}) -> revisar duplicación; ejemplos en "
            f"dup_examples.csv",
        )


def _dup_compare(
    base: Path,
    gen_a: str,
    gen_b: str,
    split: str,
    real_subdir: str,
    n_samples: int,
) -> tuple[list[dict], dict]:
    """Compara por contenido las reales de dos generadores.

    Args:
        base: raíz de GenImage.
        gen_a: primer generador.
        gen_b: segundo generador.
        split: split a comparar (``train`` o ``val``).
        real_subdir: carpeta de reales.
        n_samples: nº de ficheros comunes a hashear.

    Returns:
        Par (filas de muestra, resumen). El resumen trae el solape de
        nombres y la fracción de contenido idéntico.
    """
    names_a = _list_image_names(base / gen_a / split / real_subdir)
    names_b = _list_image_names(base / gen_b / split / real_subdir)
    common = sorted(set(names_a) & set(names_b))
    union = set(names_a) | set(names_b)
    overlap = len(common) / max(len(union), 1)
    rng = random.Random(0)
    pick = common if len(common) <= n_samples else rng.sample(
        common, n_samples,
    )
    rows: list[dict] = []
    same_hash = 0
    for name in tqdm(
        pick, desc=f"hash {gen_a}~{gen_b}", unit="img", leave=False,
    ):
        ha = _sha1(names_a[name])
        hb = _sha1(names_b[name])
        eq = ha == hb
        same_hash += int(eq)
        rows.append({
            "basename": name,
            "gen_a": gen_a,
            "gen_b": gen_b,
            "same_content": int(eq),
            "sha1_a": ha[:12],
            "sha1_b": hb[:12],
        })
    frac_eq = same_hash / max(len(pick), 1)
    summary = {
        "gen_a": gen_a,
        "gen_b": gen_b,
        "n_a": len(names_a),
        "n_b": len(names_b),
        "name_overlap": round(overlap, 4),
        "n_compared": len(pick),
        "frac_same_content": round(frac_eq, 4),
    }
    return rows, summary


def main() -> None:
    """Ejecuta la verificación B0 y vuelca el informe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/detect_genimage.yaml",
    )
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument(
        "--metadata", default=None,
        help="ruta a genimage_metadata.csv; activa el modo CSV "
             "(verificación sin necesidad de extraer)",
    )
    parser.add_argument(
        "--corrupted", default=None,
        help="ruta a corrupted_files.txt; se descuenta del conteo",
    )
    parser.add_argument(
        "--hash-gens", default=None,
        help="lista de generadores (coma) para B0 por hash de "
             "contenido sobre disco, p. ej. 'ADM,BigGAN,glide'",
    )
    parser.add_argument("--hash-samples", type=int, default=300)
    parser.add_argument(
        "--out-dir", default="artifacts/tables/genimage_b0",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]
    root = str(ds["root"])
    base = Path(root)
    generators = list(ds["generators"])
    real_subdir = str(ds.get("real_subdir", "nature"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # modo CSV: autoritativo y disponible antes de la extracción
    if args.metadata is not None:
        print(f"[B0] modo CSV: {args.metadata}")
        corrupted = _load_corrupted(args.corrupted)
        if corrupted:
            print(f"[B0] corrupted_files: {len(corrupted)} rutas")
        _verify_from_csv(args.metadata, generators, out, corrupted)
        if args.hash_gens:
            gl = [
                g.strip() for g in args.hash_gens.split(",")
                if g.strip()
            ]
            print(
                f"\n[B0-hash] contenido entre {gl} "
                f"(muestra {args.hash_samples}/gen)",
            )
            dup = _hash_content_check(
                root, gl, args.hash_samples,
                real_subdir=real_subdir,
            )
            print("[VEREDICTO B0-hash]")
            if dup:
                print(
                    "  colisiones de contenido -> reales DUPLICADAS; "
                    "real_dedup quedaría justificado",
                )
            else:
                print(
                    "  sin colisiones -> reales DISTINTAS por "
                    "contenido; no deduplicar (real_dedup innecesario)",
                )
        return

    print(f"[B0] raíz: {root}")
    # carpetas reales presentes vs configuradas
    present = sorted(
        e.name for e in os.scandir(base) if e.is_dir()
    ) if base.exists() else []
    cfg_set, pres_set = set(generators), set(present)
    print(f"[B0] generadores en config : {sorted(cfg_set)}")
    print(f"[B0] carpetas en disco     : {present}")
    faltan = sorted(cfg_set - pres_set)
    extra = sorted(pres_set - cfg_set)
    if faltan:
        print(f"[B0][aviso] configurados sin carpeta: {faltan}")
    if extra:
        print(f"[B0][aviso] carpetas no configuradas: {extra}")

    # 1) conteos de estructura
    print("\n[B0] contando ai/nature por generador y split...")
    counts = verify_structure(root, generators, real_subdir)
    rows_counts: list[dict] = []
    for g in tqdm(generators, desc="conteo", unit="gen"):
        c = counts[g]
        rows_counts.append({"generador": g, **c})
    path_counts = out / "counts.csv"
    _save_csv(path_counts, rows_counts)
    print(f"\n[guardado] {path_counts}")
    _show_random(rows_counts, k=15)

    # generadores con reales en train disponibles para comparar
    avail = [
        g for g in generators
        if counts[g]["train_real"] > 0
    ]

    # 2) test de duplicación entre pares
    rows_dup: list[dict] = []
    summaries: list[dict] = []
    if len(avail) < 2:
        print(
            "\n[B0] menos de 2 generadores con reales extraídos; "
            "el test de duplicación se reintenta tras la extracción.",
        )
    else:
        anchor = avail[0]
        print(
            f"\n[B0] comparando reales de '{anchor}' contra el resto "
            f"(muestra de {args.samples} por par)...",
        )
        for g in avail[1:]:
            r, s = _dup_compare(
                base, anchor, g, "train", real_subdir, args.samples,
            )
            rows_dup.extend(r)
            summaries.append(s)
            print(
                f"  {anchor} ~ {g}: solape_nombres="
                f"{s['name_overlap']:.3f}  "
                f"contenido_idéntico={s['frac_same_content']:.3f} "
                f"({s['n_compared']} comparadas)",
            )
        path_dup = out / "dup_sample.csv"
        _save_csv(path_dup, rows_dup)
        _save_csv(out / "dup_summary.csv", summaries)
        print(f"\n[guardado] {path_dup}")
        _show_random(rows_dup, k=15)

    # 3) balance real/sintético con y sin dedup (sobre train)
    total_real_all = sum(counts[g]["train_real"] for g in generators)
    one_real = counts[avail[0]]["train_real"] if avail else 0
    total_ai = sum(counts[g]["train_ai"] for g in generators)
    print("\n[B0] balance train real:sintético")
    if total_ai > 0:
        print(
            f"  sin dedup (reales ×M): {total_real_all} real / "
            f"{total_ai} ai  -> ratio {total_real_all/total_ai:.2f}:1",
        )
        if one_real:
            print(
                f"  con dedup (1 generador): {one_real} real / "
                f"{total_ai} ai  -> ratio {one_real/total_ai:.2f}:1",
            )

    # veredicto
    dup_detected = any(
        s["frac_same_content"] > 0.9 and s["name_overlap"] > 0.5
        for s in summaries
    )
    print("\n[VEREDICTO B0]")
    if dup_detected:
        print(
            "  reales DUPLICADAS entre generadores -> usar "
            "real_dedup=true (ya es el valor por defecto del loader).",
        )
    elif summaries:
        print(
            "  no se detecta duplicación clara; revisar el informe "
            "antes de decidir real_dedup.",
        )
    else:
        print(
            "  test de duplicación pendiente (faltan generadores "
            "extraídos).",
        )


if __name__ == "__main__":
    main()
