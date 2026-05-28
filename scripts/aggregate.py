"""Agrega los logs y produce las tres tablas tabularx del paper.

Lee:

- ``artifacts/logs/codes.csv``                 — Tabla 1
- ``artifacts/logs/attn/run.csv``              — Tabla 2 (top-1 y
  diversidad de cabezas tras la última época)
- ``artifacts/logs/detect/<gen>.csv``          — Tabla 3 (LOGO)

Y vuelca:

- ``artifacts/tables/tabla_codigos.tex``,
- ``artifacts/tables/tabla_atencion.tex``,
- ``artifacts/tables/tabla_logo.tex``.

Junto a las tablas se generan figuras PNG bajo
``artifacts/figs/``:

- histograma de cosenos por código,
- traza del descenso (E_min, theta_min) por código,
- redundancia por capa al final del entrenamiento (atención),
- ROC por generador (detección).
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.viz.sphere import cos_histogram


def _read_csv(p: Path) -> list[dict]:
    """Lee un CSV con cabecera y devuelve lista de filas."""
    if not p.exists():
        return []
    with p.open("r") as fh:
        rdr = csv.DictReader(fh)
        return list(rdr)


def _table_codes(rows: list[dict]) -> str:
    """Genera el cuerpo tabularx de la tabla de códigos."""
    out: list[str] = []
    out.append(
        r"\begin{tabularx}{\textwidth}{l r r r r r r c}"
    )
    out.append(r"\toprule")
    out.append(
        "nombre & $d$ & $K$ & "
        r"$\theta_{\min}$ & $\theta_{\text{med}}$ & "
        r"$\theta_{\text{máx}}$ & $E_\text{Riesz}$ & kiss. \\"
    )
    out.append(r"\midrule")
    for r in rows:
        ok = r.get("kissing_ok", "")
        ok_str = "—"
        if ok == "1":
            ok_str = r"\checkmark"
        elif ok == "0":
            ok_str = r"\times"
        out.append(
            f"{r['name']} & {r['d']} & {r['K']} & "
            f"{float(r['theta_min_deg']):.2f} & "
            f"{float(r['theta_median_deg']):.2f} & "
            f"{float(r['theta_max_deg']):.2f} & "
            f"{float(r['energy_riesz_s1']):.3f} & {ok_str}"
            r" \\"
        )
    out.append(r"\bottomrule")
    out.append(r"\end{tabularx}")
    return "\n".join(out)


def _table_attn(rows: list[dict]) -> str:
    """Tabla resumen de la corrida A: épocas seleccionadas + final."""
    out: list[str] = []
    out.append(r"\begin{tabularx}{\textwidth}{r r r r r r r}")
    out.append(r"\toprule")
    out.append(
        r"época & CE & R$_\text{div}$ & redund. & "
        r"$\theta_{\min}$ & top-1 & top-5 \\"
    )
    out.append(r"\midrule")
    if rows:
        ep_idx = sorted({int(r["epoch"]) for r in rows})
        # se muestran como mucho 6 filas: inicio, ¼, ½, ¾, penúltima,
        # final
        keep = sorted(set([
            ep_idx[0],
            ep_idx[len(ep_idx) // 4],
            ep_idx[len(ep_idx) // 2],
            ep_idx[3 * len(ep_idx) // 4],
            ep_idx[-2] if len(ep_idx) > 1 else ep_idx[-1],
            ep_idx[-1],
        ]))
        by_ep = {int(r["epoch"]): r for r in rows}
        for ep in keep:
            r = by_ep[ep]
            out.append(
                f"{ep} & {float(r['ce']):.3f} & "
                f"{float(r['div']):.3f} & "
                f"{float(r['redundancy_mean']):.3f} & "
                f"{float(r['theta_min_mean_deg']):.2f} & "
                f"{float(r['val_top1']):.3f} & "
                f"{float(r['val_top5']):.3f}"
                r" \\"
            )
    out.append(r"\bottomrule")
    out.append(r"\end{tabularx}")
    return "\n".join(out)


def _table_logo(per_gen: dict[str, list[dict]]) -> str:
    """Tabla LOGO con mean±std por generador sobre semillas.

    ``per_gen`` mapea nombre de generador excluido a la lista de
    últimas filas (una por semilla). Si hay una sola fila, std=0.
    """
    out: list[str] = []
    out.append(r"\begin{tabularx}{\textwidth}{l r r r r}")
    out.append(r"\toprule")
    out.append(
        r"generador excluido & AUROC & OSCR & top-1 cerr. & "
        r"acc real/sint \\"
    )
    out.append(r"\midrule")

    def _mean_std(values: list[float]) -> tuple[float, float]:
        """Media y desv. tip. poblacional de una lista."""
        n = len(values)
        if n == 0:
            return 0.0, 0.0
        m = sum(values) / n
        if n == 1:
            return m, 0.0
        var = sum((v - m) ** 2 for v in values) / n
        return m, var ** 0.5

    def _fmt(m: float, s: float) -> str:
        """Formato 'm±s' o 'm' si s≈0."""
        if s < 1.0e-6:
            return f"{m:.3f}"
        return f"{m:.3f}$\\pm${s:.3f}"

    if per_gen:
        agg_au: list[float] = []
        agg_os: list[float] = []
        agg_t1: list[float] = []
        agg_rf: list[float] = []
        for g in sorted(per_gen):
            rows = per_gen[g]
            g_esc = g.replace("_", r"\_")
            au = [float(r["val_auroc_realfake"]) for r in rows]
            os_ = [float(r["val_oscr"]) for r in rows]
            t1 = [float(r["val_closed_top1"]) for r in rows]
            rf = [float(r["val_acc_realfake"]) for r in rows]
            m_au, s_au = _mean_std(au)
            m_os, s_os = _mean_std(os_)
            m_t1, s_t1 = _mean_std(t1)
            m_rf, s_rf = _mean_std(rf)
            agg_au.extend(au)
            agg_os.extend(os_)
            agg_t1.extend(t1)
            agg_rf.extend(rf)
            out.append(
                f"{g_esc} & "
                f"{_fmt(m_au, s_au)} & "
                f"{_fmt(m_os, s_os)} & "
                f"{_fmt(m_t1, s_t1)} & "
                f"{_fmt(m_rf, s_rf)}"
                r" \\"
            )
        out.append(r"\midrule")
        out.append(
            r"\textbf{media} & "
            f"{_fmt(*_mean_std(agg_au))} & "
            f"{_fmt(*_mean_std(agg_os))} & "
            f"{_fmt(*_mean_std(agg_t1))} & "
            f"{_fmt(*_mean_std(agg_rf))}"
            r" \\"
        )
    out.append(r"\bottomrule")
    out.append(r"\end{tabularx}")
    return "\n".join(out)


def _fig_attn_layerwise(rows: list[dict], path: Path) -> None:
    """Redundancia y theta_min por capa al final del entrenamiento."""
    if not rows:
        return
    last_ep = max(int(r["epoch"]) for r in rows)
    sel = [r for r in rows if int(r["epoch"]) == last_ep]
    sel.sort(key=lambda r: int(r["layer"]))
    layers = [int(r["layer"]) for r in sel]
    redun = [float(r["redundancy"]) for r in sel]
    thmin = [float(r["theta_min_deg"]) for r in sel]
    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax2 = ax1.twinx()
    ax1.plot(layers, redun, "o-", color="#3a5fcd", label="redund.")
    ax2.plot(layers, thmin, "s--", color="#cd5c3a",
             label=r"$\theta_{\min}$")
    ax1.set_xlabel("capa")
    ax1.set_ylabel("redundancia (|cos|)", color="#3a5fcd")
    ax2.set_ylabel(r"$\theta_{\min}$ (grados)", color="#cd5c3a")
    ax1.set_title(f"diversidad por capa, época {last_ep}")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _fig_detect_roc(pred_dir: Path, fig_path: Path) -> None:
    """Curvas ROC por generador a partir de los dumps de predicciones.

    Espera la estructura
    ``checkpoints/detect/predictions/<gen>/ep<NN>.pt``.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    plotted = False
    for sub in sorted(pred_dir.glob("*")):
        if not sub.is_dir():
            continue
        eps = sorted(sub.glob("ep*.pt"))
        if not eps:
            continue
        blob = torch.load(eps[-1], map_location="cpu",
                          weights_only=False)
        roc = blob.get("roc")
        if roc is None:
            continue
        ax.plot(roc["fpr"], roc["tpr"], label=sub.name, lw=1.0)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC real vs sintético por generador excluido")
    ax.legend(fontsize=7, loc="lower right")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _fig_code_traces(codes_dir: Path, out_dir: Path) -> None:
    """Histograma de cosenos y traza del descenso por código."""
    for pt in sorted(codes_dir.glob("*.pt")):
        blob = torch.load(pt, map_location="cpu", weights_only=False)
        name = pt.stem
        edges = blob.get("hist_cos_edges")
        counts = blob.get("hist_cos_counts")
        if edges and counts:
            cos_histogram(
                edges, counts,
                title=f"{name}: histograma cos(theta) por pares",
                path=str(out_dir / f"hist_{name}.png"),
            )
        trace = blob.get("trace") or []
        if trace:
            steps = [t[0] for t in trace]
            es = [t[1] for t in trace]
            ths = [t[2] for t in trace]
            fig, ax1 = plt.subplots(figsize=(6, 3.5))
            ax2 = ax1.twinx()
            ax1.plot(steps, es, "-", color="#3a5fcd",
                     label="E Riesz")
            ax2.plot(steps, ths, "--", color="#cd5c3a",
                     label=r"$\theta_{\min}$")
            ax1.set_xlabel("paso")
            ax1.set_ylabel("E Riesz", color="#3a5fcd")
            ax2.set_ylabel(r"$\theta_{\min}$ (grados)",
                           color="#cd5c3a")
            ax1.set_title(f"{name}: descenso de Riesz")
            out_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_dir / f"descenso_{name}.png",
                        dpi=110, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    """Produce tablas y figuras del paper a partir de los logs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs", default="artifacts/logs",
    )
    parser.add_argument(
        "--codes", default="artifacts/codes",
    )
    parser.add_argument(
        "--ckpts", default="artifacts/checkpoints",
    )
    parser.add_argument(
        "--tables", default="artifacts/tables",
    )
    parser.add_argument(
        "--figs", default="artifacts/figs",
    )
    parser.add_argument(
        "--attn-subdir", default="attn",
        help="subcarpeta dentro de logs con los csvs de Fase 3",
    )
    parser.add_argument(
        "--detect-subdir", default="detect",
        help="subcarpeta dentro de logs con los csvs de Fase 4",
    )
    args = parser.parse_args()

    logs = Path(args.logs)
    codes = Path(args.codes)
    ckpts = Path(args.ckpts)
    tables = Path(args.tables)
    figs = Path(args.figs)
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    # tabla 1: códigos
    codes_rows = _read_csv(logs / "codes.csv")
    (tables / "tabla_codigos.tex").write_text(
        _table_codes(codes_rows), encoding="utf-8",
    )

    # tabla 2: atención
    attn_rows = _read_csv(logs / args.attn_subdir / "run.csv")
    (tables / "tabla_atencion.tex").write_text(
        _table_attn(attn_rows), encoding="utf-8",
    )

    # tabla 3: LOGO. agrupamos por nombre de generador y reducimos
    # sobre semillas. el sufijo "_seed<N>" del stem se quita; los
    # csvs auxiliares (per_class, per_gen_auroc) se ignoran aquí.
    per_gen: dict[str, list[dict]] = {}
    detect_dir = logs / args.detect_subdir
    if detect_dir.exists():
        for csv_p in sorted(detect_dir.glob("*.csv")):
            stem = csv_p.stem
            if stem.endswith("_per_class") or \
               stem.endswith("_per_gen_auroc"):
                continue
            # se separa el sufijo _seed<N> si existe
            if "_seed" in stem:
                gen_name = stem.rsplit("_seed", 1)[0]
            else:
                gen_name = stem
            rows = _read_csv(csv_p)
            if rows:
                per_gen.setdefault(gen_name, []).append(rows[-1])
    (tables / "tabla_logo.tex").write_text(
        _table_logo(per_gen), encoding="utf-8",
    )

    # figuras
    _fig_code_traces(codes, figs)
    _fig_attn_layerwise(
        _read_csv(logs / args.attn_subdir / "layerwise.csv"),
        figs / "attn_layerwise.png",
    )
    pred_root = ckpts / "detect" / "predictions"
    if pred_root.exists():
        _fig_detect_roc(pred_root, figs / "detect_roc.png")

    # se imprime un muestreo aleatorio de 15 líneas del artefacto
    all_lines: list[str] = []
    for p in [
        tables / "tabla_codigos.tex",
        tables / "tabla_atencion.tex",
        tables / "tabla_logo.tex",
    ]:
        all_lines.extend(p.read_text().splitlines())
    print("artefactos guardados:")
    for d in (tables, figs):
        print(f"  {d}/")
        for x in sorted(d.iterdir()):
            print(f"    {x.name}")
    print("\n15 líneas aleatorias del conjunto de tablas:")
    sample = random.sample(all_lines, min(15, len(all_lines)))
    for ln in sample:
        print(f"  {ln}")


if __name__ == "__main__":
    main()
