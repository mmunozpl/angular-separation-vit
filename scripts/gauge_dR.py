"""e0 — d(r) de la ecuación (2) sobre los gauges muestreados.

ejecuta el prerregistro `Paper_X/prereg_E0_dR.md` (respaldo c7773e4,
21-09-2026): postprocesado sin forwards. se reproduce cada r de la
fase g, de `decision_rota.csv` y de `decision_rota_lm.csv` desde su
semilla ---`torch.Generator("cpu").manual_seed(1000*capa + j)`, una r
por cabeza en el mismo orden que `src.gauge_flip.aplica_gauge_ov`,
r = randn + escala_id·i en float64--- y se calcula por cabeza la
distancia de la ecuación (2) a la clase inocua {cq}, la desviación
típica de los valores singulares partida por su norma. gate: la
delta_r recomputada coincide con `desv_R` del csv de la fase g a
1e-6, fila a fila.

uso:
    python scripts/gauge_dR.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from scripts.run_fase_G import ESCALAS  # noqa: E402

CSV_G = "artifacts/logs/fase_G/gauge_flip.csv"
CSV_ROTA = "artifacts/logs/decision_rota/decision_rota.csv"
CSV_LM = "artifacts/logs/decision_rota/decision_rota_lm.csv"
SALIDA = "artifacts/logs/fase_G/gauge_dR.csv"
ESCALA_ROTA = 8.0
GATE_TOL = 1e-6
DIM = {"vitb": (12, 64), "vitl": (16, 64), "dinov2": (16, 64),
       "pythia410m": (16, 64)}


def genera_r(semilla: int, n_cabezas: int, dim_cabeza: int,
             escala_id: float) -> list[torch.Tensor]:
    """las r por cabeza de un gauge, en el orden del generador."""
    g = torch.Generator(device="cpu").manual_seed(semilla)
    ident = torch.eye(dim_cabeza, dtype=torch.float64)
    rs = []
    for _ in range(n_cabezas):
        rt = torch.randn(dim_cabeza, dim_cabeza, generator=g,
                         dtype=torch.float64)
        rs.append(rt + escala_id * ident)
    return rs


def delta_r(r: torch.Tensor) -> float:
    """desviación respecto al mejor múltiplo escalar de la identidad."""
    dh = r.shape[0]
    ident = torch.eye(dh, dtype=torch.float64)
    s = r.diagonal().mean()
    return float((r - s * ident).norm() / (s.abs() * dh ** 0.5 + 1e-8))


def d_r(r: torch.Tensor) -> float:
    """distancia de la ecuación (2) a la clase conforme ortogonal."""
    sv = torch.linalg.svdvals(r)
    return float(((sv - sv.mean()) ** 2).sum().sqrt()
                 / (sv ** 2).sum().sqrt())


def fila(fuente: str, arch: str, capa: int, semilla: int,
         escala: float) -> dict:
    """medias sobre cabezas de delta_r y d(r) para un gauge."""
    h, dh = DIM[arch]
    rs = genera_r(semilla, h, dh, escala)
    return {"fuente": fuente, "arch": arch, "capa": capa,
            "semilla": semilla, "escala_id": escala,
            "delta_R": sum(delta_r(r) for r in rs) / h,
            "d_R": sum(d_r(r) for r in rs) / h}


def main() -> None:
    """recorre las tres fuentes, pasa el gate y guarda el csv."""
    filas: list[dict] = []
    g = pd.read_csv(CSV_G)
    # gate sobre la fase g: la r no depende del checkpoint, así que
    # cada (arch, capa, escala) se contrasta con todas sus filas
    peor = 0.0
    for (arch, capa, escala), grp in tqdm(
            g.groupby(["arch", "capa", "escala_id"]), desc="fase g"):
        j = ESCALAS.index(float(escala))
        f = fila("gauge_flip", arch, int(capa), 1000 * int(capa) + j,
                 float(escala))
        peor = max(peor, float((grp["desv_R"] - f["delta_R"]).abs()
                               .max()))
        filas.append(f)
    print(f"[gate] |desv_R csv - delta_R recomputada| máx = {peor:.2e}")
    if peor > GATE_TOL:
        raise SystemExit("gate de reproducción fallado: sin veredicto")
    for fuente, ruta, arch in (("decision_rota", CSV_ROTA, "vitb"),
                               ("decision_rota_lm", CSV_LM,
                                "pythia410m")):
        d = pd.read_csv(ruta)
        d = d[d["gauge_idx"] > 0]
        for (capa, gi), _ in tqdm(d.groupby(["capa", "gauge_idx"]),
                                  desc=fuente):
            filas.append(fila(fuente, arch, int(capa),
                              1000 * int(capa) + int(gi) - 1,
                              ESCALA_ROTA))
    df = pd.DataFrame(filas)
    Path(SALIDA).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SALIDA, index=False)
    print(f"[guardado] {SALIDA} ({len(df)} filas)")
    res = df[df["fuente"] == "gauge_flip"].groupby(
        ["arch", "escala_id"]).agg(delta_m=("delta_R", "mean"),
                                   d_m=("d_R", "mean"),
                                   d_s=("d_R", "std")).reset_index()
    print(res.to_string())
    for fuente in ("decision_rota", "decision_rota_lm"):
        sub = df[df["fuente"] == fuente]["d_R"]
        print(f"{fuente}: d(R) en [{sub.min():.3f}, {sub.max():.3f}], "
              f"media {sub.mean():.3f}")
    rng = random.Random(0)
    print("\n--- 15 observaciones aleatorias ---")
    print(df.iloc[rng.sample(range(len(df)), 15)].to_string())


if __name__ == "__main__":
    main()
