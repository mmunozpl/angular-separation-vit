"""órbita de gauge de v1(w_o) sobre los pesos reales, en cpu.

certifica tres cosas sobre las cinco base de vit-b: (i) la banda de
ángulos que el gauge gl(d_h) puede imprimir a cada par de cabezas sin
tocar la función ---[theta_1, theta_dh] de ángulos principales entre
espacios fila---; (ii) la realización explícita del extremo inferior
en un par por semilla (gauge construido, circuito ov invariante a
precisión de máquina); (iii) el certificado SIMULTÁNEO: un descenso
restringido a los espacios fila busca 12 direcciones (una por cabeza,
cada una dentro de su s_h) con theta_min >= theta* = 84,78; si lo
alcanza, existe una configuración de gauges de coste funcional CERO
que cumple el regularizador por completo ---la banda por par no basta:
son 66 restricciones acopladas---.

uso:
    python scripts/orbita_gauge.py [--seeds 42 ...]
"""

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.firma_funcional import w_o_por_cabeza, w_v_por_cabeza
from src.models.vit_backbone import HeadProjections

N_CABEZAS, DIM_CABEZA = 12, 64
THETA_STAR = math.degrees(math.acos(1.0 / (N_CABEZAS - 1)))
SALIDA = "artifacts/logs/orbita_gauge/orbita_gauge.csv"


def _modelo(seed: int):
    """vit interno con los pesos base de una semilla, en cpu."""
    hp = HeadProjections(
        model_name="vit_base_patch16_224", pretrained=False,
        num_classes=100, img_size=224)
    blob = torch.load(
        f"artifacts/checkpoints/vitb_clean/attnA_base_seed{seed}"
        f"_last.pt", map_location="cpu", weights_only=False)
    sd = {k.replace("backbone.", "", 1): v
          for k, v in blob["model"].items()}
    hp.load_state_dict(sd)
    return hp.model.eval()


def _bases_y_v1(w_o: torch.Tensor) -> tuple[torch.Tensor,
                                            torch.Tensor]:
    """bases ortonormales de fila [h, d, dh] y v1 [h, d] por cabeza.

    se separan del grafo: los pesos del modelo llevan requires_grad y
    el certificado optimiza SOLO sus coeficientes, no los pesos.
    """
    with torch.no_grad():
        u, s, vh = torch.linalg.svd(w_o.double(),
                                    full_matrices=False)
    return vh.transpose(-2, -1).detach(), vh[:, 0, :].detach()


def _angulo(u: torch.Tensor, w: torch.Tensor) -> float:
    """ángulo sin signo entre direcciones, en grados."""
    c = float(torch.dot(u, w).abs().clamp(max=1.0))
    return math.degrees(math.acos(c))


def _gauge_hacia(w_o: torch.Tensor, objetivo: torch.Tensor,
                 gen: torch.Generator) -> torch.Tensor:
    """r en gl(d_h) con v1(r^-1 w_o) = +-objetivo (objetivo en fila)."""
    u, s, vh = torch.linalg.svd(w_o.double(), full_matrices=False)
    dh = w_o.shape[0]
    c = vh @ objetivo
    c = c / c.norm()
    aleat = torch.randn(dh, dh - 1, generator=gen,
                        dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.cat([c.unsqueeze(1), aleat], dim=1))
    q = q * torch.sign(q[:, 0] @ c)
    esp = torch.linspace(1.0, 0.1, dh, dtype=torch.float64)
    p, _ = torch.linalg.qr(torch.randn(dh, dh, generator=gen,
                                       dtype=torch.float64))
    m = p @ torch.diag(esp) @ q.t()
    return (u * s) @ torch.linalg.inv(m)


def certificado_simultaneo(bases: torch.Tensor,
                           pasos: int = 4000) -> float:
    """theta_min alcanzable con las 12 direcciones a la vez.

    desciende una penalización hinge sobre |cos| > 1/(h-1) con cada
    dirección restringida a su espacio fila (coordenadas libres en la
    base de s_h, normalizadas). devuelve el theta_min final: si
    >= theta*, el cumplimiento total del regularizador está en la
    órbita de gauge conjunta.

    Args:
        bases: tensor [h, d, dh] de bases ortonormales de fila.
        pasos: iteraciones de adam.

    Returns:
        theta_min de la configuración final, en grados.
    """
    h = bases.shape[0]
    coef = torch.randn(h, bases.shape[2], dtype=torch.float64,
                       requires_grad=True)
    opt = torch.optim.Adam([coef], lr=5.0e-2)
    umbral = 1.0 / (h - 1)
    for _ in range(pasos):
        opt.zero_grad()
        dirs = torch.einsum("hdk,hk->hd", bases, coef)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)
        g = (dirs @ dirs.t()).abs()
        g = g - torch.eye(h, dtype=torch.float64) * g.diagonal()
        exceso = (g - umbral).clamp(min=0.0)
        loss = (exceso ** 2).sum()
        if float(loss) == 0.0:
            break
        loss.backward()
        opt.step()
    with torch.no_grad():
        dirs = torch.einsum("hdk,hk->hd", bases, coef)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)
        g = (dirs @ dirs.t()).abs().clamp(max=1.0)
        g.fill_diagonal_(0.0)
        return math.degrees(math.acos(float(g.max())))


def main() -> None:
    """mide la órbita de gauge sobre las cinco base y guarda el csv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    gen = torch.Generator().manual_seed(7)

    filas: list[dict] = []
    certs: list[dict] = []
    for seed in tqdm(args.seeds, desc="semillas"):
        vit = _modelo(seed)
        for capa in tqdm(range(12), desc="capas", leave=False):
            w_o = w_o_por_cabeza(vit, capa, N_CABEZAS, DIM_CABEZA)
            bases, v1s = _bases_y_v1(w_o)
            for i in range(N_CABEZAS):
                for j in range(i + 1, N_CABEZAS):
                    sv = torch.linalg.svd(
                        bases[i].t() @ bases[j],
                        full_matrices=False).S.clamp(-1.0, 1.0)
                    ap = torch.rad2deg(torch.arccos(sv))
                    filas.append({
                        "seed": seed, "capa": capa, "i": i, "j": j,
                        "th_obs": round(_angulo(v1s[i], v1s[j]), 3),
                        "th_min_alc": round(float(ap.min()), 3),
                        "th_max_alc": round(float(ap.max()), 3)})
            th_cert = certificado_simultaneo(bases)
            certs.append({"seed": seed, "capa": capa,
                          "theta_min_simultaneo": round(th_cert, 3),
                          "certifica": th_cert >= THETA_STAR})
        # realización explícita del extremo inferior (un par por
        # semilla): gauge construido, ov invariante, v1 en el objetivo
        w_o = w_o_por_cabeza(vit, 0, N_CABEZAS, DIM_CABEZA)
        w_v = w_v_por_cabeza(vit, 0, N_CABEZAS, DIM_CABEZA)
        bases, v1s = _bases_y_v1(w_o)
        m01 = bases[0].t() @ bases[1]
        uu, _, vvh = torch.linalg.svd(m01, full_matrices=False)
        obj0 = bases[0] @ uu[:, 0]
        obj1 = bases[1] @ vvh[0]
        derivas, th_real = [], None
        nuevos = []
        for h, obj in ((0, obj0), (1, obj1)):
            r = _gauge_hacia(w_o[h], obj, gen)
            ov0 = w_v[h].double().t() @ w_o[h].double()
            ov1 = ((w_v[h].double().t() @ r)
                   @ (torch.linalg.inv(r) @ w_o[h].double()))
            derivas.append(float((ov1 - ov0).abs().max()))
            nuevos.append(torch.linalg.svd(
                torch.linalg.inv(r) @ w_o[h].double(),
                full_matrices=False).Vh[0])
        th_real = _angulo(nuevos[0], nuevos[1])
        print(f"\n[seed {seed}] realización capa 0 par (0,1): "
              f"deriva_ov_max={max(derivas):.1e}  "
              f"th_realizado={th_real:.2f} "
              f"(certificado [{filas[0]['th_min_alc'] if seed == args.seeds[0] else '...'}])")

    out = Path(SALIDA)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        wr.writeheader()
        wr.writerows(filas)
    cert_path = out.parent / "certificado_simultaneo.csv"
    with cert_path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(certs[0].keys()))
        wr.writeheader()
        wr.writerows(certs)
    print(f"[guardado] {out} ({len(filas)} filas) + {cert_path}")

    rng = random.Random(0)
    for fila in rng.sample(filas, 15):
        print(fila)

    lo = min(f["th_min_alc"] for f in filas)
    hi = max(f["th_max_alc"] for f in filas)
    dentro = sum(1 for f in filas
                 if f["th_min_alc"] <= THETA_STAR <= f["th_max_alc"])
    obs_en_banda = sum(1 for f in filas
                       if f["th_min_alc"] <= f["th_obs"]
                       <= f["th_max_alc"])
    n_cert = sum(1 for c in certs if c["certifica"])
    peor_cert = min(c["theta_min_simultaneo"] for c in certs)
    print(f"\nbanda global alcanzable por gauge: [{lo:.2f}, {hi:.2f}]")
    print(f"theta*={THETA_STAR:.2f} dentro de la banda del par: "
          f"{dentro}/{len(filas)}")
    print(f"th_obs dentro de su banda: {obs_en_banda}/{len(filas)}")
    print(f"certificado simultáneo (theta_min >= theta*): "
          f"{n_cert}/{len(certs)} (peor: {peor_cert:.2f})")


if __name__ == "__main__":
    main()
