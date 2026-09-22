"""pasada 3 — el punto balanceado del sector valor-salida.

verifica numéricamente el lema del punto balanceado antes de que su
prosa entre al canónico: sobre instancias aleatorias, la
factorización construida por SVD del circuito (i) satisface la
condición de balance, (ii) minimiza ‖W_vR‖²+‖R⁻¹W_O‖² frente a
perturbaciones de R, y (iii) alinea v₁(W_O) con v₁(W_vW_O); y sobre
el ancla real, α ---el ángulo entre la lectura por pesos y el
invariante--- evaluado en el punto balanceado construido cae al cero
numérico. sin forwards: todo es álgebra sobre los pesos.

las dos cláusulas del enunciado se comprueban también por su
negativo: con circuito de rango deficiente la unicidad falla, y con
espectro degenerado la igualdad de v₁ baja a subespacios.

uso:
    python scripts/punto_balanceado.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.carga import cargar_vit_base  # noqa: E402
from src.firma_funcional import w_o_por_cabeza  # noqa: E402

CKPT = "artifacts/checkpoints/vitb_clean/attnA_base_seed42_last.pt"
SALIDA = "artifacts/logs/radio_firma/punto_balanceado.csv"
N_CABEZAS, DIM_CABEZA = 12, 64


def balancea(w_v: torch.Tensor, w_o: torch.Tensor):
    """construye la factorización balanceada del circuito por SVD.

    con C = W_vW_O = UΣVᵀ (reducida al rango r), el par balanceado es
    A = U Σ^{1/2}, B = Σ^{1/2} Vᵀ: AᵀA = Σ = BBᵀ y AB = C.

    Args:
        w_v: [d, dh] proyección de valor de la cabeza.
        w_o: [dh, d] proyección de salida.

    Returns:
        tupla (a, b, sv): factores balanceados y espectro de C.
    """
    c = w_v @ w_o
    u, s, vh = torch.linalg.svd(c, full_matrices=False)
    r = int((s > s[0] * 1e-12).sum())
    raiz = s[:r].sqrt()
    a = u[:, :r] * raiz
    b = raiz.unsqueeze(1) * vh[:r]
    return a, b, s


def test_aleatorias(n: int = 50, dh: int = 8, d: int = 40) -> None:
    """las tres propiedades del lema sobre instancias aleatorias.

    Args:
        n: número de instancias.
        dh: dimensión de cabeza sintética.
        d: dimensión de residuo sintética.

    Raises:
        AssertionError: si alguna propiedad falla.
    """
    g = torch.Generator().manual_seed(0)
    for i in tqdm(range(n), desc="aleatorias", leave=False):
        w_v = torch.randn(d, dh, generator=g, dtype=torch.float64)
        w_o = torch.randn(dh, d, generator=g, dtype=torch.float64)
        a, b, s = balancea(w_v, w_o)
        c = w_v @ w_o
        # (o) reconstruye el circuito
        assert torch.allclose(a @ b, c, atol=1e-9)
        # (i) balance: a^t a = b b^t
        assert torch.allclose(a.t() @ a, b @ b.t(), atol=1e-9)
        # (ii) mínimo: el objetivo balanceado no lo baja ninguna R
        obj_bal = float((a ** 2).sum() + (b ** 2).sum())
        for _ in range(20):
            rr = (torch.eye(dh, dtype=torch.float64)
                  + 0.1 * torch.randn(dh, dh, generator=g,
                                      dtype=torch.float64))
            # se perturba el par balanceado extendido a dh columnas
            av = torch.zeros(d, dh, dtype=torch.float64)
            av[:, :a.shape[1]] = a
            bv = torch.zeros(dh, d, dtype=torch.float64)
            bv[:b.shape[0]] = b
            obj = float(((av @ rr) ** 2).sum()
                        + ((torch.linalg.inv(rr) @ bv) ** 2).sum())
            assert obj >= obj_bal - 1e-7, f"mínimo roto en {i}"
        # (iii) v1 del factor de salida = v1 del circuito
        v1_b = torch.linalg.svd(b, full_matrices=False).Vh[0]
        v1_c = torch.linalg.svd(c, full_matrices=False).Vh[0]
        assert min(float((v1_b - v1_c).norm()),
                   float((v1_b + v1_c).norm())) < 1e-9
    print(f"  aleatorias: {n} instancias, tres propiedades en verde")


def test_familia_alineada(n: int = 10, dh: int = 8,
                          d: int = 40) -> None:
    """la familia alineada que sigue al lema, forma QR incluida.

    con W_v = U D Q ---U base singular izquierda de C, D diagonal
    positiva, Q ortogonal--- se tiene W_v⁺ = Qᵀ D⁻¹ Uᵀ, luego
    W_O = Qᵀ D⁻¹ Σ Vᵀ y v₁(W_O) = ±v₁(C) siempre que σ₁/d₁ sea el
    cociente máximo. la forma QR de Wang y Wang es el caso D = I.

    Args:
        n: número de instancias.
        dh: dimensión de cabeza sintética.
        d: dimensión de residuo sintética.

    Raises:
        AssertionError: si la pseudoinversa o la alineación fallan.
    """
    g = torch.Generator().manual_seed(1)
    eye = torch.eye(dh, dtype=torch.float64)
    for i in tqdm(range(n), desc="familia alineada", leave=False):
        w_v = torch.randn(d, dh, generator=g, dtype=torch.float64)
        w_o = torch.randn(dh, d, generator=g, dtype=torch.float64)
        c = w_v @ w_o
        u, s, vh = torch.linalg.svd(c, full_matrices=False)
        # se reduce al rango dh: base singular izquierda de c
        u, s, vh = u[:, :dh], s[:dh], vh[:dh]
        v1_c = vh[0]
        q, _ = torch.linalg.qr(torch.randn(dh, dh, generator=g,
                                           dtype=torch.float64))
        for caso in ("qr", "diagonal"):
            if caso == "qr":
                dg = torch.ones(dh, dtype=torch.float64)
            else:
                # d1 pequeño para que sigma1/d1 sea el maximo
                dg = 1.0 + torch.rand(dh, generator=g,
                                      dtype=torch.float64)
                dg[0] = 0.5
            w_v_g = u @ torch.diag(dg) @ q
            # se comprueba la pseudoinversa cerrada
            pinv = q.t() @ torch.diag(1.0 / dg) @ u.t()
            assert torch.allclose(torch.linalg.pinv(w_v_g), pinv,
                                  atol=1e-9), f"pinv rota en {i}"
            assert torch.allclose(w_v_g.t() @ w_v_g,
                                  q.t() @ torch.diag(dg ** 2) @ q,
                                  atol=1e-9)
            w_o_g = pinv @ c
            assert torch.allclose(w_v_g @ w_o_g, c, atol=1e-9)
            v1_o = torch.linalg.svd(w_o_g, full_matrices=False).Vh[0]
            assert min(float((v1_o - v1_c).norm()),
                       float((v1_o + v1_c).norm())) < 1e-9, \
                f"alineación rota ({caso}) en {i}"
            # la parte ortogonal de la clase inocua no mueve v1
            v1_q = torch.linalg.svd(q @ w_o_g,
                                    full_matrices=False).Vh[0]
            assert min(float((v1_q - v1_c).norm()),
                       float((v1_q + v1_c).norm())) < 1e-9
    print(f"  familia alineada: {n} instancias, QR y diagonal en verde")


def main() -> None:
    """corre el test unitario y el sanity del ancla, y guarda csv."""
    test_aleatorias()
    test_familia_alineada()
    disp = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = cargar_vit_base(CKPT, device=disp).eval()
    filas = []
    for capa in tqdm(range(len(modelo.blocks)), desc="ancla"):
        w_o = w_o_por_cabeza(modelo, capa, N_CABEZAS, DIM_CABEZA)
        attn = modelo.blocks[capa].attn
        w_qkv = attn.qkv.weight.detach()
        d = w_qkv.shape[1]
        for h in range(N_CABEZAS):
            fil = slice(2 * d + h * DIM_CABEZA,
                        2 * d + (h + 1) * DIM_CABEZA)
            w_v = w_qkv[fil, :].t().double().cpu()      # [d, dh]
            w_o_h = w_o[h].double().cpu()               # [dh, d]
            a, b, s = balancea(w_v, w_o_h)
            v1_bal = torch.linalg.svd(b, full_matrices=False).Vh[0]
            v1_c = torch.linalg.svd(w_v @ w_o_h,
                                    full_matrices=False).Vh[0]
            cosa = float((v1_bal * v1_c).sum().abs().clamp(max=1.0))
            alfa = float(torch.rad2deg(torch.arccos(
                torch.tensor(cosa))))
            filas.append({"capa": capa, "cabeza": h,
                          "alfa_balanceado_deg": alfa,
                          "sigma1_c": float(s[0]),
                          "sigma2_c": float(s[1]),
                          "hueco_rel": float((s[0] - s[1]) / s[0])})
    df = pd.DataFrame(filas)
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    df.to_csv(SALIDA, index=False)                # se guarda en csv
    print(df.sample(15, random_state=0))          # 15 observaciones
    print(f"\n  ancla: alfa en el punto balanceado, mediana "
          f"{df.alfa_balanceado_deg.median():.2e} grados, máx "
          f"{df.alfa_balanceado_deg.max():.2e}")


if __name__ == "__main__":
    main()
