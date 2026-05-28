# Notas para integrar en `separacion_angular_paper.tex`

Cambios de redacción derivados de la auditoría 2026-05-25 entre el
código y el .tex. Son cambios de texto, **no de implementación**.

---

## Decisión #4 — afinado vs desde cero (§594-595)

El texto actual dice "La contribución A se entrena desde cero sobre
ImageNet". Con el cómputo disponible (una RTX 5090) un entrenamiento
de ViT-B/16 desde scratch al estilo DeiT (300 épocas + mixup +
cutmix + EMA + distillation) no es viable en plazos razonables.
Reescribir como:

> La contribución A parte de pesos preentrenados de ViT-B/16
> (timm) y se afina sobre ImageNet con la pérdida $\mathcal{L}$ de
> la ecuación~\eqref{eq:loss}. El regulador
> $\mathcal{R}_{\mathrm{div}}$ actúa así como un aditivo geométrico
> sobre una columna ya competente, lo que aísla mejor su efecto que
> un entrenamiento conjunto desde cero.

Eliminar la mención "desde cero" y dejar explícito el afinado.

---

## Nota #6 — Validación geométrica con inicialización canónica (§442-449)

El texto actual sugiere que el algoritmo de descenso (alg.~1)
recupera por sí solo la separación angular óptima desde
inicialización aleatoria. En la práctica, para
$(d, K) = (4, 24)$, el descenso desde una inicialización gaussiana se
estanca de forma reproducible en $\theta_{\min}\approx 55{,}23^\circ$
(mínimo local con energía $208{,}352$ vs $208{,}337$ de $D_4$). El
código del proyecto resuelve este caso inicializando desde la
configuración canónica conocida ($D_4$, $E_8$, etc.) y dejando que
el descenso refine.

Añadir un párrafo:

> Para los pares $(d, K)$ con configuración óptima cerrada
> conocida ---$D_4$ en $d{=}4$, $E_8$ en $d{=}8$, raíces del retículo
> Leech en $d{=}24$--- el descenso desde inicialización aleatoria
> presenta cuencas espurias que retiene con energía a $0{,}007\%$
> del óptimo y separación angular sub-óptima. Para esos casos el
> protocolo arranca desde la configuración canónica y verifica que
> el descenso la conserva; para las dimensiones reales del modelo
> ($d{=}768$, $K\in\{9,12\}$) la inicialización es aleatoria y la
> separación recuperada se reporta en la \cref{tab:geo}.

---

## Nota #10 — ETF y código esférico colapsan en régimen $K \ll d$ (§617)

El paper plantea la ablación (iii) "Código esférico frente a marco
equiangular ajustado frente a inicialización aleatoria". En las
dimensiones del modelo ($d{=}768$, $K\in\{9,12\}$) el minimizador de
Riesz coincide numéricamente con el ETF: ambos tienen
$\theta_{\min} = \arccos(-1/(K-1))$. La ablación efectiva se reduce a
**código esférico/ETF (indistinguibles) vs inicialización aleatoria**.

Reescribir el ítem (iii) como:

> (iii) Código esférico frente a inicialización aleatoria. En las
> dimensiones del modelo ($K \ll d$) el código de Riesz coincide
> numéricamente con el marco equiangular ajustado, por lo que la
> tercera opción no se reporta como condición separada.

---

## Nota #11 — Algoritmo de descenso real (§423 alg.~1)

El pseudocódigo actual describe SGD ingenuo con renormalización por
paso. La implementación efectiva usa, por razones de convergencia en
alta dimensión:

- optimizador Adam,
- proyección del gradiente al espacio tangente a $\Sph^{d-1}$ antes
  del paso,
- planificador de tasa de aprendizaje *cosine annealing*.

Reescribir el alg.~1 para reflejar esto:

```
Require: dimensión d, tamaño K, exponente s, pasos T, tasa eta_0
inicializar u_1..u_K aleatorias en S^{d-1}
optimizador Adam con tasa eta_0
planificador cosine annealing
for t = 1..T:
    g_i = grad_{u_i} E_s
    g_i^tan = g_i - <g_i, u_i> u_i      # proyección tangente
    u_i = paso de Adam con g_i^tan
    u_i = u_i / ||u_i||                  # reproyección a S^{d-1}
    actualizar tasa con el planificador
return {u_1..u_K}
```

El motivo (proyección tangente + Adam) puede mencionarse en una
nota al pie: "El optimizador adaptativo y la proyección tangente
mejoran la convergencia frente al SGD ingenuo en $d \gg 1$, donde la
energía de Riesz tiene un paisaje altamente no convexo."
