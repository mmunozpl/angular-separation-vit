#!/usr/bin/env bash
# cola ViT-L (trio multi-arquitectura, n=3 seeds 42-44). se encadena
# tras la cola ViT-B: espera a COLA_FINDE_DONE (o, si /tmp se borro, a
# que dura46 tenga 30 epocas y no quede proceso de cola_finde). orden:
# base x3 -> fases de senal (--arch vitl) -> blanda x3 -> dura x3 ->
# ablation. idempotente como cola_finde (salta 30/30, borra residuo).
# ABORTA si una corrida termina incompleta (OOM/crash): no quema dias
# de GPU en corridas rotas.
set -u
cd /media/manpla/Pruebas/Hiperesferas
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch28
export PYTHONPATH=/media/manpla/Pruebas/Hiperesferas
LOG=/tmp/cola_vitl.log
echo "=== cola vitl arranca $(date) ===" > "$LOG"

# epocas efectivas = filas con ce > 0 (una corrida bloqueada por skip
# escribe filas idle con ce==0: 30 filas no son 30 epocas). el indice
# de la columna ce se lee de la cabecera, no se asume.
efectivas () {
  local rc="$1"
  [ -f "$rc" ] || { echo 0; return; }
  local col_ce
  col_ce=$(head -1 "$rc" | awk -F, \
    '{for (i=1; i<=NF; i++) if ($i == "ce") print i}')
  [ -n "$col_ce" ] || { echo 0; return; }
  # ($c+0) fuerza comparacion numerica ("0.0000" como cadena es mayor
  # que "0") y LC_ALL=C fuerza el punto decimal (con es_ES, mawk lee
  # "1.25" como 1 y "0.99" como 0)
  LC_ALL=C awk -F, -v c="$col_ce" 'NR>1 && ($c+0)>0' "$rc" | wc -l
}

# --- espera a que la cola ViT-B cierre -------------------------------
VITB_DURA46=artifacts/logs/vitb_clean/attnA_dura_seed46/run.csv
while true; do
  if grep -q COLA_FINDE_DONE /tmp/cola_finde.log 2>/dev/null; then
    break
  fi
  if ! pgrep -f "bash scripts/cola_finde.sh" >/dev/null \
     && [ "$(efectivas "$VITB_DURA46")" -ge 30 ]; then
    break
  fi
  sleep 300
done
echo "=== vitb cerrada, arranca vitl $(date) ===" | tee -a "$LOG"

run_one () {
  local variant="$1" seed="$2" cfg="$3"
  local run="attnA_${variant}_seed${seed}"
  local rc="artifacts/logs/vitl_clean/${run}/run.csv"
  if [ "$(efectivas "$rc")" -ge 30 ]; then
    echo "[salto] $run ya tiene 30 épocas efectivas" | tee -a "$LOG"
    return
  fi
  rm -f "artifacts/checkpoints/vitl_clean/${run}_last.pt" \
        "artifacts/checkpoints/vitl_clean/${run}_nan_capture.pt"
  rm -rf "artifacts/logs/vitl_clean/${run}"
  echo "=== $run arranca $(date) ===" | tee -a "$LOG"
  python scripts/run_attn.py --config "$cfg" --seed "$seed" \
    --run-name "$run" --save-every 0 >> "$LOG" 2>&1
  local n=$(($(wc -l < "$rc" 2>/dev/null || echo 1)-1))
  echo "=== $run fin $(date): ${n}/30 ===" | tee -a "$LOG"
  if [ "$n" -lt 30 ]; then
    echo "=== COLA_VITL_ABORT: $run incompleta (${n}/30) ===" \
      | tee -a "$LOG"
    exit 1
  fi
}

# 1) base n=3 (ancla de las fases de senal)
for s in 42 43 44; do
  run_one base "$s" configs/attn_vitl_base_clean.yaml
done

# 2) fases de senal baratas sobre las base (sin reentrenar)
echo "=== fases premisa vitl n=3 $(date) ===" | tee -a "$LOG"
python scripts/run_fase_G.py --arch vitl >> "$LOG" 2>&1
python scripts/run_fase_0.py --arch vitl >> "$LOG" 2>&1
python scripts/dissociation_D.py --variants base \
  --seeds 42 43 44 --arch vitl >> "$LOG" 2>&1
echo "=== fases premisa vitl listas $(date) ===" | tee -a "$LOG"

# 3) blanda y dura n=3 (sonda de inercia + columna dura)
for s in 42 43 44; do
  run_one blanda "$s" configs/attn_vitl_blanda_clean.yaml
done
for s in 42 43 44; do
  run_one dura "$s" configs/attn_vitl_dura_clean.yaml
done

# 4) ablacion vitl (merge-por-arch en el csv comun)
echo "=== ablation_table_A vitl $(date) ===" | tee -a "$LOG"
python scripts/ablation_table_A.py --arch vitl >> "$LOG" 2>&1
echo "=== COLA_VITL_DONE $(date) ===" | tee -a "$LOG"
