#!/usr/bin/env bash
# cola DINOv2 (tercera columna del trio, n=3 seeds 42-44). se encadena
# tras la cola ViT-L: espera a COLA_VITL_DONE (o, si /tmp se borro, a
# que dura44 de vitl tenga 30 epocas y no quede proceso de cola_vitl).
# solo brazo BASE congelado (la columna no se afina: blanda/dura no
# aplican) + fases de senal con --arch dinov2. idempotente; aborta si
# una corrida queda incompleta.
set -u
cd /media/manpla/Pruebas/Hiperesferas
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch28
export PYTHONPATH=/media/manpla/Pruebas/Hiperesferas
LOG=/tmp/cola_dinov2.log
echo "=== cola dinov2 arranca $(date) ===" > "$LOG"

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

VITL_FIN=artifacts/logs/vitl_clean/attnA_dura_seed44/run.csv
while true; do
  if grep -q COLA_VITL_DONE /tmp/cola_vitl.log 2>/dev/null; then
    break
  fi
  if ! pgrep -f "bash scripts/cola_vitl.sh" >/dev/null \
     && [ "$(efectivas "$VITL_FIN")" -ge 30 ]; then
    break
  fi
  sleep 300
done
echo "=== vitl cerrada, arranca dinov2 $(date) ===" | tee -a "$LOG"

run_one () {
  local variant="$1" seed="$2" cfg="$3"
  local run="attnA_${variant}_seed${seed}"
  local rc="artifacts/logs/dinov2_clean/${run}/run.csv"
  if [ "$(efectivas "$rc")" -ge 30 ]; then
    echo "[salto] $run ya tiene 30 épocas efectivas" | tee -a "$LOG"
    return
  fi
  rm -f "artifacts/checkpoints/dinov2_clean/${run}_last.pt" \
        "artifacts/checkpoints/dinov2_clean/${run}_nan_capture.pt"
  rm -rf "artifacts/logs/dinov2_clean/${run}"
  echo "=== $run arranca $(date) ===" | tee -a "$LOG"
  python scripts/run_attn.py --config "$cfg" --seed "$seed" \
    --run-name "$run" --save-every 0 >> "$LOG" 2>&1
  local n=$(($(wc -l < "$rc" 2>/dev/null || echo 1)-1))
  echo "=== $run fin $(date): ${n}/30 ===" | tee -a "$LOG"
  if [ "$n" -lt 30 ]; then
    echo "=== COLA_DINOV2_ABORT: $run incompleta (${n}/30) ===" \
      | tee -a "$LOG"
    exit 1
  fi
}

# 1) base congelada n=3 (ancla de tarea; la columna no cambia)
for s in 42 43 44; do
  run_one base "$s" configs/attn_dinov2_base_clean.yaml
done

# 2) fases de senal sobre la columna congelada
echo "=== fases premisa dinov2 n=3 $(date) ===" | tee -a "$LOG"
python scripts/run_fase_G.py --arch dinov2 >> "$LOG" 2>&1
python scripts/run_fase_0.py --arch dinov2 >> "$LOG" 2>&1
python scripts/dissociation_D.py --variants base \
  --seeds 42 43 44 --arch dinov2 >> "$LOG" 2>&1
echo "=== fases premisa dinov2 listas $(date) ===" | tee -a "$LOG"

# 3) ablacion (solo fila base; blanda/dura no aplican congelado)
echo "=== ablation_table_A dinov2 $(date) ===" | tee -a "$LOG"
python scripts/ablation_table_A.py --arch dinov2 >> "$LOG" 2>&1
echo "=== COLA_DINOV2_DONE $(date) ===" | tee -a "$LOG"
