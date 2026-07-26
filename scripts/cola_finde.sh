#!/usr/bin/env bash
# cola del confirmatorio (fin de semana, GPU libre). secuencial = sin
# solapamiento. idempotente: salta lo que ya tiene 30 épocas en run.csv
# (no por existir _last.pt, que se guarda por época). borra residuo antes
# de cada lanzamiento (atómico, fuera de compuesto que truncaría).
# train_attn tiene NaN-skip + guard n_batches=0 (la blanda degenera la
# SVD post-convergencia y salta épocas finales; el guard deja completar
# con métricas convergidas válidas). dura no sufre eso (lambda=0, sin
# backprop por SVD).
set -u
cd /media/manpla/Pruebas/Hiperesferas
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch28
export PYTHONPATH=/media/manpla/Pruebas/Hiperesferas
LOG=/tmp/cola_finde.log
echo "=== cola finde arranca $(date) ===" > "$LOG"

run_one () {
  local variant="$1" seed="$2" cfg="$3"
  local run="attnA_${variant}_seed${seed}"
  local rc="artifacts/logs/vitb_clean/${run}/run.csv"
  # idempotencia por epocas EFECTIVAS (ce > 0), no por filas: una
  # corrida bloqueada por skip escribe filas idle con ce==0 y llega a
  # 30 filas sin 30 epocas (dura44/45 del 11-07). la columna 4 = ce se
  # verifica contra la cabecera antes de contar.
  if [ -f "$rc" ]; then
    local col_ce
    col_ce=$(head -1 "$rc" | awk -F, \
      '{for (i=1; i<=NF; i++) if ($i == "ce") print i}')
    if [ -n "$col_ce" ]; then
      local efectivas
      # ($c+0) fuerza comparacion numerica ("0.0000" como cadena es
      # mayor que "0") y LC_ALL=C fuerza el punto decimal (con
      # es_ES, mawk lee "1.25" como 1 y "0.99" como 0)
      efectivas=$(LC_ALL=C awk -F, -v c="$col_ce" \
        'NR>1 && ($c+0)>0' "$rc" | wc -l)
      if [ "$efectivas" -ge 30 ]; then
        echo "[salto] $run ya tiene 30 épocas efectivas" \
          | tee -a "$LOG"; return
      fi
      echo "[re-corre] $run: ${efectivas}/30 efectivas" | tee -a "$LOG"
    fi
  fi
  rm -f "artifacts/checkpoints/vitb_clean/${run}_last.pt" \
        "artifacts/checkpoints/vitb_clean/${run}_nan_capture.pt"
  rm -rf "artifacts/logs/vitb_clean/${run}"
  echo "=== $run arranca $(date) ===" | tee -a "$LOG"
  python scripts/run_attn.py --config "$cfg" --seed "$seed" \
    --run-name "$run" --save-every 0 >> "$LOG" 2>&1
  local n=$(($(wc -l < "$rc" 2>/dev/null || echo 1)-1))
  echo "=== $run fin $(date): ${n}/30 ===" | tee -a "$LOG"
}

# 1) fases baratas a n=5 (premisa; solo necesitan las 5 base, ya hechas)
echo "=== fases premisa n=5 $(date) ===" | tee -a "$LOG"
python scripts/run_fase_G.py >> "$LOG" 2>&1
python scripts/run_fase_0.py >> "$LOG" 2>&1
python scripts/dissociation_D.py --variants base \
  --seeds 42 43 44 45 46 --arch vitb >> "$LOG" 2>&1
echo "=== fases premisa listas $(date) ===" | tee -a "$LOG"

# 2) blanda 43-46 (42 hecho) y dura 42-46 (la sonda, tab:divattn)
for s in 43 44 45 46; do
  run_one blanda "$s" configs/attn_vitb_blanda_clean.yaml
done
for s in 42 43 44 45 46; do
  run_one dura "$s" configs/attn_vitb_dura_clean.yaml
done

# 3) ablación a n=5 (lee base/blanda/dura run.csv)
echo "=== ablation_table_A n=5 $(date) ===" | tee -a "$LOG"
python scripts/ablation_table_A.py --arch vitb >> "$LOG" 2>&1
echo "=== COLA_FINDE_DONE $(date) ===" | tee -a "$LOG"
