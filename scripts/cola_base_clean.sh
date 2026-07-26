#!/usr/bin/env bash
# completa el ancla ViT-B base limpio: seeds 43-46 (la 42 ya está).
# secuencial, 1 gpu. ~5h/corrida (caché caliente, ep1 ~9min si la RAM
# retiene IN-100 entre corridas). alimenta fase_G/fase_0/dissociation a
# n=5.
set -u
cd /media/manpla/Pruebas/Hiperesferas
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch28
export PYTHONPATH=/media/manpla/Pruebas/Hiperesferas

LOG=/tmp/cola_base_clean.log
echo "=== cola base limpio arranca $(date) ===" > "$LOG"

for SEED in 43 44 45 46; do
  RUN="attnA_base_seed${SEED}"
  CK="artifacts/checkpoints/vitb_clean/${RUN}_last.pt"
  if [ -f "$CK" ]; then
    echo "[salto] $RUN ya existe" | tee -a "$LOG"; continue
  fi
  echo "=== $RUN arranca $(date) ===" | tee -a "$LOG"
  python scripts/run_attn.py --config configs/attn_vitb_base_clean.yaml \
    --seed "$SEED" --run-name "$RUN" --save-every 0 >> "$LOG" 2>&1
  if [ $? -ne 0 ]; then
    echo "### $RUN FALLÓ ###" | tee -a "$LOG"
  else
    echo "=== $RUN listo $(date) ===" | tee -a "$LOG"
  fi
done
echo "=== COLA_BASE_CLEAN_DONE $(date) ===" | tee -a "$LOG"
