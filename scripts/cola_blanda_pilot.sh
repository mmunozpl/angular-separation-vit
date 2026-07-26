#!/usr/bin/env bash
# piloto de la inercia (4o apoyo): UNA blanda (seed42) tras la cola base.
# espera COLA_BASE_CLEAN_DONE para no competir por la gpu, luego corre
# blanda42. el veredicto (theta_min>=80 y |ds_func|<=0.015) decide si se
# sueltan las 9 restantes (blanda43-46 + dura x5) o se replantea el 4o
# apoyo. dura y el resto de blanda ESPERAN este piloto.
set -u
cd /media/manpla/Pruebas/Hiperesferas
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch28
export PYTHONPATH=/media/manpla/Pruebas/Hiperesferas

BASELOG=/tmp/cola_base_clean.log
LOG=/tmp/cola_blanda_pilot.log
echo "=== piloto blanda en espera de la cola base $(date) ===" > "$LOG"
# espera el sentinela de la cola base
until grep -q "COLA_BASE_CLEAN_DONE" "$BASELOG" 2>/dev/null; do
  sleep 120
done
echo "=== base lista; arranca piloto blanda seed42 $(date) ===" | tee -a "$LOG"

RUN="attnA_blanda_seed42"
CK="artifacts/checkpoints/vitb_clean/${RUN}_last.pt"
if [ -f "$CK" ]; then
  echo "[salto] $RUN ya existe" | tee -a "$LOG"
else
  python scripts/run_attn.py --config configs/attn_vitb_blanda_clean.yaml \
    --seed 42 --run-name "$RUN" --save-every 0 >> "$LOG" 2>&1
  [ $? -ne 0 ] && echo "### $RUN FALLÓ ###" | tee -a "$LOG"
fi
echo "=== BLANDA_PILOT_DONE $(date) ===" | tee -a "$LOG"
