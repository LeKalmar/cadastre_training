#!/bin/bash
# Nuit du 2026-11-08 : choix automatique du réglage de perte (sweep 3 époques), puis
# entraînement complet DINOv3-Small 60 époques, puis évaluation finale.
# Journal : work_dirs/night/night.log
cd "/home/lekalmar/BACKUP WINDOWS FILES/cadastre_training"
mkdir -p work_dirs/night
LOG=work_dirs/night/night.log
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# 1. attendre la fin du sweep (il écrit work_dirs/sweep.json en dernier)
log "attente de la fin du sweep V1/V2..."
until [ -s work_dirs/sweep.json ]; do sleep 30; done
log "sweep terminé"

# 2. choisir le réglage : meilleur taux de parcelles fermées (seuil optimal)
read ALPHA BETA PW CHOICE <<< "$(.venv/bin/python - <<'EOF'
import json
r = json.load(open("work_dirs/sweep.json"))
cfg = {"complet_ep5(a25b75)": (0.25, 0.75, 2.0), "V1_ep3(a50b50)": (0.5, 0.5, 2.0), "V2_ep3(a40b60,pw1)": (0.4, 0.6, 1.0)}
score = {}
for k, v in r.items():
    name = k.split(" (x")[0]
    score[name] = v["by_threshold"][str(v["best_threshold"])]["tout"]["parcelles"]
best = max(score, key=score.get)
a, b, pw = cfg[best]
print(a, b, pw, best.replace(" ", "_"))
EOF
)"
log "scores sweep : $(.venv/bin/python -c "
import json; r=json.load(open('work_dirs/sweep.json'))
print({k.split(' (x')[0]: round(v['by_threshold'][str(v['best_threshold'])]['tout']['parcelles'],3) for k,v in r.items()})")"
log "réglage retenu : $CHOICE -> alpha=$ALPHA beta=$BETA pos_weight=$PW"

# 3. entraînement complet
WD=work_dirs/v2_dinov3_small_night
mkdir -p $WD
log "entraînement complet -> $WD"
.venv/bin/python -u train.py --data-root data/cadastre_lines_v2 --work-dir $WD \
  --encoder tu-convnext_small.dinov3_lvd1689m --epochs 60 --warmup-epochs 3 --batch-size 8 --lr 3e-4 \
  --alpha $ALPHA --beta $BETA --gamma 1.5 --pos-weight $PW --w-tversky 0.2 --w-skel 0.6 --w-bce 0.2 \
  --text-weight 3 --patience 15 > $WD/train.out 2>&1
log "entraînement terminé (code $?) : $(tail -1 $WD/train_log.csv)"

# 4. évaluation finale (mêmes 1 471 tuiles de validation)
.venv/bin/python -u tools/compare_models.py --batch 8 --data data/cadastre_lines_v2 \
  --model "Bertrand=work_dirs/Bertrand_Couderc_mitb3_topology/best_model.pth" \
  --model "AB_dinov3_ep7=work_dirs/v2_dinov3_small_test/best_model.pth" \
  --model "nuit_best=$WD/best_model.pth" \
  --model "nuit_last=$WD/latest.pth" \
  --scales 1 0.5 --thresholds 0.3 0.4 0.5 0.6 \
  --out work_dirs/night/final.json --viz work_dirs/night/final.png > work_dirs/night/final.out 2>&1
log "évaluation finale terminée"
echo FINI >> "$LOG"
