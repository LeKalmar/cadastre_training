#!/usr/bin/env bash
# Relance ce qui est mort parmi le géoréférencement et l'extraction de la nuit.
# Détaché (setsid) et plafonné en mémoire : une extraction monte à ~15 Go.
# Usage : tools/georef/nuit_relance.sh   (ne relance rien de ce qui est vivant)
cd "$(dirname "$0")/../.." || exit 1
L=logs/nuit_20261111
N="/home/lekalmar/BACKUP WINDOWS FILES/Travail/Cartographie/INDEXATION CADASTRE/Cadastre XIX/Cartes/Communes/59 - Nord"
alive() { [ -n "$1" ] && kill -0 "$1" 2>/dev/null; }
G=$(cat $L/pid_georef 2>/dev/null); E=$(cat $L/pid_extraction 2>/dev/null)

if ! alive "$G" && ! grep -q "^TERMINE" $L/georef.log; then
  setsid bash -c 'echo $$ > '$L'/pid_georef; exec systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=2G .venv-georef/bin/python tools/georef/run.py --src "'"$N"'" --out data/georef_nord --jobs 4 --sauter-faits' >> $L/georef.log 2>&1 < /dev/null &
  sleep 2; G=$(cat $L/pid_georef)
  echo "$(date +%H:%M) relance géoréférencement : $G" | tee -a $L/journal.txt
fi
if ! alive "$E"; then
  setsid bash -c 'echo $$ > '$L'/pid_extraction; exec .venv/bin/python tools/georef/importer_lot.py --out data/georef_nord --lancer --workers 1 --memoire-libre 15 --attendre-pid '"$G" >> $L/extraction.log 2>&1 < /dev/null &
  sleep 2; E=$(cat $L/pid_extraction)
  echo "$(date +%H:%M) relance extraction : $E" | tee -a $L/journal.txt
fi
echo "$G $E" > $L/pids
echo "georef $G $(alive $G && echo vivant || echo fini) ; extraction $E $(alive $E && echo vivant || echo fini)"
