#!/usr/bin/env bash
# Sauvegarde complète de la base d'annotation (à lancer souvent, et avant toute
# remise à zéro du conteneur staging : la base vit dans son dossier de données).
#   tools/annotation/sauvegarder.sh
# Restauration : pg_restore -d "service=cadastre_annotation" --clean <fichier.dump>
set -euo pipefail
DEST="$(cd "$(dirname "$0")/../.." && pwd)/annotation_parcelles/sauvegardes"
mkdir -p "$DEST"
OUT="$DEST/cadastre_annotation_$(date +%Y%m%d-%H%M%S).dump"
docker exec terrapartus-terrapartus-db-1 sh -c 'pg_dump -U "$POSTGRES_USER" -Fc cadastre_annotation' > "$OUT"
ls -1t "$DEST"/*.dump | tail -n +31 | xargs -r rm --   # garde les 30 dernières
echo "Sauvegarde : $OUT ($(du -h "$OUT" | cut -f1))"
