#!/usr/bin/env bash
# Installe l'extension QGIS « Annotation cadastrale ».
#   tools/annotation/installer_extension.sh          lien symbolique dans le profil QGIS (par défaut) :
#                                                    toute modification du dépôt est prise en compte
#                                                    au prochain démarrage de QGIS (ou décocher/recocher).
#   tools/annotation/installer_extension.sh --zip    fabrique annotation_cadastre.zip, à installer sur une
#                                                    autre machine : Extensions > Installer depuis un ZIP.
set -euo pipefail
ICI="$(cd "$(dirname "$0")" && pwd)"
SRC="$ICI/annotation_cadastre"
if [[ "${1:-}" == "--zip" ]]; then
    OUT="$ICI/annotation_cadastre.zip"
    rm -f "$OUT"
    (cd "$ICI" && zip -qr "$OUT" annotation_cadastre -x '*__pycache__*')
    echo "Archive : $OUT"
    exit 0
fi
PROFIL="${QGIS_PROFIL:-$HOME/.local/share/QGIS/QGIS3/profiles/default}"
mkdir -p "$PROFIL/python/plugins"
ln -sfn "$SRC" "$PROFIL/python/plugins/annotation_cadastre"
echo "Lien : $PROFIL/python/plugins/annotation_cadastre -> $SRC"
INI="$PROFIL/QGIS/QGIS3.ini"
if pgrep -x 'qgis|qgis.bin' >/dev/null; then
    echo "QGIS est ouvert : activer l'extension dans Extensions > Gérer les extensions."
elif [[ -f "$INI" ]] && ! grep -q '^annotation_cadastre=' "$INI"; then
    if grep -q '^\[PythonPlugins\]' "$INI"; then
        sed -i '/^\[PythonPlugins\]/a annotation_cadastre=true' "$INI"
    else
        printf '\n[PythonPlugins]\nannotation_cadastre=true\n' >> "$INI"
    fi
    echo "Extension activée."
fi
