#!/usr/bin/env python3
"""Génère les icônes SVG de la barre « Validation annotation » (annotation_cadastre/icones/).

Une icône par action (nom = objectName sans « annotation_ »). Pictogramme 24x24, couleur
par famille d'outils, et la TOUCHE de l'outil dans un badge en bas à droite (pour retenir les
raccourcis). Relancer après une modification : python3 tools/annotation/generer_icones.py
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent / "annotation_cadastre" / "icones"

FAMILLES = {  # couleur du pictogramme et du badge
    "valid": "#2b8a3e", "rejet": "#c92a2a", "neutre": "#495057", "outil": "#1971c2",
    "feuille": "#1864ab", "dev": "#7048e8", "calage": "#e8590c", "zone": "#f08c00",
    "numero": "#d6336c", "main": "#212529",
}


def badge(touche, couleur):
    if not touche:
        return ""
    return (f'<rect x="14.5" y="14.5" width="9" height="9" rx="2" fill="{couleur}" stroke="#fff" stroke-width="1"/>'
            f'<text x="19" y="21.6" font-family="DejaVu Sans, Arial, sans-serif" font-size="7.5" '
            f'font-weight="bold" fill="#fff" text-anchor="middle">{touche}</text>')


def svg(corps, touche, couleur):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">'
            f'<g fill="none" stroke="{couleur}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'{corps}</g>{badge(touche, couleur)}</svg>')


CHECK = '<path d="M3 9.5 L7 13.5 L15 4.5"/>'
CROIX = '<path d="M4 4 L13 13 M13 4 L4 13"/>'
FLECHE = '<path d="M3 9 H13 M9.5 5 L13.5 9 L9.5 13"/>'
GRILLE = '<path d="M2 6 H16 M2 11 H16 M6.5 2 V15 M11.5 2 V15" stroke-width="1.4" opacity=".55"/>'
ZONE = '<path d="M3 4 L14 2.5 L16 12 L8 15.5 L2 11 Z" stroke-dasharray="2.2 1.8" stroke-width="1.6"/>'
PARCELLE = ('<rect x="2" y="2.5" width="14" height="12" stroke-width="2.4"/>'
            '<path d="M9 2.5 V14.5 M2 8.5 H9" stroke-width="1" opacity=".7"/>')

ICONES = {
    # validation des propositions IA
    "valider": ("valid", "1", CHECK + '<circle cx="9" cy="9" r="8" stroke-width="1.2" opacity=".35"/>'),
    "rejeter": ("rejet", "2", CROIX + '<circle cx="8.5" cy="8.5" r="8" stroke-width="1.2" opacity=".35"/>'),
    "attente": ("neutre", "3", '<path d="M4 2 H14 M4 16 H14" stroke-width="2.2"/>'
                '<path d="M5.5 2.5 L12.5 2.5 L9 8.8 Z M9 9.2 L12.5 15.5 L5.5 15.5 Z" stroke-width="1.5"/>'
                '<path d="M7.4 14.6 L10.6 14.6 L9 12 Z" fill="#495057" stroke="none"/>'),
    "suivante": ("outil", "4", FLECHE + '<path d="M16 3 V15"/>'),
    "outil": ("outil", "5", '<path d="M3 2 L3 15 L6.5 11.5 L9 16.5 L11 15.5 L8.6 10.6 L13.5 10.6 Z" stroke-width="1.6"/>'),
    "sommets": ("outil", "6", '<path d="M3 13 L5 4 L13 3 L15 11 Z" stroke-width="1.3"/>'
                '<g fill="#1971c2" stroke="none"><circle cx="3" cy="13" r="2"/><circle cx="5" cy="4" r="2"/>'
                '<circle cx="13" cy="3" r="2"/><circle cx="15" cy="11" r="2"/></g>'),
    "developpement": ("dev", "7", '<rect x="2" y="2" width="5" height="5" stroke-width="1.6"/>'
                      '<rect x="8" y="8" width="8.5" height="8.5" stroke-width="1.6"/><path d="M5 9.5 L5 12 M5 12 H6.8" stroke-width="1.3"/>'
                      '<path d="M7.5 7.5 L10 10" stroke-width="1.4"/>'),
    "marquer_dev": ("dev", "0", '<path d="M2 3 H10 L15.5 8.5 L9.5 14.5 L2 7 Z" stroke-width="1.7"/>'
                    '<circle cx="5.5" cy="6" r="1.2" fill="#7048e8" stroke="none"/>'),
    # feuilles
    "feuille": ("feuille", "8", '<path d="M3 2 H11 L14 5 V16 H3 Z" stroke-width="1.6"/>'
                '<circle cx="8.5" cy="10" r="2.6" stroke-width="1.4"/><path d="M8.5 5.5 V7 M8.5 13 V14.5 M4 10 H5.6 M11.4 10 H13" stroke-width="1.4"/>'),
    "toutes": ("feuille", "9", '<path d="M6 2 H14 V12" stroke-width="1.4" opacity=".6"/>'
               '<path d="M4 4 H12 V14" stroke-width="1.4" opacity=".8"/><rect x="2" y="6" width="8" height="10" stroke-width="1.7"/>'),
    "actualiser": ("feuille", "", '<path d="M18 7 A8 8 0 1 0 19 13"/><path d="M18 2.5 V7.5 H13"/>'),
    # calages du géoréférencement
    "calage_suivant": ("calage", "C", GRILLE + FLECHE),
    "calage_bon": ("valid", "B", GRILLE + CHECK),
    "calage_faux": ("rejet", "F", GRILLE + CROIX),
    # zone utile
    "zone_suivante": ("zone", "Z", ZONE + '<path d="M6 9 H12 M9.5 6.5 L12 9 L9.5 11.5" stroke-width="1.6"/>'),
    "zone_valider": ("valid", "V", ZONE + '<path d="M5.5 9 L8 11.5 L12.5 6.5" stroke-width="1.8"/>'),
    "zone_rejeter": ("rejet", "X", ZONE + '<path d="M6 6 L12 12 M12 6 L6 12" stroke-width="1.8"/>'),
    # numéros de parcelles
    "numeros": ("numero", "N", '<path d="M9 16 C9 16 3.5 10.5 3.5 7 A5.5 5.5 0 0 1 14.5 7 C14.5 10.5 9 16 9 16 Z" stroke-width="1.7"/>'
                '<text x="9" y="9.6" font-family="DejaVu Sans, Arial, sans-serif" font-size="6.5" font-weight="bold" '
                'fill="#d6336c" stroke="none" text-anchor="middle">12</text>'),
    # parcelles principales
    "main_suivante": ("main", "P", PARCELLE + '<path d="M17 9 H21 M19.5 7 L21.5 9 L19.5 11" stroke-width="1.6"/>'),
    "main_valider": ("valid", "M", PARCELLE + '<path d="M4.5 9.5 L8 13 L14 5.5" stroke="#2b8a3e" stroke-width="2.2"/>'),
    "main_regrouper": ("main", "G", '<rect x="2" y="3" width="6" height="11" stroke-width="1.6"/>'
                       '<rect x="10" y="3" width="6" height="11" stroke-width="1.6"/><path d="M1 1.5 H17 V15.5 H1 Z" stroke-width="1" stroke-dasharray="1.5 1.5"/>'),
    "main_detacher": ("main", "U", '<rect x="1.5" y="3" width="6" height="11" stroke-width="1.6"/>'
                      '<rect x="11" y="3" width="6" height="11" stroke-width="1.6"/>'
                      '<path d="M9.2 2 V15" stroke-width="1.2" stroke-dasharray="1.6 1.4"/>'),
    "main_type": ("main", "T", '<rect x="2" y="2" width="6.5" height="6.5" fill="#e64646" stroke="none"/>'
                  '<rect x="9.5" y="2" width="6.5" height="6.5" fill="#5aaae6" stroke="none"/>'
                  '<rect x="2" y="9.5" width="6.5" height="6.5" fill="#3caa3c" stroke="none"/>'
                  '<rect x="9.5" y="9.5" width="6.5" height="6.5" fill="#961e3c" stroke="none"/>'),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for nom, (fam, touche, corps) in ICONES.items():
        c = FAMILLES[fam]
        (OUT / f"{nom}.svg").write_text(svg(corps, touche, c), encoding="utf-8")
    print(f"{len(ICONES)} icônes -> {OUT}")


if __name__ == "__main__":
    main()
