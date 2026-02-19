import os
from collections import Counter
from pathlib import Path

# --- CONFIGURATION ---
# Remplace par le chemin vers ton dossier d'images ou de masques
data_dir = r"data\cadastre_lines_tiled\images"


# ---------------------

def count_tiles_per_sheet(directory):
    path = Path(directory)
    if not path.exists():
        print(f"❌ Erreur : Le dossier {directory} n'existe pas.")
        return

    # Liste tous les fichiers (images)
    files = [f.stem for f in path.iterdir() if f.is_file()]

    sheets = []
    for f in files:
        # On coupe au premier '_y' pour garder tout ce qui précède (le nom de la feuille)
        sheet_name = f.split('_y')[0]
        sheets.append(sheet_name)

    # Compte les occurrences
    counts = Counter(sheets)

    # Affichage des résultats
    print(f"\n📊 Analyse du dossier : {directory}")
    print(f"{'Nom de la Feuille':<40} | {'Nombre de Tuiles':<15}")
    print("-" * 60)

    total_tiles = 0
    # Tri par nom de feuille
    for sheet, count in sorted(counts.items()):
        print(f"{sheet:<40} | {count:<15}")
        total_tiles += count

    print("-" * 60)
    print(f"{'TOTAL':<40} | {total_tiles:<15}")
    print(f"Nombre de feuilles uniques : {len(counts)}")


if __name__ == "__main__":
    count_tiles_per_sheet(data_dir)