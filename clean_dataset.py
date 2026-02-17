import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def clean_empty_tiles(dataset_dir, threshold=5):
    """
    Supprime les tuiles vides (noires) dans labels et les images correspondantes.
    threshold: nombre de pixels minimum pour considérer que la tuile n'est pas vide.
    """
    img_dir = Path(dataset_dir) / "images"
    mask_dir = Path(dataset_dir) / "labels"
    
    # On liste les masques
    mask_files = list(mask_dir.glob("*.png")) + list(mask_dir.glob("*.jpg"))
    
    print(f"🔍 Analyse de {len(mask_files)} tuiles...")
    removed_count = 0

    for mask_path in tqdm(mask_files):
        # 1. Lire le masque
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # 2. Vérifier s'il est vide (si la somme des pixels blancs est < au seuil)
        # On compte le nombre de pixels > 0 (le tracé des lignes)
        if np.count_nonzero(mask) < threshold:
            # 3. Déterminer le chemin de l'image correspondante
            # On cherche le même nom de fichier dans le dossier images
            img_path = img_dir / mask_path.name
            
            # 4. Suppression
            try:
                if mask_path.exists():
                    os.remove(mask_path)
                if img_path.exists():
                    os.remove(img_path)
                removed_count += 1
            except Exception as e:
                print(f"❌ Erreur lors de la suppression de {mask_path.name}: {e}")

    print(f"✅ Nettoyage terminé !")
    print(f"🗑️ Tuiles supprimées : {removed_count}")
    print(f"📂 Tuiles restantes : {len(mask_files) - removed_count}")

if __name__ == "__main__":
    # Remplacez par votre dossier de tuiles
    clean_empty_tiles("data/cadastre_lines_tiled")