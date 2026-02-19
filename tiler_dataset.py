import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def tile_dataset(base_dir, output_dir, tile_size=512, overlap=0):
    # Configuration des chemins
    img_src = Path(base_dir) / "images"
    mask_src = Path(base_dir) / "labels"
    
    img_out = Path(output_dir) / "images"
    mask_out = Path(output_dir) / "labels"
    
    # Création des dossiers de sortie
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)
    
    # Extensions supportées
    valid_ext = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    image_files = [f for f in img_src.iterdir() if f.suffix.lower() in valid_ext]
    
    print(f"🚀 Début du tuilage de {len(image_files)} images...")
    
    stride = tile_size - overlap
    total_tiles = 0

    for img_path in tqdm(image_files):
        # 1. Charger l'image
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        # 2. Chercher le masque correspondant (même nom)
        # On essaie plusieurs extensions au cas où le masque est en .png et l'image en .jpg
        mask = None
        for ext in ['.png', '.jpg', '.png']:
            potential_mask = mask_src / (img_path.stem + ext)
            if potential_mask.exists():
                mask = cv2.imread(str(potential_mask), cv2.IMREAD_GRAYSCALE)
                break
        
        if mask is None:
            print(f"⚠️ Masque introuvable pour {img_path.name}, passage à la suivante.")
            continue

        h, w = img.shape[:2]
        
        # 3. Boucle de découpe
        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                tile_img = img[y:y+tile_size, x:x+tile_size]
                tile_mask = mask[y:y+tile_size, x:x+tile_size]
                
                # OPTIONNEL: Filtrage des tuiles vides
                # On garde la tuile si elle n'est pas toute blanche (image) 
                # ET s'il y a au moins un peu de tracé (masque)
                if np.mean(tile_img) < 250 or np.any(tile_mask > 0):
                    tile_name = f"{img_path.stem}_y{y}_x{x}.png"
                    
                    cv2.imwrite(str(img_out / tile_name), tile_img)
                    cv2.imwrite(str(mask_out / tile_name), tile_mask)
                    total_tiles += 1

    print(f"✅ Terminé ! {total_tiles} tuiles créées dans {output_dir}")

if __name__ == "__main__":
    # Adapte les chemins ici si nécessaire
    tile_dataset(
        base_dir="data/dataset_remaster",
        output_dir="data/cadastre_lines_tiled", 
        tile_size=512, 
        overlap=64 # Un peu d'overlap aide l'IA à voir les lignes coupées
    )