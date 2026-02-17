import rasterio
import numpy as np
from PIL import Image
import argparse

Image.MAX_IMAGE_PIXELS = None

def merge_png_to_tif(original_tif, ai_png, output_path):
    # 1. Lire les coordonnées du TIF original
    with rasterio.open(original_tif) as src:
        profile = src.profile.copy()
        print(f"🌍 Coordonnées récupérées depuis {original_tif}")
        print(f"📏 Taille attendue : {src.width}x{src.height}")

    # 2. Lire le masque PNG généré par l'IA
    img_png = Image.open(ai_png).convert('L') # Force en niveaux de gris
    mask_data = np.array(img_png)
    
    # Vérification de sécurité
    if mask_data.shape[0] != profile['height'] or mask_data.shape[1] != profile['width']:
        print("⚠️ Attention : Les dimensions ne correspondent pas parfaitement !")
        # On force le redimensionnement si nécessaire pour que ça rentre dans le cadre
        img_png = img_png.resize((profile['width'], profile['height']), Image.NEAREST)
        mask_data = np.array(img_png)

    # 3. Préparer le profil de sortie propre pour QGIS
    profile.update(
        dtype=rasterio.uint8,
        count=1,          # Une seule bande (Noir et Blanc)
        compress='lzw',   # Pour que le fichier soit léger
        nodata=0          # Pour que le fond soit transparent dans QGIS
    )

    # 4. Écrire le nouveau TIFF
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(mask_data, 1)
    
    print(f"✅ Terminé ! Fichier prêt : {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tif', required=True, help='Le TIFF original')
    parser.add_argument('--png', required=True, help='Le PNG de l\'IA')
    parser.add_argument('--out', required=True, help='Nom du fichier final')
    args = parser.parse_args()
    merge_png_to_tif(args.tif, args.png, args.out)