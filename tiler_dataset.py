"""
Tuilage du dataset cadastre — compatible tile_size 512 et 1024.

Usage typique pour 1024px :
    python tiler_dataset.py --base-dir data/dataset --output-dir data/cadastre_lines_tiled_1024  --tile-size 1024 --overlap 128 --min-pixels 200

Usage typique pour 512px :
    python tiler_dataset.py --base-dir data/dataset --output-dir data/cadastre_lines_tiled_512 --tile-size 512 --overlap 64 --min-pixels 50
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def tile_dataset(base_dir, output_dir, tile_size=1024, overlap=128, min_pixels=200):
    img_src  = Path(base_dir) / "images"
    mask_src = Path(base_dir) / "labels"
    img_out  = Path(output_dir) / "images"
    mask_out = Path(output_dir) / "labels"

    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    valid_ext    = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    image_files  = [f for f in img_src.iterdir() if f.suffix.lower() in valid_ext]

    stride = tile_size - overlap
    total_tiles = skipped = 0

    print(f"🚀 Tuilage de {len(image_files)} images")
    print(f"   tile_size={tile_size}  overlap={overlap}  stride={stride}  min_pixels={min_pixels}\n")

    for img_path in tqdm(image_files):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        mask = None
        for ext in ['.png', '.jpg', '.jpeg', '.tif']:
            p = mask_src / (img_path.stem + ext)
            if p.exists():
                mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                break

        if mask is None:
            print(f"  ⚠️  Masque introuvable : {img_path.name}")
            continue

        h, w = img.shape[:2]

        # Ignorer les images trop petites pour produire au moins une tuile
        if h < tile_size or w < tile_size:
            print(f"  ⚠️  Image trop petite ({w}×{h}) pour tile_size={tile_size} : {img_path.name}")
            continue

        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                tile_img  = img [y:y+tile_size, x:x+tile_size]
                tile_mask = mask[y:y+tile_size, x:x+tile_size]

                n_white = np.count_nonzero(tile_mask > 127)

                # Filtrage : garder si assez de lignes annotées OU si l'image
                # n'est pas quasi-blanche (tuile de bord de feuille utile)
                img_mean = np.mean(tile_img)
                if n_white < min_pixels and img_mean > 245:
                    skipped += 1
                    continue

                tile_name = f"{img_path.stem}_y{y}_x{x}.png"
                cv2.imwrite(str(img_out  / tile_name), tile_img)
                cv2.imwrite(str(mask_out / tile_name), tile_mask)
                total_tiles += 1

    print(f"\n✅ {total_tiles} tuiles créées  |  {skipped} tuiles vides ignorées")
    print(f"📂 Sortie : {output_dir}")


if __name__ == "__main__":
    P = argparse.ArgumentParser(description="Tuilage dataset cadastre")
    P.add_argument("--base-dir",    default="data/dataset",
                   help="Dossier source contenant images/ et labels/")
    P.add_argument("--output-dir",  default="data/cadastre_lines_tiled_1024",
                   help="Dossier de sortie des tuiles")
    P.add_argument("--tile-size",   type=int, default=1024,
                   help="Taille des tuiles en pixels (512 ou 1024, défaut: 1024)")
    P.add_argument("--overlap",     type=int, default=128,
                   help="Recouvrement entre tuiles (défaut: 128 = 12.5% de 1024)")
    P.add_argument("--min-pixels",  type=int, default=200,
                   help="Pixels blancs minimum dans le masque pour garder la tuile "
                        "(défaut: 200 pour 1024px, utiliser 50 pour 512px)")
    args = P.parse_args()

    tile_dataset(
        base_dir   = args.base_dir,
        output_dir = args.output_dir,
        tile_size  = args.tile_size,
        overlap    = args.overlap,
        min_pixels = args.min_pixels,
    )