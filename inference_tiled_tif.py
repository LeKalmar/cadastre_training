#!/usr/bin/env python3
"""
INFÉRENCE PRO : GeoTIFF + Fusion Gaussienne + Batch Processing + Squelettisation
Version optimisée pour les grands plans cadastraux.
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import gc
import cv2
import rasterio
from rasterio.windows import Window

class ProTiledInference:
    def __init__(self, checkpoint_path, tile_size=1024, overlap=128, batch_size=8):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Chargement du modèle
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', {})
        base_filters = config.get('base_filters', 64)

        from inference import UNet
        self.model = UNet(n_channels=3, n_classes=1, base_filters=base_filters).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 2. Fenêtre de fusion Gaussienne (évite les traces de coutures)
        self.window = self._create_gaussian_window(tile_size, sigma=0.25)
        print(f"✅ Modèle prêt sur {self.device} | Tuiles: {tile_size}px | Batch: {batch_size}")

    def _create_gaussian_window(self, size, sigma=0.25):
        center = size // 2
        x = np.arange(size) - center
        y = np.arange(size) - center
        X, Y = np.meshgrid(x, y)
        gaussian = np.exp(-(X**2 + Y**2) / (2 * (sigma * size)**2))
        return (gaussian / gaussian.max()).astype(np.float32)

    def skeletonize(self, binary_mask):
        """Réduit les lignes à 1 pixel d'épaisseur"""
        return cv2.ximgproc.thinning(binary_mask) if hasattr(cv2, 'ximgproc') else binary_mask

    def run(self, input_path, output_path, threshold=0.4, use_skeleton=False):
        # Désactiver la limite de pixels pour les très gros plans
        Image.MAX_IMAGE_PIXELS = None
        
        with rasterio.open(input_path) as src:
            profile = src.profile.copy()
            w_full, h_full = src.width, src.height
            
            # Préparation des accumulateurs
            full_mask = np.zeros((h_full, w_full), dtype=np.float32)
            count_mask = np.zeros((h_full, w_full), dtype=np.float32)

            # Calcul des coordonnées des tuiles
            stride = self.tile_size - self.overlap
            coords = []
            for y in range(0, h_full, stride):
                for x in range(0, w_full, stride):
                    # Ajustement pour ne pas dépasser les bords
                    cur_x = min(x, w_full - self.tile_size)
                    cur_y = min(y, h_full - self.tile_size)
                    if (cur_x, cur_y) not in coords:
                        coords.append((cur_x, cur_y))

            print(f"🚀 Traitement de {len(coords)} tuiles...")

            with torch.no_grad():
                for i in tqdm(range(0, len(coords), self.batch_size)):
                    batch_coords = coords[i : i + self.batch_size]
                    batch_imgs = []
                    
                    for bx, by in batch_coords:
                        # Lecture directe du TIF par fenêtre
                        window_read = Window(bx, by, self.tile_size, self.tile_size)
                        # Lecture 3 bandes (RGB), même si le TIF en a 4
                        img_tile = src.read([1, 2, 3], window=window_read, boundless=True, fill_value=255)
                        img_tile = np.moveaxis(img_tile, 0, -1) # (C,H,W) -> (H,W,C)
                        batch_imgs.append(self.transform(Image.fromarray(img_tile.astype(np.uint8))))
                    
                    # Inférence Batch
                    inputs = torch.stack(batch_imgs).to(self.device)
                    # Utilisation de autocast pour la vitesse (Blackwell)
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        outputs = torch.sigmoid(self.model(inputs))
                    
                    preds = outputs.cpu().float().numpy()
                    
                    # Accumulation avec fenêtre gaussienne
                    for idx, (bx, by) in enumerate(batch_coords):
                        full_mask[by:by+self.tile_size, bx:bx+self.tile_size] += preds[idx, 0] * self.window
                        count_mask[by:by+self.tile_size, bx:bx+self.tile_size] += self.window

            # 3. Finalisation du masque
            print("📊 Normalisation et binarisation...")
            final_mask = np.divide(full_mask, count_mask, out=np.zeros_like(full_mask), where=count_mask!=0)
            
            # Binarisation automatique par Otsu si le seuil manuel échoue
            mask_8bit = (final_mask * 255).astype(np.uint8)
            _, binary_mask = cv2.threshold(mask_8bit, int(threshold * 255), 255, cv2.THRESH_BINARY)
            
            if use_skeleton:
                print("🦴 Squelettisation active...")
                binary_mask = self.skeletonize(binary_mask)

            # 4. Sauvegarde GeoTIFF
            profile.update(
                dtype=rasterio.uint8,
                count=1,
                compress='lzw',
                nodata=0,
                tiled=True,
                blockxsize=256,
                blockysize=256
            )

            print(f"💾 Sauvegarde finale : {output_path}")
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(binary_mask, 1)

            # Nettoyage mémoire
            del full_mask, count_mask, binary_mask
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--skeleton', action='store_true', help="Produit des lignes de 1 pixel")
    args = parser.parse_args()

    processor = ProTiledInference(args.checkpoint, batch_size=args.batch_size)
    processor.run(args.input, args.output, threshold=args.threshold, use_skeleton=args.skeleton)