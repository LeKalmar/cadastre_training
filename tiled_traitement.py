#!/usr/bin/env python3
"""
Inférence par tuiles pour grandes images - Version avec Squelettisation intégrée
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
import cv2

class TiledInference:
    def __init__(self, checkpoint_path, tile_size=1024, overlap=128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Charger modèle
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', {})
        base_filters = config.get('base_filters', 64)

        from inference import UNet
        self.model = UNet(n_channels=3, n_classes=1, base_filters=base_filters).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.tile_size = tile_size
        self.overlap = overlap
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print(f"✅ Modèle chargé (tile_size={tile_size}, overlap={overlap})")

    def skeletonize(self, binary_mask):
        """Réduit les lignes à une épaisseur de 1 pixel"""
        print("🦴 Squelettisation en cours...")
        
        # Vérifier si l'extension ximgproc est disponible (plus rapide/précis)
        try:
            skeleton = cv2.ximgproc.thinning(binary_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            return skeleton
        except (AttributeError, cv2.error):
            # Méthode alternative native (érosion itérative)
            size = np.size(binary_mask)
            skel = np.zeros(binary_mask.shape, np.uint8)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
            done = False
            temp_mask = binary_mask.copy()

            while not done:
                eroded = cv2.erode(temp_mask, element)
                temp = cv2.dilate(eroded, element)
                temp = cv2.subtract(temp_mask, temp)
                skel = cv2.bitwise_or(skel, temp)
                temp_mask = eroded.copy()
                if cv2.countNonZero(temp_mask) == 0:
                    done = True
            return skel

    def process_tile(self, tile):
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = self.model(tile)
                output = output.float()
        return output.squeeze().cpu().numpy()

    def create_gaussian_window(self, size, sigma=0.25):
        center = size // 2
        x = np.arange(size) - center
        y = np.arange(size) - center
        X, Y = np.meshgrid(x, y)
        gaussian = np.exp(-(X**2 + Y**2) / (2 * (sigma * size)**2))
        return (gaussian / gaussian.max()).astype(np.float32)

    def predict_large_image(self, image_path, threshold=0.4, apply_skeleton=False):
        Image.MAX_IMAGE_PIXELS = None
        image = Image.open(image_path).convert('RGB')
        original_width, original_height = image.size
        image_np = np.array(image)

        stride = self.tile_size - self.overlap
        n_tiles_x = int(np.ceil((original_width - self.overlap) / stride))
        n_tiles_y = int(np.ceil((original_height - self.overlap) / stride))

        full_mask = np.zeros((original_height, original_width), dtype=np.float32)
        count_mask = np.zeros((original_height, original_width), dtype=np.float32)
        window = self.create_gaussian_window(self.tile_size)

        pbar = tqdm(total=n_tiles_x * n_tiles_y, desc="Processing tiles")

        for i in range(n_tiles_y):
            for j in range(n_tiles_x):
                x_start = j * stride
                y_start = i * stride
                x_end = min(x_start + self.tile_size, original_width)
                y_end = min(y_start + self.tile_size, original_height)
                
                if x_end == original_width: x_start = original_width - self.tile_size
                if y_end == original_height: y_start = original_height - self.tile_size
                
                tile_np = image_np[y_start:y_end, x_start:x_end]
                tile_tensor = self.transform(tile_np).unsqueeze(0).to(self.device)
                tile_pred = self.process_tile(tile_tensor)
                
                full_mask[y_start:y_end, x_start:x_end] += tile_pred * window
                count_mask[y_start:y_end, x_start:x_end] += window
                pbar.update(1)

        pbar.close()
        full_mask /= (count_mask + 1e-8)
        
        # Binarisation
        mask_8bit = (full_mask * 255).astype(np.uint8)
        _, mask_binary = cv2.threshold(mask_8bit, int(threshold * 255), 255, cv2.THRESH_BINARY)
        
        # Application de la squelettisation si demandée
        if apply_skeleton:
            mask_binary = self.skeletonize(mask_binary)
            
        return mask_binary, image_np

    def save_results(self, mask_binary, original_image, output_path, image_name):
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mask_path = output_dir / f"{Path(image_name).stem}_mask.png"
        Image.fromarray(mask_binary).save(mask_path)
        print(f"✅ Résultat sauvegardé: {mask_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='tiled_predictions')
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--skeleton', action='store_true', help='Activer la squelettisation')

    args = parser.parse_args()
    processor = TiledInference(args.checkpoint)
    
    mask_binary, original_image = processor.predict_large_image(
        args.input, threshold=args.threshold, apply_skeleton=args.skeleton
    )
    processor.save_results(mask_binary, original_image, args.output, args.input)

if __name__ == '__main__':
    main()