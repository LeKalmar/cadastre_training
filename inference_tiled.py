#!/usr/bin/env python3
"""
Inférence par tuiles pour grandes images - Version corrigée sans lignes dédoublées
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

    def process_tile(self, tile):
        """Traite une seule tuile"""
        with torch.no_grad():
            # Mixed precision
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = self.model(tile)
                output = output.float()  # Convertir en float32
        return output.squeeze().cpu().numpy()

    def create_gaussian_window(self, size, sigma=0.3):
        """Crée une fenêtre gaussienne pour fusion lisse"""
        center = size // 2
        x = np.arange(size) - center
        y = np.arange(size) - center
        X, Y = np.meshgrid(x, y)
        
        # Gaussienne 2D
        gaussian = np.exp(-(X**2 + Y**2) / (2 * (sigma * size)**2))
        
        # Normaliser entre 0 et 1
        gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())
        
        return gaussian.astype(np.float32)

    def clean_double_lines(self, mask, threshold=0.4):
        """Nettoie les lignes dédoublées"""
        # Convertir en uint8
        mask_8bit = (mask * 255).astype(np.uint8)
        
        # 1. Seuillage global
        _, binary = cv2.threshold(mask_8bit, int(threshold * 255), 255, cv2.THRESH_BINARY)
        
        # 2. Squelettisation (si disponible)
        skeleton = binary.copy()
        try:
            if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
                skeleton = cv2.ximgproc.thinning(binary)
        except:
            # Méthode alternative: érosion pour amincir
            kernel = np.ones((3, 3), np.uint8)
            skeleton = cv2.erode(binary, kernel, iterations=1)
        
        # 3. Distance transform pour épaissir uniformément
        dist = cv2.distanceTransform(255 - skeleton, cv2.DIST_L2, 3)
        
        # 4. Seuillage de la distance pour épaissir
        line_width = 2  # Épaisseur désirée des lignes
        thickened = (dist <= line_width).astype(np.uint8) * 255
        
        return thickened.astype(np.float32) / 255.0

    def predict_large_image(self, image_path, threshold=0.4):
        """Prédiction par tuiles avec fusion CORRIGÉE"""
        # Ignorer l'avertissement DecompressionBomb
        Image.MAX_IMAGE_PIXELS = None

        # Charger l'image
        image = Image.open(image_path).convert('RGB')
        original_width, original_height = image.size
        image_np = np.array(image)

        print(f"📏 Image: {original_width}x{original_height} pixels")
        print(f"🔲 Tuiles: {self.tile_size}x{self.tile_size} avec overlap {self.overlap}")

        # Calculer les tuiles - OPTIMISÉ
        tile_size = self.tile_size
        overlap = min(self.overlap, tile_size // 4)  # Max 25% d'overlap
        stride = tile_size - overlap
        
        # Nombre de tuiles
        n_tiles_x = int(np.ceil((original_width - overlap) / stride))
        n_tiles_y = int(np.ceil((original_height - overlap) / stride))
        
        print(f"🧩 Nombre de tuiles: {n_tiles_x} x {n_tiles_y} = {n_tiles_x * n_tiles_y}")
        print(f"📐 Stride: {stride}, Overlap: {overlap}")

        # Masque de sortie
        full_mask = np.zeros((original_height, original_width), dtype=np.float32)
        count_mask = np.zeros((original_height, original_width), dtype=np.float32)

        # Fenêtre GAUSSIENNE (pas cosinus)
        window = self.create_gaussian_window(tile_size, sigma=0.25)

        pbar = tqdm(total=n_tiles_x * n_tiles_y, desc="Processing tiles")

        for i in range(n_tiles_y):
            for j in range(n_tiles_x):
                # Position de la tuile EXACTE
                x_start = j * stride
                y_start = i * stride
                x_end = min(x_start + tile_size, original_width)
                y_end = min(y_start + tile_size, original_height)
                
                # Ajuster pour la dernière tuile
                if x_end == original_width:
                    x_start = original_width - tile_size
                if y_end == original_height:
                    y_start = original_height - tile_size
                
                # Extraire la tuile
                tile_np = image_np[y_start:y_end, x_start:x_end]
                tile_h, tile_w = tile_np.shape[:2]
                
                # Redimensionner si nécessaire
                if tile_h != tile_size or tile_w != tile_size:
                    tile_pil = Image.fromarray(tile_np)
                    tile_pil = tile_pil.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
                    tile_np = np.array(tile_pil)

                # Convertir en tensor
                tile_tensor = self.transform(tile_np).unsqueeze(0).to(self.device)

                # Traiter la tuile
                tile_pred = self.process_tile(tile_tensor)
                
                # Redimensionner si nécessaire
                if tile_h != tile_size or tile_w != tile_size:
                    tile_pred_img = Image.fromarray((tile_pred * 255).astype(np.uint8))
                    tile_pred_img = tile_pred_img.resize((tile_w, tile_h), Image.Resampling.NEAREST)
                    tile_pred_resized = np.array(tile_pred_img).astype(np.float32) / 255.0
                    
                    window_resized = cv2.resize(window, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
                else:
                    tile_pred_resized = tile_pred
                    window_resized = window[:tile_h, :tile_w]

                # Fusion
                full_mask[y_start:y_end, x_start:x_end] += tile_pred_resized * window_resized
                count_mask[y_start:y_end, x_start:x_end] += window_resized

                # Nettoyer
                del tile_tensor, tile_pred
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

                pbar.update(1)

        pbar.close()

        # Moyenne pondérée
        count_mask[count_mask == 0] = 1.0
        full_mask = full_mask / count_mask
        
        # AFFICHER DES STATISTIQUES POUR DEBUG
        print(f"\n📊 Statistiques du masque avant seuillage:")
        print(f"   Min: {full_mask.min():.4f}")
        print(f"   Max: {full_mask.max():.4f}")
        print(f"   Moyenne: {full_mask.mean():.4f}")
        print(f"   Médiane: {np.median(full_mask):.4f}")
        print(f"   Seuil utilisé: {threshold}")

        # Créer un histogramme pour choisir le bon seuil
        plt.figure(figsize=(10, 6))
        plt.hist(full_mask.flatten(), bins=100, alpha=0.7)
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'Seuil (threshold={threshold})')
        plt.xlabel('Valeur du masque')
        plt.ylabel('Fréquence')
        plt.title('Distribution des valeurs du masque')
        plt.legend()
        plt.tight_layout()

        # Sauvegarder l'histogramme
        output_dir = Path("debug_output")
        output_dir.mkdir(exist_ok=True)
        hist_path = output_dir / f"{Path(image_path).stem}_histogram.png"
        plt.savefig(hist_path, dpi=150)
        plt.close()
        print(f"📈 Histogramme sauvegardé: {hist_path}")

        # Normaliser si nécessaire
        if full_mask.max() > 0.9 and full_mask.min() < 0.1:
            print("⚠️  Normalisation automatique appliquée")
            mask_min, mask_max = full_mask.min(), full_mask.max()
            if mask_max - mask_min > 0.01:
                full_mask = (full_mask - mask_min) / (mask_max - mask_min)

        # Post-traitement CRITIQUE pour éviter les doublons
        full_mask = self.clean_double_lines(full_mask, threshold)
        
        # Binariser
        mask_8bit = (full_mask * 255).astype(np.uint8)
        _, mask_binary = cv2.threshold(mask_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return mask_binary, image_np

    def post_process_mask(self, mask):
        """Post-traitement simple - version safe"""
        # Conversion pour OpenCV
        mask_8bit = (mask * 255).astype(np.uint8)
        
        # Seuillage global simple
        _, binary = cv2.threshold(mask_8bit, 127, 255, cv2.THRESH_BINARY)
        
        # Fermeture légère pour combler les petits trous
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Filtre médian pour réduire le bruit
        denoised = cv2.medianBlur(closed, 3)
        
        return denoised.astype(np.float32) / 255.0

    def save_results(self, mask_binary, original_image, output_path, image_name, threshold):
        """Sauvegarde les résultats"""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarder le masque
        mask_img = Image.fromarray(mask_binary, mode='L')
        mask_path = output_dir / f"{Path(image_name).stem}_mask.png"
        mask_img.save(mask_path)
        print(f"✅ Masque sauvegardé: {mask_path}")

        # Sauvegarder aussi le masque en tant que probabilités (pour debug)
        prob_mask = (mask_binary > 0).astype(np.float32)
        prob_img = Image.fromarray((prob_mask * 255).astype(np.uint8))
        prob_path = output_dir / f"{Path(image_name).stem}_prob.png"
        prob_img.save(prob_path)

        # Créer visualisation
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))

        axes[0].imshow(original_image)
        axes[0].set_title('Image originale')
        axes[0].axis('off')

        axes[1].imshow(mask_binary, cmap='gray')
        axes[1].set_title(f'Masque prédit (seuil={threshold})')
        axes[1].axis('off')

        # Overlay
        overlay = original_image.copy()
        mask_indices = mask_binary > 127
        overlay[mask_indices] = [255, 0, 0]  # Rouge pour les lignes

        axes[2].imshow(overlay)
        axes[2].set_title('Overlay (rouge = lignes)')
        axes[2].axis('off')

        plt.tight_layout()
        viz_path = output_dir / f"{Path(image_name).stem}_viz.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Visualisation sauvegardée: {viz_path}")


def main():
    parser = argparse.ArgumentParser(description='Inférence par tuiles pour grandes images')
    parser.add_argument('--checkpoint', required=True, help='Chemin checkpoint')
    parser.add_argument('--input', required=True, help='Image à traiter')
    parser.add_argument('--output', default='tiled_predictions', help='Dossier de sortie')
    parser.add_argument('--tile-size', type=int, default=1024, help='Taille des tuiles')
    parser.add_argument('--overlap', type=int, default=128, help='Chevauchement des tuiles')  # Réduit à 128
    parser.add_argument('--threshold', type=float, default=0.4, help='Seuil de binarisation')  # Augmenté à 0.4
    parser.add_argument('--no-postprocess', action='store_true', help='Désactiver le post-traitement')

    args = parser.parse_args()

    # Initialiser le processeur
    processor = TiledInference(
        checkpoint_path=args.checkpoint,
        tile_size=args.tile_size,
        overlap=args.overlap
    )

    # Désactiver post-traitement si demandé
    if args.no_postprocess:
        processor.post_process_mask = lambda x: x

    # Traiter l'image
    mask_binary, original_image = processor.predict_large_image(
        args.input,
        threshold=args.threshold
    )

    # Sauvegarder
    processor.save_results(
        mask_binary,
        original_image,
        args.output,
        args.input,
        args.threshold
    )

    print(f"\n✅ Traitement terminé!")


if __name__ == '__main__':
    main()