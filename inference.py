#!/usr/bin/env python3
"""
Script d'inférence pour segmentation cadastres napoléoniennes
Prédit les masques de lignes cadastrales sur de nouvelles images
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


# ============================================================================
# Modèle U-Net (même architecture que train.py)
# ============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, base_filters=64):
        super().__init__()
        
        self.enc1 = DoubleConv(n_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv(base_filters * 4, base_filters * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(base_filters * 8, base_filters * 16)
        
        self.up1 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_filters * 16, base_filters * 8)
        
        self.up2 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_filters * 8, base_filters * 4)
        
        self.up3 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_filters * 4, base_filters * 2)
        
        self.up4 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_filters * 2, base_filters)
        
        self.out = nn.Conv2d(base_filters, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec1 = self.up1(bottleneck)
        dec1 = torch.cat([dec1, enc4], dim=1)
        dec1 = self.dec1(dec1)
        
        dec2 = self.up2(dec1)
        dec2 = torch.cat([dec2, enc3], dim=1)
        dec2 = self.dec2(dec2)
        
        dec3 = self.up3(dec2)
        dec3 = torch.cat([dec3, enc2], dim=1)
        dec3 = self.dec3(dec3)
        
        dec4 = self.up4(dec3)
        dec4 = torch.cat([dec4, enc1], dim=1)
        dec4 = self.dec4(dec4)
        
        out = self.out(dec4)
        return self.sigmoid(out)


# ============================================================================
# Inference
# ============================================================================

def predict_image(model, image_path, device, transform, threshold=0.5):
    """Prédit le masque pour une image"""
    # Charger image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)

    # Transform
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Prédiction
    model.eval()
    with torch.no_grad():
        # Option 1: Avec autocast (mixed precision)
        if device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(image_tensor)
                output = output.float()  # Convertir en float32
        else:
            # CPU
            output = model(image_tensor)

    # Post-processing
    mask = output.squeeze().cpu().numpy()
    mask_binary = (mask > threshold).astype(np.uint8) * 255

    # Convertir en PIL Image avec taille originale
    mask_img = Image.fromarray(mask_binary, mode='L')
    mask_img = mask_img.resize(original_size, Image.NEAREST)

    return mask_img, mask


def batch_inference(model, input_dir, output_dir, device, transform, threshold=0.5):
    """Inférence batch sur un dossier d'images"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Lister toutes les images
    image_files = sorted(list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg')))
    
    print(f"🔍 Trouvé {len(image_files)} images dans {input_dir}")
    
    for img_path in tqdm(image_files, desc="Prédiction"):
        # Prédire
        mask_img, _ = predict_image(model, img_path, device, transform, threshold)
        
        # Sauvegarder
        output_path = output_dir / img_path.name
        mask_img.save(output_path)
    
    print(f"✅ Masques sauvegardés dans: {output_dir}")


def visualize_prediction(model, image_path, device, transform, output_path=None, threshold=0.5):
    """Visualiser prédiction (image + masque + overlay)"""
    # Charger image originale
    image = Image.open(image_path).convert('RGB')
    
    # Prédire
    mask_img, mask = predict_image(model, image_path, device, transform, threshold)
    
    # Créer figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Image originale
    axes[0].imshow(image)
    axes[0].set_title('Image originale')
    axes[0].axis('off')
    
    # Masque prédit
    axes[1].imshow(mask_img, cmap='gray')
    axes[1].set_title(f'Masque prédit (seuil={threshold})')
    axes[1].axis('off')
    
    # Overlay
    overlay = np.array(image).copy()
    mask_np = np.array(mask_img)
    overlay[mask_np > 0] = [255, 0, 0]  # Rouge pour les lignes
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 Visualisation sauvegardée: {output_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Inférence segmentation cadastre')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Chemin vers le checkpoint (.pth)')
    parser.add_argument('--input', type=str, required=True,
                        help='Image ou dossier d\'images')
    parser.add_argument('--output', type=str, default='predictions',
                        help='Dossier de sortie pour les masques')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Seuil de binarisation (0-1)')
    parser.add_argument('--visualize', action='store_true',
                        help='Créer visualisations (image + masque + overlay)')
    parser.add_argument('--base-filters', type=int, default=64,
                        help='Nombre de filtres de base U-Net (doit correspondre au modèle)')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Charger modèle
    print(f"📂 Chargement modèle: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Récupérer base_filters si présent dans config
    if 'config' in checkpoint and 'base_filters' in checkpoint['config']:
        base_filters = checkpoint['config']['base_filters']
        print(f"   Base filters détecté: {base_filters}")
    else:
        base_filters = args.base_filters
        print(f"   Utilisation base filters par défaut: {base_filters}")
    
    model = UNet(n_channels=3, n_classes=1, base_filters=base_filters).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Modèle chargé (epoch {checkpoint['epoch']}, IoU={checkpoint.get('val_iou', 0):.4f})")
    
    # Transform (même que l'entraînement)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    input_path = Path(args.input)
    
    # Cas 1: Input est une image unique
    if input_path.is_file():
        print(f"\n📸 Prédiction sur image unique: {input_path}")
        
        if args.visualize:
            # Créer visualisation
            output_vis = Path(args.output) / f"{input_path.stem}_viz.png"
            output_vis.parent.mkdir(parents=True, exist_ok=True)
            visualize_prediction(model, input_path, device, transform, output_vis, args.threshold)
        
        # Sauvegarder masque
        output_mask = Path(args.output) / f"{input_path.stem}_mask.png"
        output_mask.parent.mkdir(parents=True, exist_ok=True)
        mask_img, _ = predict_image(model, input_path, device, transform, args.threshold)
        mask_img.save(output_mask)
        print(f"✅ Masque sauvegardé: {output_mask}")
    
    # Cas 2: Input est un dossier
    elif input_path.is_dir():
        print(f"\n📁 Prédiction batch sur dossier: {input_path}")
        output_dir = Path(args.output)
        batch_inference(model, input_path, output_dir, device, transform, args.threshold)
        
        # Créer quelques visualisations si demandé
        if args.visualize:
            print(f"\n🎨 Création de visualisations...")
            vis_dir = output_dir / 'visualizations'
            vis_dir.mkdir(exist_ok=True)
            
            image_files = sorted(list(input_path.glob('*.png')) + list(input_path.glob('*.jpg')))
            # Visualiser les 5 premières images
            for img_path in image_files[:5]:
                output_vis = vis_dir / f"{img_path.stem}_viz.png"
                visualize_prediction(model, img_path, device, transform, output_vis, args.threshold)
            
            print(f"✅ Visualisations dans: {vis_dir}")
    
    else:
        print(f"❌ Erreur: {input_path} n'existe pas")
        return
    
    print(f"\n✅ Inférence terminée!")


if __name__ == '__main__':
    main()
