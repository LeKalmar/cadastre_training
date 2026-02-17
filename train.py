#!/usr/bin/env python3
"""
Script d'entraînement pour segmentation de cadastres napoléoniens
Optimisé pour PyTorch + RTX 30xx/40xx/50xx
Correction: Synchronisation des augmentations image/masque
"""

import os
import json
import argparse
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# Librairie pour les augmentations synchronisées (Image + Masque)
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ============================================================================
# Dataset
# ============================================================================

class CadastreDataset(Dataset):
    """Dataset pour images cadastrales + masques de segmentation"""

    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        # Extensions d'images supportées
        valid_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        self.images = sorted([f for f in self.img_dir.iterdir() if f.suffix.lower() in valid_extensions])

        print(f"📁 Dataset chargé: {len(self.images)} images depuis {img_dir}")
        if len(self.images) == 0:
            print("⚠️ ATTENTION: Aucune image trouvée ! Vérifiez le chemin.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        # Gestion intelligente de l'extension du masque (.jpg -> .png)
        # On suppose que le masque a le même nom de base (stem)
        mask_path = self.mask_dir / (img_path.stem + ".png")
        
        # Fallback si le masque n'est pas en png, on essaie la même extension que l'image
        if not mask_path.exists():
            mask_path = self.mask_dir / img_path.name
            
        if not mask_path.exists():
            raise FileNotFoundError(f"Masque introuvable pour {img_path}")

        # Charger image et masque avec OpenCV (plus rapide et natif pour Albumentations)
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Binarisation stricte du masque (0 ou 255)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Normaliser le masque à 0 ou 1 (float)
        mask = mask.astype(np.float32) / 255.0

        # Appliquer les transformations conjointes (Image + Masque)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Si pas de transform, on convertit quand même en Tensor
        if not isinstance(image, torch.Tensor):
            transform_basic = ToTensorV2()
            augmented = transform_basic(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Le masque doit avoir une dimension de canal (1, H, W) pour PyTorch
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask


# ============================================================================
# Modèle U-Net (Inchangé car correct)
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
        return self.out(dec4)


# ============================================================================
# Loss & Metrics
# ============================================================================

class CombinedLoss(nn.Module):
    """BCE + Dice Loss combinées avec pondération pour classes déséquilibrées"""
    def __init__(self, bce_weight=0.5, pos_weight=10.0):
        super().__init__()
        # pos_weight donne plus d'importance aux pixels blancs (lignes)
        # car ils sont rares par rapport au fond noir.
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight])) 
        self.bce_weight = bce_weight
        self.smooth = 1.0

    def dice_loss(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice

    def forward(self, pred, target):
        # Pour BCE, on s'assure que pos_weight est sur le bon device
        if self.bce.pos_weight.device != pred.device:
            self.bce.pos_weight = self.bce.pos_weight.to(pred.device)
            
        bce = self.bce(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice

def calculate_iou(pred, target, threshold=0.5):
    pred_mask = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_mask * target).sum()
    union = pred_mask.sum() + target.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()

# ============================================================================
# Utils visualization
# ============================================================================

def save_debug_prediction_fixed(model, val_ds, device, epoch, work_dir):
    model.eval()
    # Liste des noms de fichiers cibles (sans extension pour être flexible)
    target_names = [
        "achicourt_b_0_y3136_x6720",
        "lille_b_2_73",
        "arras_k_7_y5376_x1344"
    ]
    
    found_indices = []
    # On parcourt le dataset pour trouver les indices correspondant à ces noms
    for i, img_path in enumerate(val_ds.images):
        if img_path.stem in target_names:
            found_indices.append(i)
    
    if not found_indices:
        return

    # Création d'une figure avec 3 lignes (une par image) et 3 colonnes (Original, Ground Truth, Prediction)
    fig, axes = plt.subplots(len(found_indices), 3, figsize=(15, 5 * len(found_indices)))
    
    # Si on n'en trouve qu'une seule, matplotlib ne renvoie pas une matrice 2D, on corrige :
    if len(found_indices) == 1: axes = np.expand_dims(axes, axis=0)

    with torch.no_grad():
        for idx, ds_idx in enumerate(found_indices):
            image, mask = val_ds[ds_idx]
            input_tensor = image.unsqueeze(0).to(device)
            output = torch.sigmoid(model(input_tensor)).cpu().numpy()[0, 0]
            
            # Image originale (denormalisation pour l'affichage)
            img_np = image.permute(1, 2, 0).numpy()
            img_np = (img_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            img_np = np.clip(img_np, 0, 1)

            # Affichage
            axes[idx, 0].imshow(img_np)
            axes[idx, 0].set_title(f"Original: {val_ds.images[ds_idx].stem}")
            axes[idx, 1].imshow(mask.squeeze(), cmap='gray')
            axes[idx, 1].set_title("Ground Truth (Label)")
            axes[idx, 2].imshow(output > 0.4, cmap='gray') # Seuil à 0.4
            axes[idx, 2].set_title(f"Prediction (Epoch {epoch})")
            
            for ax in axes[idx]: ax.axis('off')

    plt.tight_layout()
    plt.savefig(work_dir / f"debug_epoch_{epoch}.png")
    plt.close()

# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, masks)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        iou = calculate_iou(outputs, masks)
        running_loss += loss.item()
        running_iou += iou
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'iou': f'{iou:.3f}'})
        
    return running_loss / len(loader), running_iou / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Val", leave=False):
            images, masks = images.to(device), masks.to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            iou = calculate_iou(outputs, masks)
            running_loss += loss.item()
            running_iou += iou
            
    return running_loss / len(loader), running_iou / len(loader)

# ============================================================================
# Main
# ============================================================================
def save_debug_prediction_fixed(model, val_ds, device, epoch, work_dir):
    model.eval()
    # Liste des noms de fichiers cibles (sans extension pour être flexible)
    target_names = [
        "achicourt_b_0_y3136_x6720",
        "lille_b_2_73",
        "arras_k_7_y5376_x1344"
    ]
    
    found_indices = []
    # On parcourt le dataset pour trouver les indices correspondant à ces noms
    for i, img_path in enumerate(val_ds.images):
        if img_path.stem in target_names:
            found_indices.append(i)
    
    if not found_indices:
        return

    # Création d'une figure avec 3 lignes (une par image) et 3 colonnes (Original, Ground Truth, Prediction)
    fig, axes = plt.subplots(len(found_indices), 3, figsize=(15, 5 * len(found_indices)))
    
    # Si on n'en trouve qu'une seule, matplotlib ne renvoie pas une matrice 2D, on corrige :
    if len(found_indices) == 1: axes = np.expand_dims(axes, axis=0)

    with torch.no_grad():
        for idx, ds_idx in enumerate(found_indices):
            image, mask = val_ds[ds_idx]
            input_tensor = image.unsqueeze(0).to(device)
            output = torch.sigmoid(model(input_tensor)).cpu().numpy()[0, 0]
            
            # Image originale (denormalisation pour l'affichage)
            img_np = image.permute(1, 2, 0).numpy()
            img_np = (img_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            img_np = np.clip(img_np, 0, 1)

            # Affichage
            axes[idx, 0].imshow(img_np)
            axes[idx, 0].set_title(f"Original: {val_ds.images[ds_idx].stem}")
            axes[idx, 1].imshow(mask.squeeze(), cmap='gray')
            axes[idx, 1].set_title("Ground Truth (Label)")
            axes[idx, 2].imshow(output > 0.4, cmap='gray') # Seuil à 0.4
            axes[idx, 2].set_title(f"Prediction (Epoch {epoch})")
            
            for ax in axes[idx]: ax.axis('off')

    plt.tight_layout()
    plt.savefig(work_dir / f"debug_epoch_{epoch}.png")
    plt.close()
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/cadastre_lines', help='Chemin données')
    parser.add_argument('--work-dir', default='work_dirs/unet_corrected', help='Output dir')
    parser.add_argument('--resume', type=str, default=None, help='Chemin checkpoint pour reprendre')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    # Poids pour compenser le fait qu'il y a peu de pixels blancs (lignes)
    parser.add_argument('--pos-weight', type=float, default=5.0) 
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device} (Blackwell Optimized)")

    # --- DEFINITION DES TRANSFORMS (Albumentations) ---
    # Ces transformations s'appliquent identiquement à l'image et au masque
    train_transform = A.Compose([
        A.Resize(512, 512), # Assure une taille constante si les tuiles varient
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # --- DATASETS & LOADERS ---
    data_root = Path(args.data_root)
    train_ds = CadastreDataset(data_root/'images'/'train', data_root/'labels'/'train', train_transform)
    val_ds = CadastreDataset(data_root/'images'/'val', data_root/'labels'/'val', val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # --- MODEL & OPTIMIZATION ---
    model = UNet(n_channels=3, n_classes=1).to(device)

    criterion = CombinedLoss(pos_weight=args.pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    scaler = torch.amp.GradScaler('cuda')

    # --- TRAINING LOOP ---
    best_iou = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': []}

    print(f"\n🎯 Début entraînement sur {len(train_ds)} images.")

    # --- RESUME LOGIC ---
    start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"🔄 Chargement du checkpoint : {args.resume}")
            checkpoint = torch.load(args.resume)
            
            # Cas 1 : C'est un checkpoint complet (ce qu'on va mettre en place)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # AJOUTE CETTE LIGNE ICI POUR FORCER LE LR :
                for param_group in optimizer.param_groups: param_group['lr'] = args.lr
                
                start_epoch = checkpoint['epoch'] + 1
            
            # Cas 2 : C'est juste le fichier de poids (votre cas actuel avec best_model.pth)
            else:
                model.load_state_dict(checkpoint)
                print("⚠️ Attention : Chargement des poids seuls (pas d'historique optimiseur).")
                print("   L'entraînement reprendra à l'epoch 1 mais avec un modèle déjà intelligent.")
        else:
            print(f"❌ Erreur : Fichier non trouvé {args.resume}")   

    # --- PRÉPARATION DU LOGGING ---
    from datetime import datetime
    import gc
    log_file = work_dir / "train_log.csv"
    if not log_file.exists():
        with open(log_file, "w") as f:
            f.write("epoch,time,lr,train_loss,train_iou,val_loss,val_iou\n")

    print(f"🎯 Début entraînement sur {len(train_ds)} images.")

    # --- DÉBUT DE LA BOUCLE CORRIGÉE ---
    for epoch in range(start_epoch, args.epochs + 1):
        now = datetime.now().strftime("%H:%M:%S")
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n🔔 Epoch {epoch}/{args.epochs} | 🕒 Début : {now} | 🚀 LR : {current_lr:.2e}")

        # 1. TRAIN : On ajoute 'scaler' qui manquait
        t_loss, t_iou = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        
        # 2. VALIDATION : On utilise le bon nom 'validate'
        v_loss, v_iou = validate(model, val_loader, criterion, device)

        # Mise à jour du scheduler
        scheduler.step(v_iou)

        # Affichage
        print(f"📊 Epoch {epoch} Terminée | T_Loss: {t_loss:.4f} T_IoU: {t_iou:.4f} | V_Loss: {v_loss:.4f} V_IoU: {v_iou:.4f}")

        # Sauvegarde CSV
        with open(log_file, "a") as f:
            f.write(f"{epoch},{now},{current_lr:.2e},{t_loss:.4f},{t_iou:.4f},{v_loss:.4f},{v_iou:.4f}\n")

        # Sauvegarde du meilleur modèle
        if v_iou > best_iou:
            best_iou = v_iou
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'config': {'base_filters': 64} # Valeur par défaut
            }
            torch.save(checkpoint, work_dir / 'best_model.pth')
            print(f"⭐ Nouveau record IoU: {best_iou:.4f} (Sauvegardé)")

        # Sauvegarde de sécurité
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_iou': best_iou,
        }, work_dir / 'latest.pth')

        gc.collect()
        torch.cuda.empty_cache()

    # Plot final curves
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_iou'], label='Val IoU')
    plt.legend()
    plt.savefig(work_dir / 'metrics.png')
    print("✅ Entraînement terminé.")

if __name__ == '__main__':
    main()