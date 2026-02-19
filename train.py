#!/usr/bin/env python3
"""
Script d'entraînement pour segmentation de cadastres napoléoniens
v2.0 - Corrections NaN + Encodeur pré-entraîné + Stabilité numérique

Dépendances à installer :
    pip install segmentation-models-pytorch albumentations opencv-python-headless tqdm matplotlib

Changements vs v1.0 :
    - Encodeur pré-entraîné (ResNet34) via segmentation-models-pytorch
    - Suppression de GradScaler (incompatible avec bfloat16)
    - Protection NaN dans la boucle d'entraînement
    - Augmentations agressives pour gérer la diversité des feuilles
    - Cosine Annealing scheduler + warmup
    - Early stopping
    - Meilleure gestion du pos_weight dynamique
    - Logging complet et checkpoint amélioré
"""

import os
import json
import argparse
import math
import gc
from pathlib import Path
from datetime import datetime
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

# segmentation-models-pytorch pour l'encodeur pré-entraîné
# pip install segmentation-models-pytorch
import segmentation_models_pytorch as smp


# ============================================================================
# Dataset
# ============================================================================

class CadastreDataset(Dataset):
    """Dataset pour images cadastrales + masques de segmentation"""

    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        valid_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        self.images = sorted([
            f for f in self.img_dir.iterdir()
            if f.suffix.lower() in valid_extensions
        ])

        print(f"📁 Dataset chargé: {len(self.images)} images depuis {img_dir}")
        if len(self.images) == 0:
            print("⚠️ ATTENTION: Aucune image trouvée !")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        # Recherche du masque (même stem, priorité .png)
        mask_path = self.mask_dir / (img_path.stem + ".png")
        if not mask_path.exists():
            mask_path = self.mask_dir / img_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Masque introuvable pour {img_path}")

        # Chargement
        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Impossible de charger l'image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Impossible de charger le masque: {mask_path}")

        # Binarisation stricte
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.float32) / 255.0

        # Transformations conjointes
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Conversion en tensor si pas déjà fait
        if not isinstance(image, torch.Tensor):
            basic = ToTensorV2()
            augmented = basic(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Le masque doit être (1, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask


# ============================================================================
# Modèle (U-Net vanilla comme fallback)
# ============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.insert(3, nn.Dropout2d(dropout))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class UNetVanilla(nn.Module):
    """U-Net basique avec dropout optionnel (si smp pas installé)"""

    def __init__(self, n_channels=3, n_classes=1, base_filters=64):
        super().__init__()
        bf = base_filters
        self.enc1 = DoubleConv(n_channels, bf)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(bf, bf * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(bf * 2, bf * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(bf * 4, bf * 8)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(bf * 8, bf * 16, dropout=0.3)
        self.up1 = nn.ConvTranspose2d(bf * 16, bf * 8, 2, stride=2)
        self.dec1 = DoubleConv(bf * 16, bf * 8, dropout=0.2)
        self.up2 = nn.ConvTranspose2d(bf * 8, bf * 4, 2, stride=2)
        self.dec2 = DoubleConv(bf * 8, bf * 4, dropout=0.1)
        self.up3 = nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2)
        self.dec3 = DoubleConv(bf * 4, bf * 2)
        self.up4 = nn.ConvTranspose2d(bf * 2, bf, 2, stride=2)
        self.dec4 = DoubleConv(bf * 2, bf)
        self.out = nn.Conv2d(bf, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        d1 = self.dec1(torch.cat([self.up1(b), e4], 1))
        d2 = self.dec2(torch.cat([self.up2(d1), e3], 1))
        d3 = self.dec3(torch.cat([self.up3(d2), e2], 1))
        d4 = self.dec4(torch.cat([self.up4(d3), e1], 1))
        return self.out(d4)


def build_model(encoder_name="resnet34", pretrained=True, n_classes=1):
    """
    Construit le modèle. Utilise smp si disponible, sinon UNet vanilla.

    Encodeurs recommandés par ordre de performance/taille :
    - resnet34 : bon compromis performance/vitesse (21M params)
    - efficientnet-b3 : meilleur si GPU le permet (12M params, plus lent)
    - resnet50 : plus lourd mais plus puissant (25M params)
    """
    try:
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=n_classes,
            decoder_attention_type="scse",  # Attention spatiale + canal
        )
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"🧠 Modèle: U-Net + {encoder_name} (pré-entraîné={pretrained})")
        print(f"   Paramètres: {param_count:.1f}M | Attention: scSE")
        return model
    except Exception as e:
        print(f"⚠️ smp non disponible ({e}), fallback sur UNet vanilla")
        return UNetVanilla(n_channels=3, n_classes=n_classes, base_filters=64)


# ============================================================================
# Loss & Metrics
# ============================================================================

class CombinedLoss(nn.Module):
    """
    BCE + Dice combinées avec protections numériques.

    Changements vs v1 :
    - Clamp des logits pour éviter les overflows dans exp()
    - Smooth factor augmenté dans Dice pour stabilité
    - pos_weight géré proprement sur le bon device dès l'init
    """

    def __init__(self, bce_weight=0.5, pos_weight=5.0, device='cuda'):
        super().__init__()
        pw = torch.tensor([pos_weight], device=device)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.bce_weight = bce_weight
        self.smooth = 1.0

    def dice_loss(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
                pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1.0 - dice

    def forward(self, pred, target):
        # Protection : clamper les logits pour éviter overflow dans BCEWithLogits
        pred_clamped = torch.clamp(pred, min=-20.0, max=20.0)

        bce = self.bce(pred_clamped, target)
        dice = self.dice_loss(pred_clamped, target)

        loss = self.bce_weight * bce + (1.0 - self.bce_weight) * dice
        return loss


def calculate_iou(pred, target, threshold=0.5):
    """IoU avec protection division par zéro"""
    with torch.no_grad():
        pred_mask = (torch.sigmoid(pred) > threshold).float()
        intersection = (pred_mask * target).sum()
        union = pred_mask.sum() + target.sum() - intersection
        if union == 0:
            return 1.0
        return (intersection / union).item()


def compute_pos_weight(dataset, num_samples=200):
    """
    Calcule le pos_weight optimal en analysant le ratio fond/lignes.
    Utile pour ne pas deviner cette valeur.
    """
    total_pos = 0
    total_neg = 0
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for idx in indices:
        _, mask = dataset[idx]
        pos = mask.sum().item()
        neg = mask.numel() - pos
        total_pos += pos
        total_neg += neg

    if total_pos == 0:
        print("⚠️ Aucun pixel positif trouvé ! Vérifiez vos masques.")
        return 10.0

    ratio = total_neg / total_pos
    # On ne veut pas un poids trop extrême
    pw = min(ratio, 30.0)
    print(f"📊 Ratio négatif/positif: {ratio:.1f} → pos_weight recommandé: {pw:.1f}")
    return pw


# ============================================================================
# Training & Validation
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device, use_amp=True):
    """
    Boucle d'entraînement d'une epoch.

    Changements vs v1 :
    - Suppression de GradScaler (incompatible bfloat16)
    - Protection NaN : skip le batch si loss est NaN
    - Gradient clipping toujours actif
    """
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    valid_batches = 0
    nan_count = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # Plus efficace que zero_grad()

        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, masks)
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)

        # *** PROTECTION NaN ***
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            if nan_count <= 3:
                print(f"  ⚠️ NaN/Inf loss détecté (batch ignoré, {nan_count} fois)")
            continue

        # Backward sans GradScaler (bfloat16 n'en a pas besoin)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        iou = calculate_iou(outputs.detach(), masks)
        running_loss += loss.item()
        running_iou += iou
        valid_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{iou:.3f}'
        })

    if valid_batches == 0:
        print("  ❌ TOUTE L'EPOCH A PRODUIT DES NaN ! Le modèle est probablement corrompu.")
        return float('nan'), 0.0

    if nan_count > 0:
        print(f"  ⚠️ {nan_count} batchs ignorés pour NaN sur {valid_batches + nan_count} total")

    return running_loss / valid_batches, running_iou / valid_batches


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp=True):
    """Validation avec protection NaN"""
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    valid_batches = 0

    for images, masks in tqdm(loader, desc="Val", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, masks)
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        iou = calculate_iou(outputs, masks)
        running_loss += loss.item()
        running_iou += iou
        valid_batches += 1

    if valid_batches == 0:
        return float('nan'), 0.0

    return running_loss / valid_batches, running_iou / valid_batches


# ============================================================================
# Visualization
# ============================================================================

def save_debug_prediction(model, val_ds, device, epoch, work_dir, target_names=None):
    """Sauvegarde des prédictions de debug pour visualiser la progression"""
    model.eval()

    if target_names is None:
        # Prend 3 images au hasard si pas de noms spécifiques
        n = min(3, len(val_ds))
        found_indices = random.sample(range(len(val_ds)), n)
    else:
        found_indices = []
        for i, img_path in enumerate(val_ds.images):
            if img_path.stem in target_names:
                found_indices.append(i)
        if not found_indices:
            # Fallback : premières images
            found_indices = list(range(min(3, len(val_ds))))

    n_images = len(found_indices)
    fig, axes = plt.subplots(n_images, 3, figsize=(15, 5 * n_images))
    if n_images == 1:
        axes = np.expand_dims(axes, axis=0)

    with torch.no_grad():
        for idx, ds_idx in enumerate(found_indices):
            image, mask = val_ds[ds_idx]
            input_tensor = image.unsqueeze(0).to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output = torch.sigmoid(model(input_tensor)).cpu().float().numpy()[0, 0]

            # Dénormalisation ImageNet
            img_np = image.permute(1, 2, 0).numpy()
            img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)

            axes[idx, 0].imshow(img_np)
            axes[idx, 0].set_title(f"Original: {val_ds.images[ds_idx].stem}", fontsize=8)
            axes[idx, 1].imshow(mask.squeeze(), cmap='gray')
            axes[idx, 1].set_title("Ground Truth")
            axes[idx, 2].imshow(output > 0.4, cmap='gray')
            axes[idx, 2].set_title(f"Prediction (Epoch {epoch})")

            for ax in axes[idx]:
                ax.axis('off')

    plt.tight_layout()
    plt.savefig(work_dir / f"debug_epoch_{epoch}.png", dpi=100)
    plt.close()


# ============================================================================
# Augmentations
# ============================================================================

def get_train_transform(img_size=512):
    """
    Augmentations agressives pour gérer la diversité des feuilles cadastrales.

    L'idée : forcer le réseau à ignorer les couleurs, textures du papier,
    et se concentrer uniquement sur la structure des lignes.
    """
    return A.Compose([
        A.Resize(img_size, img_size),

        # --- Géométrie ---
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomScale(scale_limit=0.15, p=0.3),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomCrop(img_size, img_size, p=1.0),

        # --- Couleur/Luminosité (ne s'applique qu'à l'image, pas au masque) ---
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
            A.CLAHE(clip_limit=4.0, p=1),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1, p=1),
        ], p=0.5),

        # Simuler différents types de papier (jauni, blanchi, etc.)
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
            A.ToGray(p=1),  # Simuler des feuilles N&B
            A.ChannelShuffle(p=1),  # Mélanger les canaux pour robustesse
        ], p=0.3),

        # Bruit et flou (simuler la qualité de numérisation variable)
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1),
            A.GaussianBlur(blur_limit=(3, 5), p=1),
        ], p=0.2),

        # --- Normalisation ImageNet ---
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform(img_size=512):
    """Transformations minimales pour la validation"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ============================================================================
# Cosine Annealing avec Warmup
# ============================================================================

class CosineAnnealingWarmup:
    """
    Scheduler avec warmup linéaire puis cosine annealing.
    Le warmup est crucial quand on reprend un entraînement sur un nouveau dataset.
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Warmup linéaire
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# ============================================================================
# Vérification du dataset
# ============================================================================

def verify_dataset(dataset, num_checks=20):
    """
    Vérifie que le dataset est cohérent avant de lancer l'entraînement.
    Détecte : masques tout noir, images corrompues, tailles incohérentes.
    """
    print(f"\n🔍 Vérification du dataset ({num_checks} échantillons)...")
    all_black = 0
    errors = 0
    sizes = set()

    indices = random.sample(range(len(dataset)), min(num_checks, len(dataset)))

    for idx in indices:
        try:
            img, mask = dataset[idx]
            sizes.add((img.shape[1], img.shape[2]))

            if mask.sum() == 0:
                all_black += 1
                if all_black <= 5:
                    print(f"  ⚠️ Masque tout noir: {dataset.images[idx].name}")
        except Exception as e:
            errors += 1
            print(f"  ❌ Erreur sur {dataset.images[idx].name}: {e}")

    if errors > 0:
        print(f"  ❌ {errors} images/masques corrompus détectés !")
    if all_black > num_checks * 0.5:
        print(f"  ⚠️ {all_black}/{num_checks} masques sont tout noirs. Vérifiez vos labels !")
    if len(sizes) > 1:
        print(f"  ⚠️ Tailles d'images hétérogènes détectées: {sizes}")
    else:
        print(f"  ✅ Dataset OK | Tailles: {sizes} | Masques noirs: {all_black}/{num_checks}")

    return errors == 0


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Entraînement segmentation cadastrale v2")
    parser.add_argument('--data-root', default='data/cadastre_lines')
    parser.add_argument('--work-dir', default='work_dirs/unet_v2')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint à reprendre')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (recommandé: 3e-4 avec encodeur pré-entraîné)')
    parser.add_argument('--pos-weight', type=float, default=0, help='0 = calcul automatique')
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--encoder', type=str, default='resnet34',
                        help='Encodeur: resnet34, resnet50, efficientnet-b3, etc.')
    parser.add_argument('--no-pretrained', action='store_true', help='Ne pas utiliser ImageNet')
    parser.add_argument('--no-amp', action='store_true', help='Désactiver mixed precision')
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--debug-names', nargs='*', default=None,
                        help='Noms des images de debug (stems)')
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = not args.no_amp and device.type == 'cuda'
    print(f"🚀 Device: {device} | AMP (bfloat16): {use_amp}")

    # --- TRANSFORMS ---
    train_transform = get_train_transform(args.img_size)
    val_transform = get_val_transform(args.img_size)

    # --- DATASETS ---
    data_root = Path(args.data_root)
    train_ds = CadastreDataset(data_root / 'images' / 'train', data_root / 'labels' / 'train', train_transform)
    val_ds = CadastreDataset(data_root / 'images' / 'val', data_root / 'labels' / 'val', val_transform)

    # Vérification du dataset
    verify_dataset(train_ds)
    verify_dataset(val_ds)

    # --- DATALOADERS ---
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
        persistent_workers=True
    )

    # --- MODÈLE ---
    model = build_model(
        encoder_name=args.encoder,
        pretrained=not args.no_pretrained,
        n_classes=1
    ).to(device)

    # --- CALCUL pos_weight ---
    if args.pos_weight <= 0:
        pw = compute_pos_weight(train_ds)
    else:
        pw = args.pos_weight
        print(f"📊 pos_weight manuel: {pw}")

    criterion = CombinedLoss(pos_weight=pw, device=device)

    # --- OPTIMIZER ---
    # Différencier le LR encodeur vs décodeur (fine-tuning)
    if hasattr(model, 'encoder'):
        encoder_params = list(model.encoder.parameters())
        decoder_params = [p for p in model.parameters() if not any(p is ep for ep in encoder_params)]
        optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': args.lr * 0.1},  # Encodeur : LR plus bas
            {'params': decoder_params, 'lr': args.lr},  # Décodeur : LR normal
        ], weight_decay=1e-2)
        print(f"📐 LR différencié: encodeur={args.lr * 0.1:.2e}, décodeur={args.lr:.2e}")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    scheduler = CosineAnnealingWarmup(
        optimizer, args.warmup_epochs, args.epochs, args.lr
    )

    # --- RESUME ---
    start_epoch = 1
    best_iou = 0.0

    if args.resume and os.path.isfile(args.resume):
        print(f"🔄 Chargement du checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Chargement flexible (permet des différences d'architecture)
            state_dict = checkpoint['model_state_dict']
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  ⚠️ Clés manquantes: {len(missing)} (normal si architecture différente)")
            if unexpected:
                print(f"  ⚠️ Clés inattendues: {len(unexpected)}")
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_iou = checkpoint.get('best_iou', 0.0)
            print(f"  ✅ Reprise epoch {start_epoch}, best IoU: {best_iou:.4f}")
        elif isinstance(checkpoint, dict):
            # Essai direct comme state_dict
            missing, unexpected = model.load_state_dict(checkpoint, strict=False)
            print(f"  ⚠️ Poids seuls chargés (pas d'historique)")
        else:
            print(f"  ❌ Format de checkpoint non reconnu")
    elif args.resume:
        print(f"  ❌ Fichier non trouvé: {args.resume}")

    # --- LOGGING ---
    log_file = work_dir / "train_log.csv"
    if not log_file.exists() or start_epoch == 1:
        with open(log_file, "w") as f:
            f.write("epoch,time,lr,train_loss,train_iou,val_loss,val_iou\n")

    # Sauvegarder la config
    config = vars(args)
    config['pos_weight_used'] = pw
    config['model_params'] = sum(p.numel() for p in model.parameters())
    with open(work_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # --- BOUCLE D'ENTRAÎNEMENT ---
    patience_counter = 0
    nan_epoch_count = 0

    print(f"\n🎯 Début entraînement: {len(train_ds)} train, {len(val_ds)} val, {args.epochs} epochs")
    print(f"   Batch size: {args.batch_size} | Image size: {args.img_size}")
    print(f"   Patience early stopping: {args.patience} epochs\n")

    for epoch in range(start_epoch, args.epochs + 1):
        now = datetime.now().strftime("%H:%M:%S")
        current_lr = scheduler.step(epoch - 1)

        print(f"\n🔔 Epoch {epoch}/{args.epochs} | 🕒 {now} | 🚀 LR: {current_lr:.2e}")

        # Train
        t_loss, t_iou = train_epoch(model, train_loader, criterion, optimizer, device, use_amp)

        # Validation
        v_loss, v_iou = validate(model, val_loader, criterion, device, use_amp)

        # Protection : si le train produit uniquement des NaN
        if math.isnan(t_loss):
            nan_epoch_count += 1
            print(f"  ❌ Epoch entière NaN ({nan_epoch_count}/3)")
            if nan_epoch_count >= 3:
                print("  💀 3 epochs NaN consécutives. Arrêt de l'entraînement.")
                print("     → Vérifiez votre dataset et réduisez le learning rate.")
                break
            continue
        else:
            nan_epoch_count = 0

        # Debug images (toutes les 5 epochs ou les 10 premières)
        if epoch <= 10 or epoch % 5 == 0:
            print(f"  📸 Debug image epoch {epoch}...")
            save_debug_prediction(model, val_ds, device, epoch, work_dir, args.debug_names)

        # Affichage
        print(f"  📊 T_Loss: {t_loss:.4f} T_IoU: {t_iou:.4f} | V_Loss: {v_loss:.4f} V_IoU: {v_iou:.4f}")

        # CSV
        with open(log_file, "a") as f:
            f.write(f"{epoch},{now},{current_lr:.2e},{t_loss:.4f},{t_iou:.4f},{v_loss:.4f},{v_iou:.4f}\n")

        # Meilleur modèle
        if not math.isnan(v_iou) and v_iou > best_iou:
            best_iou = v_iou
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'config': {
                    'encoder': args.encoder,
                    'img_size': args.img_size,
                    'pos_weight': pw,
                }
            }, work_dir / 'best_model.pth')
            print(f"  ⭐ Nouveau record IoU: {best_iou:.4f}")
        else:
            patience_counter += 1

        # Checkpoint de sécurité
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_iou': best_iou,
        }, work_dir / 'latest.pth')

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⏹️ Early stopping après {args.patience} epochs sans amélioration.")
            print(f"   Meilleur IoU: {best_iou:.4f}")
            break

        gc.collect()
        torch.cuda.empty_cache()

    # --- COURBES FINALES ---
    try:
        import pandas as pd
        df = pd.read_csv(log_file)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(df['epoch'], df['train_loss'], label='Train Loss')
        ax1.plot(df['epoch'], df['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_title('Loss')
        ax2.plot(df['epoch'], df['train_iou'], label='Train IoU')
        ax2.plot(df['epoch'], df['val_iou'], label='Val IoU')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('IoU')
        ax2.legend()
        ax2.set_title('IoU')
        plt.tight_layout()
        plt.savefig(work_dir / 'metrics.png', dpi=150)
        plt.close()
    except ImportError:
        print("  (pandas non disponible pour les courbes finales)")

    print(f"\n✅ Entraînement terminé. Meilleur IoU: {best_iou:.4f}")
    print(f"   Modèle sauvegardé: {work_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()