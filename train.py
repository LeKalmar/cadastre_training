#!/usr/bin/env python3
"""
Cadastre training v5 - Tversky Loss pour contrôle direct P/R

Approche: Au lieu de manipuler pos_weight (qui n'affecte que la BCE),
on utilise la Tversky Loss qui permet de contrôler DIRECTEMENT le
compromis precision/recall via alpha et beta:
  - alpha > beta → pénalise les faux positifs → meilleure precision
  - alpha < beta → pénalise les faux négatifs → meilleur recall

Combinaison: Tversky(0.4) + Skeleton(0.3) + BCE(0.3)

Usage:
    # From scratch avec Tversky (recommandé)
    python train_v5.py --batch-size 8 --epochs 300 --lr 3e-4 \
        --work-dir work_dirs/Desailly_v5

    # Fine-tune depuis v2
    python train_v5.py --batch-size 6 --epochs 200 --lr 5e-5 \
        --resume work_dirs/Desailly_v2/best_model.pth \
        --work-dir work_dirs/Desailly_v5ft
"""

import os, json, argparse, math, gc, random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib;

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


# ============================================================================
# Dataset
# ============================================================================
LINE, IGNORE, TEXT = 1, 2, 3  # codes internes du masque (0 = fond)
TEXT_WEIGHT = 3.0  # surchargé par --text-weight
LINE_IGNORE_PX = -1  # >= 0 : la bande ignorée à moins de N px d'un trait redevient du FOND (traits fins)


def split_mask(masks):
    """(B,4,H,W) -> cible (B,1,H,W), poids (B,1,H,W) ; 0 = pixel ignoré.
    Canaux 2-3 : cible et poids de la sortie « texte » (voir split_text)."""
    return masks[:, 0:1], masks[:, 1:2]


def split_text(masks):
    return masks[:, 2:3], masks[:, 3:4]


class CadastreDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        valid_ext = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        self.images = sorted([f for f in self.img_dir.iterdir() if f.suffix.lower() in valid_ext])
        print(f"📁 {len(self.images)} images depuis {img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")
        if not mask_path.exists(): mask_path = self.mask_dir / img_path.name
        if not mask_path.exists(): raise FileNotFoundError(f"Masque introuvable: {img_path}")

        image = cv2.imread(str(img_path))
        if image is None: raise RuntimeError(f"Image illisible: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None: raise RuntimeError(f"Masque illisible: {mask_path}")
        # Classes : 255 = ligne, 100 = ignoré (dataset v2), 0 = fond.
        # Les anciens masques binaires 0/255 restent lus à l'identique.
        lab = np.zeros(mask.shape, np.uint8)
        lab[mask > 127] = LINE
        lab[(mask > 80) & (mask < 120)] = IGNORE
        lab[(mask > 30) & (mask < 70)] = TEXT  # texte/symbole : fond à pénalité renforcée
        if LINE_IGNORE_PX >= 0:
            # La bande ignorée qui borde chaque trait laissait le modèle l'épaissir
            # gratuitement (~1,5 m sur Douai pour 0,5 m d'encre) : elle redevient du fond.
            # Les autres zones ignorées (non annotées, bord de découpe) restent ignorées.
            near = cv2.dilate((lab == LINE).astype(np.uint8), np.ones((3, 3), np.uint8),
                              iterations=max(1, LINE_IGNORE_PX + 1)) > 0
            lab[(lab == IGNORE) & near] = 0

        if self.transform:
            aug = self.transform(image=image, mask=lab)
            image, lab = aug['image'], aug['mask']
        if not isinstance(image, torch.Tensor):
            aug = ToTensorV2()(image=image, mask=lab)
            image, lab = aug['image'], aug['mask']
        lab = torch.as_tensor(lab)
        target = (lab == LINE).float()
        weight = (lab != IGNORE).float()
        weight[lab == TEXT] = TEXT_WEIGHT
        text = (lab == TEXT).float()
        text_w = (lab != IGNORE).float()
        # (4, H, W) : cible + poids du trait, cible + poids du texte
        return image, torch.stack([target, weight, text, text_w])


# ============================================================================
# Model
# ============================================================================
def build_model(encoder_name="resnet34", pretrained=True, classes=1):
    model = smp.Unet(encoder_name=encoder_name,
                     encoder_weights="imagenet" if pretrained else None,
                     in_channels=3, classes=classes, decoder_attention_type="scse")
    p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"🧠 U-Net + {encoder_name} | {p:.1f}M params | scSE | {classes} sortie(s)")
    return model


# ============================================================================
# Loss v5 — Tversky + clDice + BCE
# ============================================================================
class TverskyLoss(nn.Module):
    """
    Tversky Loss — généralisation de Dice avec contrôle P/R.

    Tversky Index = TP / (TP + alpha*FP + beta*FN)

    - alpha=0.5, beta=0.5  → équivalent Dice
    - alpha=0.7, beta=0.3  → pénalise les FP → meilleure precision
    - alpha=0.3, beta=0.7  → pénalise les FN → meilleur recall

    Pour les lignes cadastrales: alpha=0.6, beta=0.4 est un bon départ
    (légère préférence precision pour des lignes plus fines)
    """

    def __init__(self, alpha=0.6, beta=0.4, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred_prob = torch.sigmoid(pred)
        p_flat = pred_prob.reshape(-1)
        t_flat = target.reshape(-1)

        tp = (p_flat * t_flat).sum()
        fp = (p_flat * (1 - t_flat)).sum()
        fn = ((1 - p_flat) * t_flat).sum()

        tversky = (tp + self.smooth) / (
                tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss — renforce le focus sur les cas difficiles.

    FTL = (1 - Tversky)^gamma
    gamma > 1 → focus sur les exemples mal classés (lignes fines, intersections)
    """

    def __init__(self, alpha=0.6, beta=0.4, gamma=1.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target, weight=None):
        pred_prob = torch.sigmoid(pred)
        w = torch.ones_like(target) if weight is None else weight
        p_flat = pred_prob.reshape(-1)
        t_flat = target.reshape(-1)
        w_flat = w.reshape(-1)

        tp = (p_flat * t_flat * w_flat).sum()
        fp = (p_flat * (1 - t_flat) * w_flat).sum()
        fn = ((1 - p_flat) * t_flat * w_flat).sum()

        tversky = (tp + self.smooth) / (
                tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return (1.0 - tversky) ** self.gamma


class SoftSkeletonLoss(nn.Module):
    """clDice via soft morphological skeletonization"""

    def __init__(self, num_iters=5):
        super().__init__()
        self.num_iters = num_iters

    def soft_erode(self, img):
        return -F.max_pool2d(-img, 3, stride=1, padding=1)

    def soft_dilate(self, img):
        return F.max_pool2d(img, 3, stride=1, padding=1)

    def soft_skeleton(self, img):
        skel = F.relu(img - self.soft_dilate(self.soft_erode(img)))
        for _ in range(self.num_iters):
            img = self.soft_erode(img)
            delta = F.relu(img - self.soft_dilate(self.soft_erode(img)))
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def forward(self, pred, target, weight=None):
        pred_prob = torch.sigmoid(pred)
        # Squelettiser la prédiction BRUTE : la masquer avant reproduirait la forme des bandes
        # ignorées (qui longent les lignes) et récompenserait une prédiction "ligne partout".
        # Les pixels ignorés sont exclus seulement au comptage de la précision topologique.
        v = torch.ones_like(target) if weight is None else (weight > 0).float()
        skel_pred = self.soft_skeleton(pred_prob)
        skel_target = self.soft_skeleton(target)
        tprec = ((skel_pred * skel_target).sum() + 1.0) / ((skel_pred * v).sum() + 1.0)
        tsens = ((skel_pred * skel_target).sum() + 1.0) / (skel_target.sum() + 1.0)
        return 1.0 - 2.0 * tprec * tsens / (tprec + tsens + 1e-7)


class CombinedLossV5(nn.Module):
    """
    v5: Focal Tversky + clDice + BCE légère

    - Focal Tversky (0.40): contrôle direct P/R + focus sur cas durs
    - clDice (0.35): connectivité des lignes
    - BCE (0.25): stabilité de base avec pos_weight modéré
    """

    def __init__(self, alpha=0.6, beta=0.4, gamma=1.5,
                 pos_weight=3.0, device='cuda',
                 tversky_w=0.40, skeleton_w=0.35, bce_w=0.25):
        super().__init__()
        self.focal_tversky = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
        self.skeleton_loss = SoftSkeletonLoss(num_iters=5)
        self.pos_weight = torch.tensor([pos_weight], device=device)
        self.tversky_w = tversky_w
        self.skeleton_w = skeleton_w
        self.bce_w = bce_w

        print(f"📐 Loss v5: FocalTversky({tversky_w:.2f}, α={alpha}, β={beta}, γ={gamma}) "
              f"+ Skel({skeleton_w:.2f}) + BCE({bce_w:.2f}, pw={pos_weight})")

    def forward(self, pred, target, weight=None):
        p = torch.clamp(pred, -20.0, 20.0)
        w = torch.ones_like(target) if weight is None else weight
        l_tversky = self.focal_tversky(p, target, w)
        l_skel = self.skeleton_loss(p, target, w)
        l_bce = F.binary_cross_entropy_with_logits(
            p, target, weight=w, pos_weight=self.pos_weight, reduction='sum') / w.sum().clamp(min=1.0)
        return self.tversky_w * l_tversky + self.skeleton_w * l_skel + self.bce_w * l_bce


class MultiTaskLoss(nn.Module):
    """Sortie 0 = traits (perte v5 inchangée) ; sortie 1 (optionnelle) = texte/numéros
    (BCE + Dice pondérées, zones ignorées exclues), pour guider le découpage par numéros."""

    def __init__(self, line_loss, text_w=0.0):
        super().__init__()
        self.line_loss, self.text_w = line_loss, text_w

    def forward(self, out, masks):
        t, w = split_mask(masks)
        loss = self.line_loss(out[:, 0:1], t, w)
        if self.text_w and out.shape[1] > 1:
            tt, tw = split_text(masks)
            p = torch.clamp(out[:, 1:2], -20.0, 20.0)
            bce = F.binary_cross_entropy_with_logits(p, tt, weight=tw, reduction='sum') / tw.sum().clamp(min=1.0)
            q = torch.sigmoid(p) * tw
            dice = 1 - (2 * (q * tt).sum() + 1) / (q.sum() + (tt * tw).sum() + 1)
            loss = loss + self.text_w * (0.5 * bce + 0.5 * dice)
        return loss


# ============================================================================
# Metrics
# ============================================================================
def calculate_iou(pred, target, threshold=0.5, weight=None):
    with torch.no_grad():
        w = torch.ones_like(target) if weight is None else (weight > 0).float()
        pm = (torch.sigmoid(pred) > threshold).float() * w
        inter = (pm * target).sum()
        union = pm.sum() + (target * w).sum() - inter
        return 1.0 if union == 0 else (inter / union).item()


def calculate_metrics(pred, target, threshold=0.5, weight=None):
    with torch.no_grad():
        w = torch.ones_like(target) if weight is None else (weight > 0).float()
        pm = (torch.sigmoid(pred) > threshold).float()
        tp = (pm * target * w).sum()
        fp = (pm * (1 - target) * w).sum()
        fn = ((1 - pm) * target * w).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        return {'iou': iou.item(), 'precision': prec.item(),
                'recall': rec.item(), 'f1': f1.item()}


def closed_parcels(pred, gt, valid, min_area=400, iou_ok=0.8):
    """Parcelles fermées retrouvées sur une tuile (même définition que
    tools/compare_models.py) : chaque région entre les traits de la vérité terrain
    doit correspondre à une région prédite (IoU > ``iou_ok``). Un seul trou dans un
    trait fusionne deux régions et les fait perdre toutes les deux : c'est ce que
    l'IoU pixel ne voit pas. Renvoie (retrouvées, total)."""
    n_g, lab_g = cv2.connectedComponents((~gt).astype(np.uint8), connectivity=4)
    if n_g < 2:
        return 0, 0
    n_p, lab_p = cv2.connectedComponents((~pred).astype(np.uint8), connectivity=4)
    g, q = lab_g.ravel(), lab_p.ravel()
    area_g = np.bincount(g, minlength=n_g)
    area_p = np.bincount(q, minlength=n_p)
    ign = np.bincount(g, weights=(~valid & ~gt).ravel(), minlength=n_g)
    keep = (area_g >= min_area) & (ign <= 0.1 * area_g)   # trop petite ou zone non annotée
    keep[0] = False
    both = (g > 0) & (q > 0)
    pair, inter = np.unique(g[both].astype(np.int64) * n_p + q[both], return_counts=True)
    pg, pq = pair // n_p, pair % n_p
    best = np.zeros(n_g)
    np.maximum.at(best, pg, inter / (area_g[pg] + area_p[pq] - inter))
    return int((best[keep] > iou_ok).sum()), int(keep.sum())


# ============================================================================
# Augmentations
# ============================================================================
def get_train_transform(sz=512, scale_limit=0.15):
    return A.Compose([
        A.Resize(sz, sz),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # bords ajoutés par rotation/padding = zone ignorée (pas du "fond")
        A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT, fill_mask=IGNORE),
        A.RandomScale(scale_limit=scale_limit, p=0.3),
        A.PadIfNeeded(sz, sz, border_mode=cv2.BORDER_CONSTANT, fill_mask=IGNORE),
        A.RandomCrop(sz, sz, p=1.0),
        A.OneOf([
            A.RandomBrightnessContrast(0.3, 0.3, p=1),
            A.CLAHE(4.0, p=1),
            A.ColorJitter(0.2, 0.2, 0.3, 0.1, p=1),
        ], p=0.5),
        A.OneOf([
            A.HueSaturationValue(20, 30, 20, p=1),
            A.ToGray(p=1),
            A.ChannelShuffle(p=1),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur((3, 5), p=1),
        ], p=0.2),
        A.ElasticTransform(alpha=50, sigma=10, p=0.15),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform(sz=512):
    return A.Compose([
        A.Resize(sz, sz),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ============================================================================
# Train / Val
# ============================================================================
def train_epoch(model, loader, criterion, optimizer, device, amp=True):
    model.train()
    rl, ri, vb, nc = 0, 0, 0, 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        full = masks.to(device, non_blocking=True)
        masks, weights = split_mask(full)
        optimizer.zero_grad(set_to_none=True)

        if amp:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(imgs)
                loss = criterion(out.float(), full)
        else:
            out = model(imgs);
            loss = criterion(out, full)
        out = out[:, 0:1]

        if torch.isnan(loss) or torch.isinf(loss):
            nc += 1;
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        rl += loss.item()
        ri += calculate_iou(out.detach(), masks, weight=weights)
        vb += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'iou': f'{ri / vb:.3f}'})

    if nc: print(f"  ⚠️ {nc} NaN batchs")
    return (rl / vb, ri / vb) if vb else (float('nan'), 0)


@torch.no_grad()
def validate(model, loader, criterion, device, amp=True, rural=None, parcel_thr=0.3):
    """``rural`` : un booléen par tuile, dans l'ordre du jeu (loader non mélangé).
    Les métriques renvoyées comprennent le taux de parcelles fermées (global, rural,
    urbain) et ``parcelles_eq``, la moyenne rural/urbain."""
    model.eval()
    rl, ms, vb = 0, {'iou': 0, 'precision': 0, 'recall': 0, 'f1': 0}, 0
    reg = {'rural': [0, 0], 'urbain': [0, 0]}
    seen = 0
    for imgs, masks in tqdm(loader, desc="Val", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        full = masks.to(device, non_blocking=True)
        masks, weights = split_mask(full)
        if amp:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(imgs);
                loss = criterion(out.float(), full)
        else:
            out = model(imgs);
            loss = criterion(out, full)
        out = out[:, 0:1]
        pb = (torch.sigmoid(out.float())[:, 0] > parcel_thr).cpu().numpy()
        gb = (masks[:, 0] > 0.5).cpu().numpy()
        wb = (weights[:, 0] > 0).cpu().numpy()
        for j in range(len(pb)):
            c = 'rural' if rural is not None and rural[seen + j] else 'urbain'
            ok, tot = closed_parcels(pb[j], gb[j], wb[j])
            reg[c][0] += ok; reg[c][1] += tot
        seen += len(pb)
        if torch.isnan(loss): continue
        m = calculate_metrics(out, masks, weight=weights)
        rl += loss.item()
        for k in ms: ms[k] += m[k]
        vb += 1
    if not vb: return float('nan'), 0, {}
    res = {k: v / vb for k, v in ms.items()}
    rates = {c: reg[c][0] / reg[c][1] for c in reg if reg[c][1]}
    res['parc_rural'] = rates.get('rural', float('nan'))
    res['parc_urbain'] = rates.get('urbain', float('nan'))
    res['parcelles'] = (reg['rural'][0] + reg['urbain'][0]) / max(1, reg['rural'][1] + reg['urbain'][1])
    res['parcelles_eq'] = sum(rates.values()) / len(rates) if rates else 0.0
    return rl / vb, ms['iou'] / vb, res


# ============================================================================
# Debug viz
# ============================================================================
def save_debug(model, ds, device, epoch, wd, names=None):
    model.eval()
    idxs = [i for i, p in enumerate(ds.images) if p.stem in (names or [])] if names else []
    if not idxs: idxs = random.sample(range(len(ds)), min(3, len(ds)))
    fig, ax = plt.subplots(len(idxs), 3, figsize=(15, 5 * len(idxs)))
    if len(idxs) == 1: ax = np.expand_dims(ax, 0)
    with torch.no_grad():
        for i, di in enumerate(idxs):
            img, msk = ds[di];
            inp = img.unsqueeze(0).to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = torch.sigmoid(model(inp)).cpu().float().numpy()[0, 0]
            im = img.permute(1, 2, 0).numpy()
            im = np.clip((im * [.229, .224, .225]) + [.485, .456, .406], 0, 1)
            ax[i, 0].imshow(im);
            ax[i, 0].set_title(f"Orig: {ds.images[di].stem}", fontsize=8)
            gt = msk[0].numpy().copy(); gt[msk[1].numpy() == 0] = 0.4  # gris = ignoré
            ax[i, 1].imshow(gt, cmap='gray', vmin=0, vmax=1);
            ax[i, 1].set_title("GT")
            ax[i, 2].imshow(out > 0.5, cmap='gray');
            ax[i, 2].set_title(f"Pred (Ep {epoch})")
            for a in ax[i]: a.axis('off')
    plt.tight_layout();
    plt.savefig(wd / f"debug_epoch_{epoch}.png", dpi=100);
    plt.close()


# ============================================================================
# Scheduler
# ============================================================================
class CosineScheduler:
    def __init__(self, opt, warmup, total, base_lr, min_lr=1e-6):
        self.opt, self.warmup, self.total = opt, warmup, total
        self.base_lr, self.min_lr = base_lr, min_lr

    def step(self, epoch):
        if epoch < self.warmup:
            lr = self.base_lr * (epoch + 1) / self.warmup
        else:
            prog = (epoch - self.warmup) / max(1, self.total - self.warmup)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * prog))
        for pg in self.opt.param_groups:
            pg['lr'] = lr * pg.get('initial_lr_ratio', 1.0)
        return lr


# ============================================================================
# Main
# ============================================================================
def main():
    P = argparse.ArgumentParser(description="Cadastre v5 - Tversky + Topology Loss")
    P.add_argument('--data-root', default='data/cadastre_lines')
    P.add_argument('--work-dir', default='work_dirs/unet_v5')
    P.add_argument('--resume', type=str, default=None)
    P.add_argument('--epochs', type=int, default=300)
    P.add_argument('--batch-size', type=int, default=8)
    P.add_argument('--lr', type=float, default=3e-4)
    P.add_argument('--img-size', type=int, default=512)
    P.add_argument('--encoder', default='resnet34')
    P.add_argument('--no-pretrained', action='store_true')
    P.add_argument('--no-amp', action='store_true')
    P.add_argument('--warmup-epochs', type=int, default=5)
    P.add_argument('--patience', type=int, default=50)
    P.add_argument('--debug-names', nargs='*', default=None)
    P.add_argument('--text-weight', type=float, default=3.0,
                   help="poids des pixels de texte/symboles (négatifs renforcés, masque v2 = 50)")
    P.add_argument('--stop-after', type=int, default=0,
                   help="arrête après N époques sans changer le calendrier du LR (comparaisons courtes)")
    P.add_argument('--rural-boost', type=float, default=1.0,
                   help='poids supplémentaire des tuiles rurales (rural_frac >= 0.3) ; 0 = désactivé')

    # Tversky params
    P.add_argument('--alpha', type=float, default=0.3,
                   help='Tversky alpha (pénalise FP)')
    P.add_argument('--beta', type=float, default=0.7,
                   help='Tversky beta (pénalise FN, >0.5 indispensable pour fermer les polygones)')
    P.add_argument('--gamma', type=float, default=1.5,
                   help='Focal gamma (>1 = focus sur cas durs)')
    P.add_argument('--pos-weight', type=float, default=2.0,
                   help='BCE pos_weight')

    # Ajouts pour la logique SoDUCO (Contrôle des poids de la Loss)
    P.add_argument('--w-tversky', type=float, default=0.2, help='Poids de la Tversky Loss')
    P.add_argument('--w-skel', type=float, default=0.6, help='Poids de la Topological/Skeleton Loss')
    P.add_argument('--w-bce', type=float, default=0.2, help='Poids de la BCE Loss')
    P.add_argument('--line-ignore-px', type=int, default=-1,
                   help='>= 0 : la bande ignorée à moins de N px des traits devient du fond (traits fins) ; -1 = inchangé')
    P.add_argument('--select', choices=['parcelles', 'iou'], default='parcelles',
                   help="critère du meilleur modèle : taux de parcelles fermées, moyenne rural/urbain "
                        "(défaut), ou l'ancien IoU pixel")
    P.add_argument('--select-thr', type=float, default=0.3,
                   help='seuil de probabilité pour compter les parcelles fermées (= seuil d\'inférence)')
    P.add_argument('--scale-limit', type=float, default=0.15,
                   help="amplitude de l'augmentation d'échelle (0.15 = ±15 %%)")
    P.add_argument('--text-head', action='store_true',
                   help='ajoute une 2e sortie « texte/numéros » (masque v2 = 50)')
    P.add_argument('--w-text', type=float, default=0.3, help='poids de la perte de la sortie texte')

    args = P.parse_args()
    global TEXT_WEIGHT, LINE_IGNORE_PX
    TEXT_WEIGHT = args.text_weight
    LINE_IGNORE_PX = args.line_ignore_px
    classes = 2 if args.text_head else 1

    wd = Path(args.work_dir);
    wd.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = not args.no_amp and dev.type == 'cuda'
    print(f"🚀 {dev} | AMP: {amp}")

    dr = Path(args.data_root)
    train_ds = CadastreDataset(dr / 'images' / 'train', dr / 'labels' / 'train', get_train_transform(args.img_size, args.scale_limit))
    val_ds = CadastreDataset(dr / 'images' / 'val', dr / 'labels' / 'val', get_val_transform(args.img_size))

    # Suréchantillonnage des tuiles rurales (dataset v2 : tiles.csv produit par tools/build_dataset.py)
    sampler = None
    tiles_csv = dr / 'tiles.csv'
    val_rural = None
    if tiles_csv.exists():
        import csv as _csv
        rural = {r['tile']: float(r['rural_frac']) for r in _csv.DictReader(open(tiles_csv))}
        val_rural = [rural.get(p.stem, 0.0) >= 0.3 for p in val_ds.images]
    if args.rural_boost > 0 and tiles_csv.exists():
        w = [1.0 + args.rural_boost * (rural.get(p.stem, 0.0) >= 0.3) for p in train_ds.images]
        sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        n_r = sum(x > 1 for x in w)
        print(f"🌾 Suréchantillonnage rural x{1 + args.rural_boost:g} : {n_r}/{len(w)} tuiles rurales")
    tl = DataLoader(train_ds, args.batch_size, shuffle=sampler is None, sampler=sampler,
                    num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    vl = DataLoader(val_ds, args.batch_size,
                    num_workers=2, pin_memory=True, persistent_workers=True)

    model = build_model(args.encoder, not args.no_pretrained, classes).to(dev)

    crit = MultiTaskLoss(CombinedLossV5(
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        pos_weight=args.pos_weight, device=dev,
        tversky_w=args.w_tversky, skeleton_w=args.w_skel, bce_w=args.w_bce
    ), text_w=args.w_text if args.text_head else 0.0)

    if hasattr(model, 'encoder'):
        ep_list = list(model.encoder.parameters())
        dp_list = [p for p in model.parameters() if not any(p is e for e in ep_list)]
        opt = optim.AdamW([
            {'params': ep_list, 'lr': args.lr * 0.1, 'initial_lr_ratio': 0.1},
            {'params': dp_list, 'lr': args.lr, 'initial_lr_ratio': 1.0},
        ], weight_decay=1e-2)
        print(f"📐 LR: enc={args.lr * 0.1:.2e} dec={args.lr:.2e}")
    else:
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    sched = CosineScheduler(opt, args.warmup_epochs, args.epochs, args.lr)

    best_iou = 0.0
    best_score = -1.0   # critère --select ; repart de zéro à chaque lancement
    if args.resume and os.path.isfile(args.resume):
        print(f"🔄 Loading: {args.resume}")
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        if isinstance(ck, dict) and 'model_state_dict' in ck:
            sd = ck['model_state_dict']
            own = model.state_dict()
            for k in [k for k in sd if k in own and sd[k].shape != own[k].shape]:
                # tête 1 sortie -> 2 sorties : on garde la sortie « trait » apprise
                old = sd[k]
                new = own[k].clone(); new[:old.shape[0]] = old; sd[k] = new
                print(f"  ↪ {k} : {tuple(old.shape)} étendu à {tuple(new.shape)}")
            miss, _ = model.load_state_dict(sd, strict=False)
            if miss: print(f"  ⚠️ {len(miss)} missing keys")
            prev = ck.get('best_iou', 0);
            ep_prev = ck.get('epoch', '?')
            print(f"  ✅ Poids chargés (ep {ep_prev}, IoU {prev:.4f})")
        elif isinstance(ck, dict):
            model.load_state_dict(ck, strict=False)

    lf = wd / "train_log.csv"
    lf.write_text("epoch,time,lr,train_loss,train_iou,val_loss,val_iou,precision,recall,f1,"
                  "parcelles_eq,parc_rural,parc_urbain\n")
    (wd / "config.json").write_text(json.dumps(vars(args), indent=2))

    pc, nc = 0, 0
    print(f"\n🎯 v5: {len(train_ds)} train, {len(val_ds)} val")
    print(f"   {args.epochs} ep | batch={args.batch_size} | patience={args.patience}")
    print(f"   Tversky α={args.alpha} β={args.beta} γ={args.gamma}\n")

    for ep in range(1, args.epochs + 1):
        now = datetime.now().strftime("%H:%M:%S")
        lr = sched.step(ep - 1)
        print(f"\n🔔 Ep {ep}/{args.epochs} | {now} | LR: {lr:.2e}")

        tl_, ti = train_epoch(model, tl, crit, opt, dev, amp)
        vl_, vi, vm = validate(model, vl, crit, dev, amp, rural=val_rural, parcel_thr=args.select_thr)

        if math.isnan(tl_):
            nc += 1
            if nc >= 3: print("💀 3x NaN. Stop."); break
            continue
        nc = 0

        if ep <= 10 or ep % 5 == 0:
            save_debug(model, val_ds, dev, ep, wd, args.debug_names)

        p = vm.get('precision', 0) if vm else 0
        r = vm.get('recall', 0) if vm else 0
        f1 = vm.get('f1', 0) if vm else 0
        pe = vm.get('parcelles_eq', 0) if vm else 0
        pru = vm.get('parc_rural', float('nan')) if vm else float('nan')
        pur = vm.get('parc_urbain', float('nan')) if vm else float('nan')
        print(f"  📊 TL:{tl_:.4f} TI:{ti:.4f} | VL:{vl_:.4f} VI:{vi:.4f} "
              f"P:{p:.3f} R:{r:.3f} F1:{f1:.3f} | parcelles fermées {pe:.3f} "
              f"(rural {pru:.3f}, urbain {pur:.3f})")

        with open(lf, "a") as f:
            f.write(f"{ep},{now},{lr:.2e},{tl_:.4f},{ti:.4f},{vl_:.4f},{vi:.4f},"
                    f"{p:.4f},{r:.4f},{f1:.4f},{pe:.4f},{pru:.4f},{pur:.4f}\n")

        cur = pe if args.select == 'parcelles' else vi
        if not math.isnan(vi) and cur > best_score:
            best_score = cur
            best_iou = vi   # IoU pixel du modèle retenu (clé lue par les autres outils)
            pc = 0
            torch.save({
                'epoch': ep, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(), 'best_iou': best_iou,
                'best_score': best_score, 'select': args.select,
                'metrics': {k: float(v) for k, v in vm.items()},
                'config': {'encoder': args.encoder, 'img_size': args.img_size,
                           'alpha': args.alpha, 'beta': args.beta, 'classes': classes,
                           'outputs': ['line', 'text'][:classes],
                           'line_ignore_px': args.line_ignore_px}
            }, wd / 'best_model.pth')
            print(f"  ⭐ Meilleur ({args.select}) : {best_score:.4f} | IoU {best_iou:.4f}")
        else:
            pc += 1

        torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(), 'best_iou': best_iou},
                   wd / 'latest.pth')

        if args.stop_after and ep >= args.stop_after:
            print(f"\n⏹️ Arrêt demandé après {ep} époques. Meilleur ({args.select}) : {best_score:.4f}");
            break

        if pc >= args.patience:
            print(f"\n⏹️ Early stop. Meilleur ({args.select}) : {best_score:.4f}");
            break

        gc.collect();
        torch.cuda.empty_cache()

    # Courbes
    try:
        import pandas as pd
        df = pd.read_csv(lf)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].plot(df['epoch'], df['train_iou'], alpha=0.5, label='Train')
        axes[0].plot(df['epoch'], df['val_iou'], alpha=0.5, label='Val')
        axes[0].plot(df['epoch'], df['val_iou'].rolling(10).mean(), lw=2, label='Val avg10')
        axes[0].legend();
        axes[0].set_title('IoU');
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(df['epoch'], df['train_loss'], alpha=0.5, label='Train')
        axes[1].plot(df['epoch'], df['val_loss'], alpha=0.5, label='Val')
        axes[1].legend();
        axes[1].set_title('Loss');
        axes[1].grid(True, alpha=0.3)
        axes[2].plot(df['epoch'], df['precision'], label='Precision')
        axes[2].plot(df['epoch'], df['recall'], label='Recall')
        axes[2].plot(df['epoch'], df['val_iou'], label='IoU', ls='--')
        if 'parcelles_eq' in df:
            axes[2].plot(df['epoch'], df['parcelles_eq'], label='Parcelles fermées', lw=2)
        axes[2].legend();
        axes[2].set_title('P / R / IoU');
        axes[2].grid(True, alpha=0.3)
        plt.tight_layout();
        plt.savefig(wd / 'metrics.png', dpi=150);
        plt.close()
    except:
        pass

    print(f"\n✅ Done. Meilleur ({args.select}) : {best_score:.4f} | IoU {best_iou:.4f}")


if __name__ == '__main__':
    main()