#!/usr/bin/env python3
"""
Inférence sur feuilles cadastrales complètes avec chevauchement (overlap tiling).

Le problème des "trous" dans les prédictions vient souvent des bords des tuiles
où le modèle a moins de contexte. La solution : découper avec chevauchement et
faire la moyenne des prédictions dans les zones de recouvrement.

Usage:
    python infer.py --model work_dirs/unet_v2/best_model.pth --input feuille.tif --output result.tif
    python infer.py --model best_model.pth --input-dir ./feuilles/ --output-dir ./results/
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm


def load_model(checkpoint_path, device='cuda'):
    """Charge le modèle depuis un checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        config = checkpoint['config']
        encoder = config.get('encoder', 'resnet34')
    else:
        encoder = 'resnet34'

    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,  # On charge nos propres poids
        in_channels=3,
        classes=1,
        decoder_attention_type="scse",
    )

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def predict_with_overlap(model, image, tile_size=512, overlap=128, device='cuda',
                         threshold=0.4, batch_size=8):
    """
    Inférence par tuiles avec chevauchement.

    Args:
        model: modèle PyTorch
        image: np.array (H, W, 3) RGB
        tile_size: taille des tuiles
        overlap: chevauchement entre tuiles (plus = moins de trous, plus lent)
        threshold: seuil de binarisation final
        batch_size: nombre de tuiles traitées en parallèle

    Returns:
        binary_mask: np.array (H, W) uint8 (0 ou 255)
        prob_map: np.array (H, W) float32 (probabilités brutes)
    """
    h, w = image.shape[:2]
    stride = tile_size - overlap

    # Padding pour couvrir toute l'image
    pad_h = (stride - (h - tile_size) % stride) % stride
    pad_w = (stride - (w - tile_size) % stride) % stride

    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    ph, pw = padded.shape[:2]

    # Accumulateurs
    prob_sum = np.zeros((ph, pw), dtype=np.float64)
    count = np.zeros((ph, pw), dtype=np.float64)

    # Fenêtre de pondération gaussienne (plus de poids au centre)
    window = _gaussian_window(tile_size)

    # Normalisation ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Collecter toutes les positions de tuiles
    positions = []
    for y in range(0, ph - tile_size + 1, stride):
        for x in range(0, pw - tile_size + 1, stride):
            positions.append((y, x))

    # Traitement par batch
    with torch.no_grad():
        for i in tqdm(range(0, len(positions), batch_size), desc="Inférence"):
            batch_positions = positions[i:i + batch_size]
            tiles = []

            for y, x in batch_positions:
                tile = padded[y:y + tile_size, x:x + tile_size]
                tile = (tile.astype(np.float32) / 255.0 - mean) / std
                tile = torch.from_numpy(tile.transpose(2, 0, 1)).float()
                tiles.append(tile)

            batch = torch.stack(tiles).to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                preds = torch.sigmoid(model(batch)).cpu().float().numpy()[:, 0]

            for j, (y, x) in enumerate(batch_positions):
                prob_sum[y:y + tile_size, x:x + tile_size] += preds[j] * window
                count[y:y + tile_size, x:x + tile_size] += window

    # Moyenne pondérée
    count = np.maximum(count, 1e-8)
    prob_map = (prob_sum / count)[:h, :w].astype(np.float32)

    # Binarisation
    binary = (prob_map > threshold).astype(np.uint8) * 255

    return binary, prob_map


def _gaussian_window(size):
    """Crée une fenêtre gaussienne 2D pour le blending"""
    x = np.linspace(-1, 1, size)
    gauss = np.exp(-x ** 2 / 0.5)
    window = np.outer(gauss, gauss)
    return window


def main():
    parser = argparse.ArgumentParser(description="Inférence cadastrale avec chevauchement")
    parser.add_argument('--model', required=True, help='Chemin du checkpoint')
    parser.add_argument('--input', type=str, help='Image unique à traiter')
    parser.add_argument('--input-dir', type=str, help='Dossier d\'images à traiter')
    parser.add_argument('--output', type=str, help='Chemin de sortie (image unique)')
    parser.add_argument('--output-dir', type=str, default='./results', help='Dossier de sortie')
    parser.add_argument('--tile-size', type=int, default=512)
    parser.add_argument('--overlap', type=int, default=128,
                        help='Chevauchement en pixels (128-256 recommandé)')
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--save-prob', action='store_true',
                        help='Sauvegarder aussi la carte de probabilités')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")

    model = load_model(args.model, device)
    print(f"✅ Modèle chargé: {args.model}")

    # Collecter les images à traiter
    images = []
    if args.input:
        images = [Path(args.input)]
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        images = sorted(input_dir.glob('*.tif')) + sorted(input_dir.glob('*.png')) + sorted(input_dir.glob('*.jpg'))

    if not images:
        print("❌ Aucune image trouvée")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        print(f"\n📄 Traitement: {img_path.name}")

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ❌ Impossible de charger {img_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"  Taille: {image.shape[1]}x{image.shape[0]}")

        binary, prob_map = predict_with_overlap(
            model, image,
            tile_size=args.tile_size,
            overlap=args.overlap,
            device=device,
            threshold=args.threshold,
            batch_size=args.batch_size
        )

        # Sauvegarde
        if args.output and len(images) == 1:
            out_path = Path(args.output)
        else:
            out_path = output_dir / f"{img_path.stem}_lines.png"

        cv2.imwrite(str(out_path), binary)
        print(f"  ✅ Sauvegardé: {out_path}")

        if args.save_prob:
            prob_path = output_dir / f"{img_path.stem}_prob.tif"
            cv2.imwrite(str(prob_path), (prob_map * 255).astype(np.uint8))
            print(f"  ✅ Probabilités: {prob_path}")

    print(f"\n✅ Traitement terminé. {len(images)} image(s) traitée(s).")


if __name__ == '__main__':
    main()