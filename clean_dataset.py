import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import argparse


def analyze_dataset(dataset_dir, threshold_kb=3.0):
    """
    Analyse la distribution des pixels blancs dans les masques.
    Permet de choisir un seuil éclairé avant de supprimer quoi que ce soit.
    """
    mask_dir = Path(dataset_dir) / "labels"
    img_dir = Path(dataset_dir) / "images"

    mask_files = sorted(list(mask_dir.glob("*.png")) + list(mask_dir.glob("*.jpg")))
    print(f"🔍 Analyse de {len(mask_files)} masques...\n")

    pixel_counts = []
    file_sizes_kb = []
    categories = {"vide": 0, "quasi_vide": 0, "sparse": 0, "dense": 0}

    for mask_path in tqdm(mask_files, desc="Analyse"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        n_white = np.count_nonzero(mask > 127)
        size_kb = mask_path.stat().st_size / 1024
        pixel_counts.append(n_white)
        file_sizes_kb.append(size_kb)

        total_pixels = mask.shape[0] * mask.shape[1]
        ratio = n_white / total_pixels

        if n_white == 0:
            categories["vide"] += 1
        elif n_white < 50:
            categories["quasi_vide"] += 1
        elif ratio < 0.01:
            categories["sparse"] += 1
        else:
            categories["dense"] += 1

    pixel_counts = np.array(pixel_counts)
    file_sizes_kb = np.array(file_sizes_kb)

    print("=" * 55)
    print("📊 DISTRIBUTION DES PIXELS BLANCS PAR MASQUE")
    print("=" * 55)
    print(f"  Total masques analysés : {len(pixel_counts)}")
    print(f"  Entièrement vides (0 px)    : {categories['vide']:>5}")
    print(f"  Quasi-vides (<50 px)        : {categories['quasi_vide']:>5}")
    print(f"  Sparse (<1% pixels blancs)  : {categories['sparse']:>5}")
    print(f"  Denses (>=1% pixels blancs) : {categories['dense']:>5}")
    print()
    print("  Percentiles pixels blancs :")
    for p in [0, 5, 10, 25, 50, 75, 90, 95, 100]:
        print(f"    p{p:>3} : {np.percentile(pixel_counts, p):>8.0f} px")
    print()
    print("  Taille fichiers (KB) :")
    for p in [5, 10, 25, 50]:
        print(f"    p{p:>3} : {np.percentile(file_sizes_kb, p):>6.1f} KB")
    print()

    # Recommandations
    p5 = int(np.percentile(pixel_counts, 5))
    p10 = int(np.percentile(pixel_counts, 10))
    print("=" * 55)
    print("💡 RECOMMANDATIONS DE SEUIL")
    print("=" * 55)
    print(f"  Seuil conservateur  (p5)  : {p5} px  → supprime ~5%")
    print(f"  Seuil modéré       (p10)  : {p10} px  → supprime ~10%")
    print(f"  Seuil agressif     (vide+quasi) : supprime {categories['vide'] + categories['quasi_vide']} tuiles")
    print()

    return pixel_counts, mask_files


def clean_dataset(dataset_dir, min_pixels=50, dry_run=True):
    """
    Supprime les tuiles dont le masque a moins de min_pixels pixels blancs.
    dry_run=True : simulation sans suppression réelle.
    """
    mask_dir = Path(dataset_dir) / "labels"
    img_dir = Path(dataset_dir) / "images"

    mask_files = sorted(list(mask_dir.glob("*.png")) + list(mask_dir.glob("*.jpg")))
    print(f"\n{'🔍 SIMULATION' if dry_run else '🗑️  NETTOYAGE'} — seuil: {min_pixels} pixels\n")

    to_remove = []
    for mask_path in tqdm(mask_files, desc="Scan"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if np.count_nonzero(mask > 127) < min_pixels:
            img_path = img_dir / mask_path.name
            to_remove.append((mask_path, img_path))

    print(f"\n  Tuiles à supprimer : {len(to_remove)} / {len(mask_files)} ({100*len(to_remove)/len(mask_files):.1f}%)")
    print(f"  Tuiles conservées  : {len(mask_files) - len(to_remove)}")

    if dry_run:
        print("\n  ℹ️  Mode simulation — aucun fichier supprimé.")
        print("  Relancez avec --execute pour supprimer réellement.")
        return

    removed = 0
    for mask_path, img_path in tqdm(to_remove, desc="Suppression"):
        try:
            if mask_path.exists():
                os.remove(mask_path)
            if img_path.exists():
                os.remove(img_path)
            removed += 1
        except Exception as e:
            print(f"❌ Erreur {mask_path.name}: {e}")

    print(f"\n✅ Supprimé : {removed} paires (image + masque)")
    print(f"📂 Dataset restant : {len(mask_files) - removed} tuiles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nettoyage dataset cadastre")
    parser.add_argument("--data", default="data/cadastre_lines", help="Dossier racine du dataset")
    parser.add_argument("--split", default="train", choices=["train", "val", "all"],
                        help="Split à analyser/nettoyer")
    parser.add_argument("--min-pixels", type=int, default=50,
                        help="Seuil minimum de pixels blancs pour garder une tuile (défaut: 50)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Analyse uniquement, sans nettoyage")
    parser.add_argument("--execute", action="store_true",
                        help="Exécution réelle (sans ce flag = simulation)")
    args = parser.parse_args()

    splits = ["train", "val"] if args.split == "all" else [args.split]

    for split in splits:
        split_dir = Path(args.data) / "images" / split
        # On pointe vers le dossier parent qui contient images/ et labels/
        base_dir = Path(args.data)

        # Adapter les chemins pour que analyze/clean cherche dans images/<split> et labels/<split>
        class SplitDataset:
            def __init__(self, base, split):
                self.labels = base / "labels" / split
                self.images = base / "images" / split

        sd = SplitDataset(base_dir, split)

        print(f"\n{'='*55}")
        print(f"  Split : {split.upper()}")
        print(f"{'='*55}")

        # Analyse
        mask_files = sorted(list(sd.labels.glob("*.png")))
        print(f"🔍 Analyse de {len(mask_files)} masques...\n")
        pixel_counts = []
        categories = {"vide": 0, "quasi_vide": 0, "sparse": 0, "dense": 0}
        for mask_path in tqdm(mask_files, desc="Analyse"):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            n = np.count_nonzero(mask > 127)
            pixel_counts.append((n, mask_path))
            total = mask.shape[0] * mask.shape[1]
            if n == 0:
                categories["vide"] += 1
            elif n < 50:
                categories["quasi_vide"] += 1
            elif n / total < 0.01:
                categories["sparse"] += 1
            else:
                categories["dense"] += 1

        counts = np.array([x[0] for x in pixel_counts])
        print(f"  Vides (0 px)         : {categories['vide']:>5}")
        print(f"  Quasi-vides (<50 px) : {categories['quasi_vide']:>5}")
        print(f"  Sparse (<1%)         : {categories['sparse']:>5}")
        print(f"  Denses (>=1%)        : {categories['dense']:>5}")
        print()
        for p in [0, 5, 10, 25, 50, 75]:
            print(f"  p{p:>3} : {np.percentile(counts, p):>8.0f} px")

        if args.analyze_only:
            continue

        # Nettoyage
        to_remove = [(mp, sd.images / mp.name) for n, mp in pixel_counts if n < args.min_pixels]
        print(f"\n  Seuil retenu : {args.min_pixels} px")
        print(f"  À supprimer  : {len(to_remove)} / {len(mask_files)} ({100*len(to_remove)/max(1,len(mask_files)):.1f}%)")

        if not args.execute:
            print("  ℹ️  Simulation — ajoutez --execute pour supprimer réellement.")
            continue

        for mask_path, img_path in tqdm(to_remove, desc="Suppression"):
            try:
                if mask_path.exists():
                    os.remove(mask_path)
                if img_path.exists():
                    os.remove(img_path)
            except Exception as e:
                print(f"❌ {mask_path.name}: {e}")

        print(f"✅ {split} nettoyé : {len(mask_files) - len(to_remove)} tuiles restantes")