import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def split_dataset(source_dir, train_ratio=0.85):
    """
    Répartit les tuiles de data/cadastre_lines_tiled vers data/cadastre_lines/train et /val
    """
    # Chemins source
    src_img_dir = Path(source_dir) / "images"
    src_mask_dir = Path(source_dir) / "labels"
    
    # Chemins destination (structure attendue par ton train.py)
    base_dest = Path("data/cadastre_lines")
    dest_train_img = base_dest / "images" / "train"
    dest_train_mask = base_dest / "labels" / "train"
    dest_val_img = base_dest / "images" / "val"
    dest_val_mask = base_dest / "labels" / "val"

    # Création des dossiers
    for d in [dest_train_img, dest_train_mask, dest_val_img, dest_val_mask]:
        d.mkdir(parents=True, exist_ok=True)

    # Liste des fichiers images (on suppose que les noms sont identiques dans labels)
    all_images = sorted([f.name for f in src_img_dir.iterdir() if f.is_file()])
    random.shuffle(all_images) # Mélange aléatoire

    # Calcul du point de coupure
    split_idx = int(len(all_images) * train_ratio)
    train_files = all_images[:split_idx]
    val_files = all_images[split_idx:]

    def move_files(files, mode):
        img_dest = dest_train_img if mode == "train" else dest_val_img
        mask_dest = dest_train_mask if mode == "train" else dest_val_mask
        
        print(f"📦 Copie des fichiers vers {mode}...")
        for f_name in tqdm(files):
            # Copie Image
            shutil.copy(src_img_dir / f_name, img_dest / f_name)
            # Copie Label (on vérifie l'existence)
            if (src_mask_dir / f_name).exists():
                shutil.copy(src_mask_dir / f_name, mask_dest / f_name)

    move_files(train_files, "train")
    move_files(val_files, "val")

    print(f"\n✅ Répartition terminée !")
    print(f"📊 Train : {len(train_files)} tuiles")
    print(f"📊 Val   : {len(val_files)} tuiles")

if __name__ == "__main__":
    # Source : le dossier où tu as tes tuiles nettoyées
    # Ratio : 0.85 signifie 85% train, 15% validation
    split_dataset("data/cadastre_lines_tiled_1024", train_ratio=0.85)