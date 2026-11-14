"""Jeu d'entraînement du modèle « zone utile » : une image par feuille, réduite.

Seules les zones VALIDÉES dans QGIS (feuille_zones.statut = 'valide') servent : les découpes
DCP brutes sont des compromis, pas une vérité. Chaque feuille donne une image carrée de
--cote px (scan entier réduit, grand côté = --cote, complété par du vide) et son masque :
    0 hors zone   1 zone utile   255 ignoré (hors du scan, remplissage)
Le format (split/images, split/labels, tiles.csv, summary.json) est celui de
tools/typologie/build_dataset.py : tools/typologie/train_seg.py l'entraîne tel quel.

    .venv/bin/python tools/zone/build_zone_dataset.py --out data/zones_v1
    .venv/bin/python tools/typologie/train_seg.py --data data/zones_v1 --work-dir work_dirs/zones_v1 --batch-size 4
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.features import rasterize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "annotation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "typologie"))
import common as C  # noqa: E402
from build_dataset import slug, split_de  # noqa: E402

CLASSES = ["hors_zone", "zone"]


def image_feuille(raster, zone, cote):
    """(rgb HxWx3, masque HxW) au carré cote x cote, et le facteur de réduction."""
    with rasterio.open(raster) as src:
        k = max(src.width, src.height) / cote
        h, w = max(1, round(src.height / k)), max(1, round(src.width / k))
        rgb = src.read([1, 2, 3], out_shape=(3, h, w), resampling=Resampling.average)
        valid = src.dataset_mask(out_shape=(h, w), resampling=Resampling.nearest) > 0
        T = src.transform * src.transform.scale(src.width / w, src.height / h)
    m = rasterize([(zone, 1)], out_shape=(h, w), transform=T, dtype="uint8")
    m[~valid] = 255
    img = np.full((cote, cote, 3), 255, np.uint8)
    lab = np.full((cote, cote), 255, np.uint8)
    img[:h, :w] = np.transpose(rgb, (1, 2, 0))
    lab[:h, :w] = m
    return img, lab, k


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(C.ROOT / "data" / "zones_v1"))
    ap.add_argument("--cote", type=int, default=1024)
    ap.add_argument("--val-frac", type=float, default=0.15)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} existe déjà et n'est pas vide.")
    cur = C.connect().cursor()
    rows = cur.execute(
        "SELECT f.id, f.nom, f.split, coalesce(z.raster, f.raster), ST_AsBinary(z.geom) FROM feuille_zones z "
        "JOIN feuilles f ON f.id = z.feuille_id WHERE z.statut = 'valide' ORDER BY f.nom").fetchall()
    if not rows:
        sys.exit("Aucune zone validée : valider des zones dans QGIS (touches Z, V) d'abord.")
    for sp in ("train", "val"):
        for sub in ("images", "labels"):
            (out / sp / sub).mkdir(parents=True, exist_ok=True)
    tiles = []
    for fid, nom, split, raster, g in rows:
        if not raster or not Path(raster).exists():
            print(f"  plan introuvable : {nom}")
            continue
        sp = split_de(nom, split, a.val_frac)
        img, lab, k = image_feuille(raster, C.geom_of(g), a.cote)
        name = f"{slug(nom)}_{str(fid)[:6]}"
        Image.fromarray(img).save(out / sp / "images" / f"{name}.jpg", quality=92)
        Image.fromarray(lab).save(out / sp / "labels" / f"{name}.png", optimize=True)
        ok = lab != 255
        tiles.append({"tile": name, "f_hors_zone": round(float((lab == 0).mean()), 4),
                      "f_zone": round(float((lab == 1).mean()), 4), "f_ignore": round(1 - float(ok.mean()), 3),
                      "reduction": round(k, 2), "split": sp, "feuille": nom, "feuille_id": str(fid)})
        print(f"  {sp:5} {nom[:50]:50} réduction x{k:.1f}  zone {tiles[-1]['f_zone']:.2f}", flush=True)
    with open(out / "tiles.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(tiles[0]))
        w.writeheader()
        w.writerows(tiles)
    resume = {"classes": CLASSES, "ignore": 255, "cote": a.cote, "n_train": sum(t["split"] == "train" for t in tiles),
              "n_val": sum(t["split"] == "val" for t in tiles)}
    (out / "summary.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False))
    print(json.dumps(resume, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
