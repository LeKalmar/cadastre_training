"""Jeu d'entraînement du DÉTECTEUR de numéros de parcelles, à partir des points posés dans QGIS
(table numeros_positions, outil « Pointer les numéros », touche N).

Segmentation à 2 classes (format de tools/typologie/build_dataset.py, entraînée par train_seg.py) :
    1 numéro     disque de --rayon px autour de chaque point pointé
    0 fond       intérieur des parcelles principales POINTÉES, hors disque et hors bande autour
  255 ignoré     parcelles non pointées (leur numéro existe mais n'est pas situé), hors parcelles,
                 couronne de --marge px autour du disque, hors du scan
La détection = composantes de la classe 1 -> centres ; la lecture se fait ensuite sur une vignette
centrée (tools/numeros/lire_numeros.py).

    .venv/bin/python tools/numeros/build_dataset.py --out data/numeros_det_v1
    .venv/bin/python tools/typologie/train_seg.py --data data/numeros_det_v1 --work-dir work_dirs/numeros_det_v1
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.windows import transform as win_transform
from scipy import ndimage
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "annotation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "typologie"))
import common as C  # noqa: E402
from build_dataset import slug, split_de  # noqa: E402

TILE, STRIDE = 512, 384
CLASSES = ["fond", "numero"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(C.ROOT / "data" / "numeros_det_v1"))
    ap.add_argument("--rayon", type=int, default=10)
    ap.add_argument("--marge", type=int, default=14)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--min-points", type=int, default=20, help="points minimum pour retenir une feuille")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} existe déjà et n'est pas vide.")
    cur = C.connect().cursor()
    feuilles = cur.execute(
        "SELECT f.id, f.nom, f.raster, f.split, count(n.id) FROM numeros_positions n JOIN feuilles f ON f.id = n.feuille_id "
        "GROUP BY 1, 2, 3, 4 HAVING count(n.id) >= %s ORDER BY 2", (a.min_points,)).fetchall()
    if not feuilles:
        sys.exit(f"Aucune feuille avec >= {a.min_points} numéros pointés (QGIS, touche N).")
    for sp in ("train", "val"):
        for sub in ("images", "labels"):
            (out / sp / sub).mkdir(parents=True, exist_ok=True)
    tiles = []
    for fid, nom, raster, split, npts in feuilles:
        pts = [(C.geom_of(g), str(m)) for g, m in cur.execute(
            "SELECT ST_AsBinary(geom), main_key FROM numeros_positions WHERE feuille_id = %s", (fid,))]
        pointes = {m for _, m in pts}
        par_main = defaultdict(list)
        for g, m in cur.execute("SELECT ST_AsBinary(geom), coalesce(par_main_id, id)::text FROM parcelles "
                                "WHERE feuille_id = %s AND statut = 'valide' AND developpement IS NULL", (fid,)):
            par_main[m].append(C.geom_of(g))
        fond = [g for m, gs in par_main.items() if m in pointes for g in gs]
        sp = split_de(nom, split, a.val_frac)
        name = f"{slug(nom)}_{str(fid)[:6]}"
        n_t = 0
        with rasterio.open(raster) as src:
            inv = ~src.transform
            cols_rows = [inv * (p.x, p.y) for p, _ in pts]
            vus = set()
            for c, r in cols_rows:                    # tuiles autour des points (pas tout le scan)
                x0 = int(min(max(0, (c - TILE / 2) // STRIDE * STRIDE), max(0, src.width - TILE)))
                y0 = int(min(max(0, (r - TILE / 2) // STRIDE * STRIDE), max(0, src.height - TILE)))
                for x, y in ((x0, y0), (x0 + STRIDE, y0), (x0, y0 + STRIDE), (x0 + STRIDE, y0 + STRIDE)):
                    x, y = min(x, max(0, src.width - TILE)), min(y, max(0, src.height - TILE))
                    if (x, y) in vus:
                        continue
                    vus.add((x, y))
                    win = Window(x, y, TILE, TILE)
                    T = win_transform(win, src.transform)
                    tb = box(*rasterio.windows.bounds(win, src.transform))
                    f_in = [g for g in fond if g.intersects(tb)]
                    if not f_in:
                        continue
                    lab = np.full((TILE, TILE), 255, np.uint8)
                    lab[rasterize([(g, 1) for g in f_in], out_shape=(TILE, TILE), transform=T, dtype="uint8") > 0] = 0
                    centres = np.zeros((TILE, TILE), bool)
                    for (cc, rr) in cols_rows:
                        ci, ri = int(round(cc - x)), int(round(rr - y))
                        if 0 <= ci < TILE and 0 <= ri < TILE:
                            centres[ri, ci] = True
                    if not centres.any():
                        continue
                    d = ndimage.distance_transform_edt(~centres)
                    lab[(d > a.rayon) & (d <= a.rayon + a.marge)] = 255
                    lab[d <= a.rayon] = 1
                    valid = src.read_masks(1, window=win) > 0
                    lab[~valid] = 255
                    if (lab != 255).mean() < 0.1:
                        continue
                    rgb = src.read([1, 2, 3], window=win)
                    tname = f"{name}_y{y}_x{x}"
                    Image.fromarray(np.transpose(rgb, (1, 2, 0))).save(out / sp / "images" / f"{tname}.jpg", quality=95)
                    Image.fromarray(lab).save(out / sp / "labels" / f"{tname}.png", optimize=True)
                    tiles.append({"tile": tname, "f_fond": round(float((lab == 0).mean()), 4),
                                  "f_numero": round(float((lab == 1).mean()), 4), "split": sp, "feuille": nom})
                    n_t += 1
        print(f"  {sp:5} {nom[:50]:50} {npts:5} numéros pointés -> {n_t} tuiles", flush=True)
    with open(out / "tiles.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(tiles[0]))
        w.writeheader()
        w.writerows(tiles)
    (out / "summary.json").write_text(json.dumps(
        {"classes": CLASSES, "ignore": 255, "rayon": a.rayon, "marge": a.marge,
         "n_train": sum(t["split"] == "train" for t in tiles), "n_val": sum(t["split"] == "val" for t in tiles)},
        indent=2, ensure_ascii=False))
    print(f"{len(tiles)} tuiles -> {out}")


if __name__ == "__main__":
    main()
