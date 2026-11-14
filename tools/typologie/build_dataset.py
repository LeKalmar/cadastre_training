"""Jeu d'entraînement de la typologie visible des parcelles (segmentation par pixel).

Choix de l'utilisateur (2026-11-13) : l'IA découpe chaque parcelle en sous-polygones typés,
comme GénéFoncier (parcelle principale + typologie). On n'apprend que ce qui se VOIT sur la
feuille. Classes (masque PNG, valeur = classe) :
    0 non bâti      cour, labour (papier nu dans la parcelle)
    1 bâti          bâtiment, usine, commerce, bâtiment rural, grange, escalier
    2 bâti public   bâtiment public ou militaire, église, chapelle, temple, école
    3 eau           eau, marais
    4 végétation    bois, jardin, verger
Parcelle sans type : « non bâti » sur une feuille dont le bâti est typé (>= --null-min-bati
parcelles bâties) : GénéFoncier y découpe chaque bâtiment à part, une parcelle sans type n'en
contient donc pas (vérifié à l'œil sur Lille 1 1829 et Wazemmes B 1829) ; ignorée sinon.
  255 ignoré        parcelle d'un type non visible (prairie, cimetière…), hors
                    parcelle, bande de --bord px autour des contours (le trait d'encre et le
                    décalage éventuel du calage), recouvrement de deux classes, hors du scan.

Sources : parcelles « valide » au lien « ok » (contour sur l'encre) de toutes les feuilles.
Split par feuille : celui de la base s'il est posé, sinon le même hachage qu'export_dataset.py
(rien n'est écrit en base). Tuiles 512 px à résolution native, pas de 384.

    .venv/bin/python tools/typologie/build_dataset.py --out data/typologie_v1
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.windows import transform as win_transform
from scipy import ndimage
from shapely import STRtree
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "annotation"))
import common as C  # noqa: E402

TILE, STRIDE = 512, 384
CLASSES = ["non_bati", "bati", "bati_public", "eau", "vegetation"]
IGNORE = 255
TYPES = {
    "cour": 0, "labour": 0,
    "batiment": 1, "usine": 1, "commerce": 1, "bat-rural": 1, "bat_rural": 1, "grange": 1, "escalier": 1,
    "bat-public": 2, "bat-militaire": 2, "eglise": 2, "chapelle": 2, "temple": 2, "scolaire": 2,
    "eau": 3, "marais": 3,
    "bois": 4, "jardin": 4, "verger": 4,
}


def slug(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60]


def split_de(nom, split, val_frac):
    if split:
        return split
    h = int(hashlib.md5(nom.encode()).hexdigest(), 16) / 16 ** 32   # = export_dataset.py
    return "val" if h < val_frac else "train"


def classe(t, t2, null_non_bati=False):
    if t2 == "tunnel":            # passage couvert : dessin ambigu
        return None
    if t is None or not str(t).strip():
        return 0 if null_non_bati else None
    return TYPES.get(str(t).strip().lower())


def tuiles(raster, polys, out, name, bord, min_lab, rng, keep_neg):
    """polys = [(geom, classe)]. Écrit les tuiles, renvoie les lignes du tiles.csv."""
    rows = []
    geoms = [g for g, _ in polys]
    tree = STRtree(geoms)
    ext = box(*np.array([g.bounds for g in geoms]).T[[0, 1]].min(1),
              *np.array([g.bounds for g in geoms]).T[[2, 3]].max(1))
    with rasterio.open(raster) as src:
        H, W = src.height, src.width
        inv = ~src.transform
        xb, yb, xh, yh = ext.bounds
        c0, r0 = inv * (xb, yh)
        c1, r1 = inv * (xh, yb)
        for y0 in range(max(0, int(r0) - TILE // 2), max(1, min(H - TILE, int(r1)) + 1), STRIDE):
            for x0 in range(max(0, int(c0) - TILE // 2), max(1, min(W - TILE, int(c1)) + 1), STRIDE):
                win = Window(x0, y0, TILE, TILE)
                T = win_transform(win, src.transform)
                tb = box(*rasterio.windows.bounds(win, src.transform))
                idx = tree.query(tb)
                if len(idx) == 0:
                    continue
                lab = np.full((TILE, TILE), IGNORE, np.uint8)
                count = np.zeros((TILE, TILE), np.uint8)
                par_classe = defaultdict(list)
                for i in idx:
                    par_classe[polys[int(i)][1]].append(geoms[int(i)])
                for k, gs in par_classe.items():
                    m = rasterize([(g, 1) for g in gs], out_shape=(TILE, TILE), transform=T, dtype="uint8")
                    lab[m > 0] = k
                    count += m
                edges = rasterize([(g.boundary, 1) for i in idx for g in [geoms[int(i)]]],
                                  out_shape=(TILE, TILE), transform=T, all_touched=True, dtype="uint8") > 0
                ign = (count != 1) | ndimage.binary_dilation(edges, iterations=bord)
                valid = src.read_masks(1, window=win) > 0
                ign |= ~ndimage.binary_erosion(valid, iterations=8)
                lab[ign] = IGNORE
                labelled = lab != IGNORE
                if labelled.mean() < min_lab:
                    continue
                fr = [float((lab == k).mean()) for k in range(len(CLASSES))]
                if fr[0] > 0.98 * labelled.mean() and rng.random() > keep_neg:
                    continue                   # tuile de champs seuls : on n'en garde qu'une part
                rgb = src.read([1, 2, 3], window=win)
                tname = f"{name}_y{y0}_x{x0}"
                Image.fromarray(np.transpose(rgb, (1, 2, 0))).save(out / "images" / f"{tname}.jpg", quality=95)
                Image.fromarray(lab).save(out / "labels" / f"{tname}.png", optimize=True)
                rows.append({"tile": tname, **{f"f_{c}": round(f, 4) for c, f in zip(CLASSES, fr)},
                             "f_ignore": round(1 - float(labelled.mean()), 3)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(C.ROOT / "data" / "typologie_v1"))
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--bord", type=int, default=3, help="px ignorés de part et d'autre des contours")
    ap.add_argument("--min-lab", type=float, default=0.2, help="part minimale de pixels étiquetés par tuile")
    ap.add_argument("--keep-neg", type=float, default=0.25, help="part gardée des tuiles 100 %% non bâties")
    ap.add_argument("--min-parcelles", type=int, default=20, help="parcelles typées minimum par feuille")
    ap.add_argument("--null-min-bati", type=int, default=30,
                    help="bâtiments typés à partir desquels une parcelle sans type compte « non bâti »")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} existe déjà et n'est pas vide.")

    cur = C.connect().cursor()
    feuilles = cur.execute(
        "SELECT f.id, f.nom, f.raster, f.split FROM feuilles f WHERE f.raster IS NOT NULL AND EXISTS "
        "(SELECT 1 FROM parcelles p WHERE p.feuille_id = f.id AND p.type IS NOT NULL) ORDER BY f.nom").fetchall()
    for sp in ("train", "val"):
        for sub in ("images", "labels"):
            (out / sp / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    all_rows, bilan = [], Counter()
    for fid, nom, raster, split in feuilles:
        if not Path(raster).exists():
            print(f"  scan introuvable : {nom}", flush=True)
            continue
        polys = []
        n_bati = cur.execute("SELECT count(*) FROM parcelles WHERE feuille_id = %s AND type = ANY(%s)",
                             (fid, [k for k, v in TYPES.items() if v in (1, 2)])).fetchone()[0]
        null_nb = n_bati >= a.null_min_bati
        for t, t2, g, dev in cur.execute(
                "SELECT type, type2, ST_AsBinary(geom), developpement FROM parcelles WHERE feuille_id = %s "
                "AND statut = 'valide' AND lien = 'ok'", (fid,)):
            if dev:                         # développement transporté : position réelle ≠ scan
                continue
            k = classe(t, t2, null_nb)
            if k is not None:
                polys.append((C.geom_of(g), k))
        if len(polys) < a.min_parcelles:
            continue
        sp = split_de(nom, split, a.val_frac)
        name = f"{slug(nom)}_{str(fid)[:6]}"
        rows = tuiles(raster, polys, out / sp, name, a.bord, a.min_lab, rng, a.keep_neg)
        for r in rows:
            r.update(split=sp, feuille=nom, feuille_id=str(fid))
        all_rows += rows
        n = Counter(k for _, k in polys)
        bilan.update({f"{sp}:{CLASSES[k]}": v for k, v in n.items()})
        print(f"  {sp:5} {nom[:48]:48} {len(polys):5} parcelles typées "
              f"({', '.join(f'{CLASSES[k]} {v}' for k, v in sorted(n.items()))}) -> {len(rows)} tuiles", flush=True)
    if not all_rows:
        sys.exit("Aucune tuile.")
    with open(out / "tiles.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    resume = {"classes": CLASSES, "ignore": IGNORE, "types": TYPES, "tile": TILE, "stride": STRIDE,
              "bord_px": a.bord, "null_min_bati": a.null_min_bati, "n_train": sum(r["split"] == "train" for r in all_rows),
              "n_val": sum(r["split"] == "val" for r in all_rows),
              "parcelles": dict(sorted(bilan.items())),
              "val_feuilles": sorted({r["feuille"] for r in all_rows if r["split"] == "val"})}
    (out / "summary.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False))
    print(json.dumps(resume, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
