#!/usr/bin/env python3
"""Exporte la base d'annotation en jeu d'entraînement (format cadastre_lines_v2).

Feuilles retenues : statut « validee » (toute la zone vérifiée), et les zones
validées des feuilles « en_cours ».

Masque (PNG 8 bits, lu tel quel par train.py) :
    255 = limite d'une parcelle valide (cœur de LINE_W px), là où il y a de l'encre
     50 = faux trait d'une proposition IA rejetée (bruit, texte, hachures, double
          trait, marge, sur-segmentation) : fond à pénalité renforcée
    100 = ignoré : bande autour des limites, limite sans encre, parcelle encore
          « a_verifier », zone ignorée, hors zone vérifiée, bord/hors du scan
      0 = fond
Développements transportés (colonne ``developpement``) : la limite est tracée à sa
position SUR LE SCAN (transformation inverse) et son emplacement réel, qui est
un report, est ignoré (100) : le dessin à petite échelle qui s'y trouve n'est
jamais compté comme fond.
Zone vérifiée d'une feuille : ses zones_validees ; à défaut, si elle vient de
l'export GénéFoncier (souvent numérisée en partie), l'enveloppe de ses parcelles ;
sinon (feuille IA/manuelle validée), toute son emprise.

Le split train/val est attribué par feuille (hachage du nom, --val-frac) puis
VERROUILLÉ en base. Chaque export est enregistré dans la table datasets.

    .venv/bin/python tools/annotation/export_dataset.py --nom v3 [--out data/cadastre_lines_v3]
"""
import argparse
import csv
import hashlib
import json
import sys
import unicodedata
import re
from datetime import datetime
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
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

TILE, STRIDE = 512, 384
LINE_W, IGNORE_BAND, CROP_BAND = 3, 1, 12          # = build_dataset.py
INK_T, INK_NEAR, GAP_BRIDGE = 20, 2, 6
ZONE_CLOSE_M = 6.0
NEG_GAP_PX = 4      # un faux trait à moins de 4 px d'une vraie limite n'est pas négatif


def slug(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60]


def thick(mask, w):
    return ndimage.binary_dilation(mask, iterations=max(0, (w - 1) // 2)) if w > 1 else mask


def burn(geoms, T, shape_):
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(shape_, bool)
    return rasterize([(g, 1) for g in geoms], out_shape=shape_, transform=T,
                     all_touched=True, dtype="uint8").astype(bool)


DEV_IGNORE_M = 1.0   # marge ignorée autour de l'emplacement réel d'un développement


def vers_scan(g, developpement):
    """Ramène une géométrie transportée à sa position sur le scan (inverse de la
    similitude enregistrée par l'outil QGIS, ordre QTransform)."""
    from shapely.affinity import affine_transform

    m11, m12, m21, m22, dx, dy = json.loads(developpement)
    inv = np.linalg.inv(np.array([[m11, m21, dx], [m12, m22, dy], [0.0, 0.0, 1.0]]))
    return affine_transform(g, [inv[0, 0], inv[0, 1], inv[1, 0], inv[1, 1], inv[0, 2], inv[1, 2]])


def load_parcelles(cur, fid, statut_p, reportes):
    """Géométries à leur position sur le scan ; les emplacements réels des
    développements transportés vont dans ``reportes`` (à ignorer)."""
    out = []
    for g, dev in cur.execute("select ST_AsBinary(geom), developpement from parcelles "
                              "where feuille_id=%s and statut=%s", (fid, statut_p)):
        g = C.geom_of(g)
        if dev:
            reportes.append(g.buffer(DEV_IGNORE_M))
            g = vers_scan(g, dev)
        out.append(g)
    return out


def load_feuille(cur, fid, statut, source, fgeom):
    q = lambda sql: [C.geom_of(r[0]) for r in cur.execute(sql, (fid,))]  # noqa: E731
    reportes = []
    valides = load_parcelles(cur, fid, "valide", reportes)
    doutes = load_parcelles(cur, fid, "a_verifier", reportes)
    negatifs = [C.geom_of(r[0]) for r in cur.execute(
        "select ST_AsBinary(geom_ia) from parcelles_pending where feuille_id=%s "
        "and review_status='rejected' and motif_rejet = any(%s) and geom_ia is not null", (fid, list(C.NEGATIVE_MOTIFS)))]
    ignorees = q("select ST_AsBinary(geom) from zones_ignorees where feuille_id=%s") + reportes
    zv = q("select ST_AsBinary(geom) from zones_validees where feuille_id=%s")
    if zv:
        zone = unary_union(zv).intersection(fgeom)
    elif statut != "validee":
        zone = None                                 # en cours sans zone validée : rien
    elif source in ("export", "scan"):
        u = unary_union(valides) if valides else None
        zone = u.buffer(ZONE_CLOSE_M).buffer(-ZONE_CLOSE_M).intersection(fgeom) if u else None
    else:
        zone = fgeom
    return valides, doutes, negatifs, ignorees, zone


def tiles_for(raster, dx, dy, valides, doutes, negatifs, ignorees, zone, out_dir, name, rng):
    rows = []
    sx = lambda g: g  # noqa: E731  (les géométries sont déjà calées sur le scan)
    b_val = [g.boundary for g in valides]
    b_dou = [g.boundary for g in doutes]
    b_neg = [g.boundary for g in negatifs]
    with rasterio.open(raster) as src:
        H, W = src.height, src.width
        zb = zone.bounds
        inv = ~src.transform
        c0, r0 = inv * (zb[0], zb[3])
        c1, r1 = inv * (zb[2], zb[1])
        x_lo, x_hi = max(0, int(c0) - TILE), min(W - TILE, int(c1))
        y_lo, y_hi = max(0, int(r0) - TILE), min(H - TILE, int(r1))
        trees = {k: (STRtree(v), v) for k, v in
                 (("val", b_val), ("dou", b_dou), ("neg", b_neg), ("ign", ignorees)) if v}
        for y0 in range(max(0, y_lo), max(0, y_hi) + 1, STRIDE):
            for x0 in range(max(0, x_lo), max(0, x_hi) + 1, STRIDE):
                win = Window(x0, y0, TILE, TILE)
                T = win_transform(win, src.transform)
                tb = box(*rasterio.windows.bounds(win, src.transform))
                if not tb.intersects(zone):
                    continue
                zt = burn([zone.intersection(tb)], T, (TILE, TILE))
                if zt.mean() < 0.5:
                    continue
                valid = src.read_masks(1, window=win) > 0
                if valid.mean() < 0.6:
                    continue
                rgb = src.read([1, 2, 3], window=win)

                def near(k):
                    if k not in trees:
                        return []
                    t, v = trees[k]
                    return [v[int(i)] for i in t.query(tb)]

                core = burn([sx(g) for g in near("val")], T, (TILE, TILE))
                dark = ((765 - rgb.sum(0, dtype=np.uint16)) // 3).astype(np.int16)
                ink = (dark - ndimage.minimum_filter(dark, size=15)) > INK_T
                ink_near = ndimage.binary_dilation(ink, iterations=INK_NEAR)
                pos = core & ndimage.binary_dilation(core & ink_near, iterations=GAP_BRIDGE)
                line = thick(pos, LINE_W) & valid
                unsure = core & ~pos
                doubt = thick(burn(near("dou"), T, (TILE, TILE)), LINE_W)
                neg = thick(burn(near("neg"), T, (TILE, TILE)), LINE_W)
                neg &= ~ndimage.binary_dilation(core | doubt, iterations=NEG_GAP_PX)
                ign = (ndimage.binary_dilation(line, iterations=IGNORE_BAND) & ~line)
                ign |= ndimage.binary_dilation(unsure, iterations=2) | doubt
                ign |= burn(near("ign"), T, (TILE, TILE)) | ~zt | ~valid
                ign |= ~ndimage.binary_erosion(valid, iterations=CROP_BAND)
                n_line = int(line.sum())
                if n_line < 150 and not neg.any() and rng.random() > 0.05:
                    continue
                if ign.mean() > 0.5:
                    continue
                mask = np.zeros((TILE, TILE), np.uint8)
                mask[neg & ~ign] = 50
                mask[ign] = 100
                mask[line] = 255
                tname = f"{name}_y{y0}_x{x0}"
                Image.fromarray(np.transpose(rgb, (1, 2, 0))).save(out_dir / "images" / f"{tname}.jpg",
                                                                   quality=95)
                Image.fromarray(mask).save(out_dir / "labels" / f"{tname}.png", optimize=True)
                rows.append(dict(tile=tname, line_frac=round(n_line / TILE ** 2, 4),
                                 neg_frac=round(float((mask == 50).mean()), 4),
                                 ignore_frac=round(float(ign.mean()), 3)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nom", required=True, help="nom du jeu (unique), ex. v3")
    ap.add_argument("--out", help="dossier de sortie (défaut data/cadastre_lines_<nom>)")
    ap.add_argument("--val-frac", type=float, default=0.12)
    a = ap.parse_args()
    out = Path(a.out) if a.out else C.ROOT / "data" / f"cadastre_lines_{a.nom}"
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} existe déjà et n'est pas vide.")

    db = C.connect()
    cur = db.cursor()
    if cur.execute("select 1 from datasets where nom=%s", (a.nom,)).fetchone():
        sys.exit(f"Le jeu « {a.nom} » existe déjà dans la table datasets.")
    feuilles = cur.execute(
        "select f.id, f.nom, f.raster, f.dx, f.dy, f.statut, f.source, f.split, ST_AsBinary(f.geom), "
        "greatest(f.modifie_le, (select max(p.modifie_le) from parcelles p where p.feuille_id=f.id)) "
        "from feuilles f where f.raster is not null and (f.statut='validee' or (f.statut='en_cours' "
        "and exists (select 1 from zones_validees z where z.feuille_id=f.id))) order by f.nom"
    ).fetchall()
    if not feuilles:
        sys.exit("Aucune feuille validée (ni zone validée) à exporter.")

    # split verrouillé : attribué une fois pour toutes
    for fid, nom, *_rest in feuilles:
        split = _rest[5]
        if split is None:
            h = int(hashlib.md5(nom.encode()).hexdigest(), 16) / 16 ** 32
            cur.execute("update feuilles set split=%s where id=%s",
                        ("val" if h < a.val_frac else "train", fid))
    db.commit()
    feuilles = [(f[0], f[1], f[2], f[3], f[4], f[5], f[6],
                 cur.execute("select split from feuilles where id=%s", (f[0],)).fetchone()[0],
                 f[8], f[9]) for f in feuilles]

    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    all_rows, per_feuille = [], []
    for fid, nom, raster, dx, dy, statut, source, split, fgeom_wkb, etat in feuilles:
        fgeom = C.geom_of(fgeom_wkb)
        valides, doutes, negatifs, ignorees, zone = load_feuille(cur, fid, statut, source, fgeom)
        if zone is None or zone.is_empty:
            print(f"  {nom[:50]:50} pas de zone vérifiée, ignorée", flush=True)
            continue
        tmp = out / "_tmp"
        (tmp / "images").mkdir(parents=True, exist_ok=True)
        (tmp / "labels").mkdir(parents=True, exist_ok=True)
        name = f"{slug(nom)}_{str(fid)[:6]}"
        rows = tiles_for(raster, dx, dy, valides, doutes, negatifs, ignorees, zone, tmp, name, rng)
        for r in rows:
            for sub, ext in (("images", "jpg"), ("labels", "png")):
                (tmp / sub / f"{r['tile']}.{ext}").rename(out / sub / split / f"{r['tile']}.{ext}")
            r.update(group=nom, sheet=nom, raster=raster, split=split, feuille_id=str(fid))
        all_rows += rows
        per_feuille.append((fid, split, len(valides), len(negatifs), len(rows), etat))
        print(f"  {split:5} {nom[:50]:50} {len(valides):5} parcelles {len(negatifs):4} négatifs "
              f"{len(rows):4} tuiles", flush=True)
    (out / "_tmp" / "images").rmdir() if (out / "_tmp" / "images").exists() else None
    (out / "_tmp" / "labels").rmdir() if (out / "_tmp" / "labels").exists() else None
    (out / "_tmp").rmdir() if (out / "_tmp").exists() else None
    if not all_rows:
        sys.exit("Aucune tuile produite.")

    with open(out / "tiles.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    summary = dict(nom=a.nom, cree_le=datetime.now().isoformat(timespec="seconds"),
                   n_tiles=len(all_rows), n_train=sum(r["split"] == "train" for r in all_rows),
                   n_val=sum(r["split"] == "val" for r in all_rows), n_feuilles=len(per_feuille),
                   val_feuilles=sorted({r["group"] for r in all_rows if r["split"] == "val"}),
                   n_parcelles=sum(p[2] for p in per_feuille), n_negatifs=sum(p[3] for p in per_feuille),
                   tile=TILE, line_w=LINE_W, ignore_band=IGNORE_BAND)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    ds_id = cur.execute("insert into datasets (nom, chemin, params, resume) values (%s,%s,%s,%s) returning id",
                        (a.nom, str(out), json.dumps({"val_frac": a.val_frac, "tile": TILE,
                                                      "stride": STRIDE, "line_w": LINE_W}),
                         json.dumps(summary, ensure_ascii=False))).fetchone()[0]
    for fid, split, n_p, n_n, n_t, etat in per_feuille:
        cur.execute("insert into dataset_feuilles values (%s,%s,%s,%s,%s,%s,%s)",
                    (ds_id, fid, split, n_p, n_n, n_t, etat))
    db.commit()
    print(json.dumps({k: v for k, v in summary.items() if k != "val_feuilles"}, indent=2,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
