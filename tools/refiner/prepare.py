#!/usr/bin/env python3
"""Prépare les données du correcteur de polygonisation, feuille par feuille.

Pour chaque raster du jeu cadastre_lines_v2 (tiles.csv) :
  1. mask.tif + prob.tif  : U-Net (GPU), exactement comme extract_cadastre_task ;
  2. gt.geojson           : parcelles validées de la feuille (genefoncier_prod.gpkg),
                            recalées du décalage de calibration.csv, gardées seulement si
                            leur contour tombe sur l'encre (règle de build_dataset.py) ;
  3. grass.geojson        : polygonisation actuelle (shapely nettoyé + GRASS v.clean).

Reprenable : une étape déjà faite n'est pas refaite.

    .venv/bin/python tools/refiner/prepare.py --out work_dirs/refiner [--split val] [--limit N]
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import unicodedata
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_geom
from scipy import ndimage
from shapely import wkb
from shapely.affinity import translate
from shapely.geometry import box, mapping, shape

REPO = "/home/lekalmar/PyCharmMiscProject/Code/Geenfoncier/GeneFoncier/terrapartus"
sys.path.insert(0, REPO)
from georeferencing import grass_vectorize, inference  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GPKG = ROOT / "data/gis_export/genefoncier_prod.gpkg"
CALIB = ROOT / "data/gis_export/calibration.csv"
TILES = ROOT / "data/cadastre_lines_v2/tiles.csv"
MODEL = ROOT / "work_dirs/v2_dinov3_small_night/best_model.pth"

INK_T, INK_NEAR, ACCEPT, MIN_SAMPLES = 20, 2, 0.75, 30   # = build_dataset.py
ZONE_CLOSE_M = 6.0


def slug(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:50] + "_" + hashlib.md5(s.encode()).hexdigest()[:6]


def gpkg_geom(blob):
    flags = blob[3]
    env = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 7]
    return wkb.loads(bytes(blob[8 + env:]))


def year_of(s):
    return int(s[:4]) if s else None


def sheets():
    out = OrderedDict()
    for t in csv.DictReader(open(TILES)):
        out.setdefault(t["raster"], dict(group=t["group"], split=t["split"], sheet=t["sheet"]))
    calib = defaultdict(list)
    for r in csv.DictReader(open(CALIB)):
        if r["raster"] in out and float(r["hit"] or 0) >= 0.5:
            calib[r["raster"]].append(r)
    for rp, m in out.items():
        rows = calib.get(rp, [])
        best = max(rows, key=lambda r: float(r["hit"])) if rows else None
        m["dy"] = round(float(best["dy"])) if best else 0
        m["dx"] = round(float(best["dx"])) if best else 0
        m["feuilles"] = sorted({r["feuille_id"] for r in rows if r["mode"] == "feuille"})
        zone_years = [r["raster_year"] or r["year"] for r in rows if r["mode"] != "feuille"]
        m["zone_year"] = int(float(zone_years[0])) if zone_years and zone_years[0] else None
        m["zone_mode"] = bool(rows) and not m["feuilles"]
        m["hit"] = float(best["hit"]) if best else -1.0
    # Copies d'une même feuille (TIF, COG, montage) : un seul raster par groupe,
    # le mieux calé -- sinon une copie sans calage occupe le dossier du groupe.
    best_of = {}
    for rp, m in out.items():
        cur = best_of.get(m["group"])
        if cur is None or m["hit"] > out[cur]["hit"]:
            best_of[m["group"]] = rp
    return OrderedDict((rp, out[rp]) for rp in best_of.values())


CONTRAST_T = 25      # niveaux de gris : trait vs papier voisin
SIDE_PX = 6          # distance (px) du papier témoin, de part et d'autre du tracé


def line_contrast(dark, dark_on, inv, boundary, res, H, W):
    """Part des points du contour qui tombent sur un trait : plus sombres (max
    sur +-INK_NEAR px) que le plus clair des deux côtés à SIDE_PX px."""
    import shapely
    parts = getattr(boundary, "geoms", [boundary])
    on_all, ok_all = [], []
    for part in parts:
        L = part.length
        n = int(L / res)
        if n < 4:
            continue
        dist = np.linspace(res, L - res, n)
        p = shapely.get_coordinates(shapely.line_interpolate_point(part, dist))
        a = shapely.get_coordinates(shapely.line_interpolate_point(part, dist - res))
        b = shapely.get_coordinates(shapely.line_interpolate_point(part, dist + res))
        t = b - a
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        nrm = np.column_stack([-t[:, 1], t[:, 0]]) * SIDE_PX * res

        def px(xy):
            c = (inv.a * xy[:, 0] + inv.b * xy[:, 1] + inv.c).astype(np.int64)
            r = (inv.d * xy[:, 0] + inv.e * xy[:, 1] + inv.f).astype(np.int64)
            return r, c, (r >= 0) & (r < H) & (c >= 0) & (c < W)

        r0, c0, k0 = px(p)
        r1, c1, k1 = px(p + nrm)
        r2, c2, k2 = px(p - nrm)
        k = k0 & k1 & k2
        if not k.any():
            continue
        on = dark_on[r0[k], c0[k]].astype(np.int16)
        side = np.minimum(dark[r1[k], c1[k]], dark[r2[k], c2[k]]).astype(np.int16)
        ok_all.append(dark[r0[k], c0[k]] > 0)  # sur le scan (pas hors emprise)
        on_all.append((on - side) > CONTRAST_T)
    if not on_all:
        return None
    on, ok = np.concatenate(on_all), np.concatenate(ok_all)
    if ok.sum() < MIN_SAMPLES:
        return None
    return float(on[ok].mean())


def truth(raster, meta, con):
    """Parcelles validées de la feuille, dans le SRS du raster, acceptées sur l'encre."""
    with rasterio.open(raster) as src:
        crs, T, H, W = src.crs, src.transform, src.height, src.width
        res = abs(T.a)
        rgb = src.read([1, 2, 3])
        valid = src.dataset_mask() > 0
    # Assombrissement en uint8 ; le trait se juge par CONTRASTE LOCAL (plus sombre
    # que le papier de part et d'autre), insensible aux taches et au jaunissement
    # -- une normalisation globale laissait passer des parcelles dessinées sur une
    # autre feuille (Willerval A1 : le village est un développement séparé).
    dark = (765 - rgb.sum(axis=0, dtype=np.uint16)) // 3
    dark = dark.astype(np.uint8)
    del rgb
    dark[~valid] = 0
    dark_on = ndimage.maximum_filter(dark, size=2 * INK_NEAR + 1)

    feuilles = []
    for fid in meta["feuilles"]:
        row = con.execute("select geom, date_start from feuilles where id=?", (fid,)).fetchone()
        if row:
            feuilles.append((gpkg_geom(row[0]), year_of(row[1])))
    if not feuilles and meta.get("zone_mode"):
        # mode « zone » de calibrate.py : parcelles de l'année du scan dans son emprise
        from rasterio.features import shapes
        from shapely.ops import unary_union
        step = 8
        small = valid[::step, ::step].astype(np.uint8)
        polys = [shape(g) for g, v in shapes(small, transform=T * T.scale(step), mask=small > 0) if v]
        if polys:
            fp = max(polys, key=lambda g: g.area).buffer(-2 * step * res)
            fp = shape(transform_geom(crs, "EPSG:3857", mapping(fp)))
            feuilles = [(fp, meta.get("zone_year"))]
    if not feuilles:
        return None

    parcels = {}
    for fgeom, fyear in feuilles:
        x0, y0, x1, y1 = fgeom.bounds
        q = ("select p.id, p.geom, p.type, p.date_start from parcelles p join rtree_parcelles_geom r "
             "on p.fid = r.id where r.maxx >= ? and r.minx <= ? and r.maxy >= ? and r.miny <= ?")
        for pid, blob, typ, ds in con.execute(q, (x0, x1, y0, y1)):
            y = year_of(ds)
            if fyear and y and abs(y - fyear) > 1:
                continue
            g = gpkg_geom(blob)
            if g.is_empty or not g.point_on_surface().within(fgeom):
                continue
            parcels.setdefault(pid, (g, typ))

    dxm, dym = meta["dx"] * res, -meta["dy"] * res
    inv = ~T
    feats = []
    for pid, (g, typ) in parcels.items():
        g = shape(transform_geom("EPSG:3857", crs, mapping(g)))
        g = translate(g, dxm, dym)
        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty or g.area <= 0:
            continue
        on = line_contrast(dark, dark_on, inv, g.boundary, res, H, W)
        if on is None:
            continue
        if on >= ACCEPT:
            feats.append({"type": "Feature", "properties": {"id": pid, "type": typ, "ink": round(float(on), 3)},
                          "geometry": mapping(g)})
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    con = sqlite3.connect(GPKG)
    todo = [(rp, m) for rp, m in sheets().items() if not a.split or m["split"] == a.split]
    if a.limit:
        todo = todo[: a.limit]
    index = []
    for k, (rp, m) in enumerate(todo):
        d = out / slug(m["group"])
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        mask, prob = d / "mask.tif", d / "prob.tif"
        if not mask.exists():
            tmp = d / "prob_f32.tif"
            seg = inference.segment_geotiff(rp, str(mask), str(MODEL),
                                            params=inference.InferenceParams(threshold=0.30),
                                            prob_output_path=str(tmp))
            with rasterio.open(tmp) as src:
                p = src.read(1)
                prof = src.profile
            prof.update(dtype="uint8", nodata=None, compress="deflate", predictor=2)
            with rasterio.open(prob, "w", **prof) as dst:
                dst.write(np.clip(np.round(p * 255), 0, 255).astype(np.uint8), 1)
            tmp.unlink()
            (d / "seg.json").write_text(json.dumps(seg, indent=1))
        t1 = time.time()
        gt = d / "gt.geojson"
        if not gt.exists():
            feats = truth(rp, m, con)
            gt.write_text(json.dumps({"type": "FeatureCollection", "features": feats or []}))
        t2 = time.time()
        gr = d / "grass.geojson"
        if not gr.exists():
            feats, rep = grass_vectorize.grass_vectorize_mask(str(mask), target_srid=2154)
            gr.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
            (d / "grass_report.json").write_text(json.dumps(rep, indent=1, default=str))
        n_gt = len(json.loads(gt.read_text())["features"])
        n_gr = len(json.loads(gr.read_text())["features"])
        info = dict(dir=d.name, raster=rp, group=m["group"], split=m["split"], n_gt=n_gt, n_grass=n_gr)
        index.append(info)
        print(f"[{k + 1}/{len(todo)}] {m['split']:5} {m['group'][:45]:45} gt={n_gt:5} grass={n_gr:5} "
              f"seg {t1 - t0:4.0f}s gt {t2 - t1:4.0f}s grass {time.time() - t2:4.0f}s", flush=True)
    (out / f"index{'_' + a.split if a.split else ''}.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
