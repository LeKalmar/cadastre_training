#!/usr/bin/env python3
"""
Étape 1 du pipeline : apparier chaque GeoTIFF à sa/ses feuille(s) cadastrale(s)
et mesurer, sur des fenêtres échantillonnées le long des limites de parcelles :
  - le décalage vecteur <-> encre (px) et sa dispersion (qualité du géoréf.)
  - l'épaisseur réelle du trait d'encre (px)
  - le taux de recouvrement encre (les lignes vecteur tombent-elles sur de l'encre ?)

Deux modes d'appariement :
  - "feuille" : une feuille de communes_feuilles (même année) visible à >= 60 % sur le scan
  - "scan"    : aucune feuille définie en base -> les parcelles visibles sur le scan
                (même année ou non datées) tiennent lieu de feuille

Sortie : CSV (une ligne par couple raster/feuille).

    python3 tools/calibrate.py --gpkg data/gis_export/genefoncier_prod.gpkg \
        --rasters "/chemin/Cadastre XIX/Cartes" --out data/gis_export/calibration.csv
"""
import argparse, csv, random, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr

sys.path.insert(0, str(Path(__file__).parent))
from carto import Sheet, srs_of, mem_layer, burn, transform, darkness, best_shift, shift, ink_profile, ink_width, dilate

WIN = 768
S3857 = srs_of(3857)


def year_of(s):
    return int(s[:4]) if s else None


def load_feuilles(gpkg):
    ds = ogr.Open(str(gpkg)); l = ds.GetLayerByName("feuilles")
    out = []
    for f in l:
        if not f.GetField("n_parcelles"): continue
        g = f.GetGeometryRef().Clone(); g.AssignSpatialReference(S3857)
        out.append(dict(id=f.GetField("id"), commune=f.GetField("commune"), section=f.GetField("section"),
                        feuille_no=f.GetField("feuille_no"), year=year_of(f.GetField("date_start")),
                        n_rural=f.GetField("n_rural"), couverture=f.GetField("couverture"), geom=g))
    return out


def parcels_in(pl, zone, year, need_within=True):
    """Parcelles dont le point intérieur est dans `zone` (3857), d'année compatible.
    year=None -> toutes ; sinon même année (±1) ou non datées."""
    pl.SetSpatialFilter(zone); res = []
    for f in pl:
        y = year_of(f.GetField("date_start"))
        if year and y and abs(y - year) > 1: continue
        g = f.GetGeometryRef()
        if not need_within or g.PointOnSurface().Within(zone):
            res.append((g.Clone(), f.GetField("type"), f.GetField("id")))
    pl.SetSpatialFilter(None)
    return res


def coverage(sheet, geom_r):
    """Part d'une géométrie (SRS du raster) qui tombe sur le contenu du scan."""
    _ds, lyr = mem_layer([geom_r], sheet.srs, ogr.wkbMultiPolygon)
    th, tw = sheet.content.shape; tg = sheet.thumb_gt()
    m = burn(lyr, tg, tw, th, sheet.srs)
    area = geom_r.GetArea()
    return float((m & sheet.content).sum() * abs(tg[1] * tg[5]) / area) if area > 0 else 0.0


def content_polygon(sheet):
    """Emprise du contenu du scan (polygone, 3857) à partir de la miniature."""
    th, tw = sheet.content.shape
    mem = gdal.GetDriverByName("MEM").Create("", tw, th, 1, gdal.GDT_Byte)
    mem.SetGeoTransform(sheet.thumb_gt()); mem.SetProjection(sheet.srs.ExportToWkt())
    mem.GetRasterBand(1).WriteArray(sheet.content.astype(np.uint8))
    vds = ogr.GetDriverByName("MEM").CreateDataSource(""); vl = vds.CreateLayer("c", sheet.srs, ogr.wkbPolygon)
    vl.CreateField(ogr.FieldDefn("v", ogr.OFTInteger))
    gdal.Polygonize(mem.GetRasterBand(1), mem.GetRasterBand(1), vl, 0)
    polys = [f.GetGeometryRef().Clone() for f in vl if f.GetField("v") == 1]
    if not polys: return None
    big = max(polys, key=lambda g: g.GetArea())
    big.AssignSpatialReference(sheet.srs)
    return transform(big, sheet.srs, S3857)


def measure(sheet, parcels_r, rng, n_win=16):
    """Fenêtres centrées sur des sommets de limites, dans le contenu du scan."""
    _ds, blyr = mem_layer([g.Boundary() for g, *_ in parcels_r], sheet.srs)
    gt = sheet.gt; th, tw = sheet.content.shape
    pts = []
    for g, *_ in parcels_r:
        b = g.Boundary()
        parts = [b.GetGeometryRef(i) for i in range(b.GetGeometryCount())] or [b]
        for ls in parts:
            for k in range(0, ls.GetPointCount(), 3):
                px, py = (ls.GetX(k) - gt[0]) / gt[1], (ls.GetY(k) - gt[3]) / gt[5]
                if WIN // 2 < px < sheet.W - WIN // 2 and WIN // 2 < py < sheet.H - WIN // 2 \
                        and sheet.content[int(py * th / sheet.H), int(px * tw / sheet.W)]:
                    pts.append((px, py))
    if not pts: return None
    rng.shuffle(pts)
    shifts, gains, widths, hits = [], [], [], []
    for px, py in pts[:n_win]:
        x0, y0 = int(px - WIN // 2), int(py - WIN // 2)
        rgb, valid = sheet.read(x0, y0, WIN, WIN)
        core = burn(blyr, sheet.win_gt(x0, y0), WIN, WIN, sheet.srs) & valid
        if core.sum() < 200: continue
        d = darkness(rgb); d[~valid] = 0
        dy, dx, gain, _ = best_shift(d, core, 15)
        cs = shift(core, dy, dx)
        prof = ink_profile(d, cs, 10)
        thr = np.percentile(d[valid], 90)  # "encre" = 10 % des pixels les plus sombres
        hits.append(float(dilate(d > thr, 1)[cs].mean()))
        shifts.append((dy, dx)); gains.append(gain); widths.append(ink_width(prof))
    if not shifts: return None
    s = np.array(shifts, float); med = np.median(s, 0)
    w = np.array(widths, float); fin = np.isfinite(w)
    return dict(n_win=len(shifts), dy=float(med[0]), dx=float(med[1]),
                spread=float(np.median(np.hypot(*(s - med).T))), gain=float(np.median(gains)),
                ink_w=float(np.median(w[fin])) if fin.any() else np.nan,
                ink_w_iqr=float(np.subtract(*np.percentile(w[fin], [75, 25]))) if fin.sum() > 1 else np.nan,
                hit=float(np.median(hits)))


# --- travail par raster (exécuté dans un process) -------------------------------------------
G = {}


def init(gpkg):
    G["feuilles"] = load_feuilles(gpkg)
    G["pds"] = ogr.Open(str(gpkg)); G["pl"] = G["pds"].GetLayerByName("parcelles")


def process(args):
    rp, min_overlap, max_res = args
    rows, log = [], []
    try: sh = Sheet(rp)
    except Exception as e: return [], [f"✗ {rp.name}: {e}"]
    if not sh.ds.RasterCount or not sh.ds.GetProjection(): return [], []
    res_m = sh.res * (1 if sh.srs.IsProjected() else 111000)
    if res_m > max_res: return [], []
    rng = random.Random(hash(rp.name) & 0xffff)
    g = sh.gt; x1, y1 = g[0] + sh.W * g[1], g[3] + sh.H * g[5]
    bbox = ogr.CreateGeometryFromWkt(f"POLYGON(({g[0]} {y1},{x1} {y1},{x1} {g[3]},{g[0]} {g[3]},{g[0]} {y1}))")
    bbox.AssignSpatialReference(sh.srs); bbox3857 = transform(bbox, sh.srs, S3857)
    base = dict(raster=str(rp), res_m=round(res_m, 4), raster_year=sh.year)

    def emit(mode, fid, commune, section, fno, year, parcels, n_rural, couv, ov):
        m = measure(sh, [(transform(pg, S3857, sh.srs), t, i) for pg, t, i in parcels], rng)
        if not m: return
        rows.append(dict(base, mode=mode, feuille_id=fid, commune=commune, section=section, feuille_no=fno,
                         year=year, n_parcelles=len(parcels), n_rural=n_rural, couverture=couv, overlap=round(ov, 3),
                         **{k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()}))
        log.append(f"{mode:7} {commune} {section or ''}{fno if fno is not None else ''} {year} <- {rp.name[:55]} | "
                   f"ov={ov:.2f} n={len(parcels)} shift=({m['dy']:+.0f},{m['dx']:+.0f}) spread={m['spread']:.1f} "
                   f"ink_w={m['ink_w']:.1f} hit={m['hit']:.2f}")

    matched = False
    for f in G["feuilles"]:
        if not f["geom"].Intersects(bbox3857): continue
        if sh.year and f["year"] and abs(sh.year - f["year"]) > 1: continue
        ov = coverage(sh, transform(f["geom"], S3857, sh.srs))
        if ov < min_overlap: continue
        parcels = parcels_in(G["pl"], f["geom"], f["year"])
        if len(parcels) < 10: continue
        matched = True
        emit("feuille", f["id"], f["commune"], f["section"], f["feuille_no"], f["year"], parcels,
             f["n_rural"], f["couverture"], ov)
    if not matched:
        zone = content_polygon(sh)
        if zone is not None:
            parcels = parcels_in(G["pl"], zone, sh.year)
            if len(parcels) >= 30:
                rural = sum(t in ("labour", "prairie", "bois", "jardin", "marais") for _, t, _ in parcels)
                emit("scan", None, rp.parent.name, None, None, sh.year, parcels, rural, None, 1.0)
    return rows, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpkg", required=True); ap.add_argument("--rasters", required=True, nargs="+", help="un ou plusieurs dossiers")
    ap.add_argument("--out", required=True); ap.add_argument("--min-overlap", type=float, default=0.6)
    ap.add_argument("--max-res", type=float, default=0.5, help="m/px ; au-delà = tableau d'assemblage, ignoré")
    ap.add_argument("--filter", default=None, help="sous-chaîne du chemin (debug)")
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    rasters = sorted({p.resolve() for root in a.rasters for p in Path(root).rglob("*")
                      if p.suffix.lower() in (".tif", ".tiff", ".jp2", ".gpkg")
                      and (not a.filter or a.filter in str(p))})
    print(f"{len(rasters)} rasters", flush=True)
    rows = []
    with Pool(a.jobs, initializer=init, initargs=(a.gpkg,)) as pool:
        for i, (r, log) in enumerate(pool.imap_unordered(process, [(p, a.min_overlap, a.max_res) for p in rasters])):
            rows += r
            for line in log: print(f"[{i+1}/{len(rasters)}] {line}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"✅ {len(rows)} couples raster/feuille -> {a.out}")


if __name__ == "__main__":
    main()
