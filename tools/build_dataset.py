#!/usr/bin/env python3
"""
Étape 2 du pipeline : génère le dataset de tuiles (image + masque) à partir des
couples raster/feuille validés par calibrate.py.

Masque (PNG 8 bits) :
    255 = ligne (cœur de LINE_W px, centré sur le tracé vecteur)
     50 = texte / symbole : fond à pénalité renforcée (le modèle ne doit pas y tracer)
    100 = ignoré (bande de 1 px autour des lignes, limites sans encre, parcelles
          rejetées, zones non annotées, hors scan)
      0 = fond
  -> avec l'ancien train.py (seuil 127), 100 est lu comme du fond : compatible.

Principes (cf. mesures de calibration) :
  - résolution NATIVE (pas de ré-échantillonnage) : le trait d'encre fait ~4 px
    sur tous les scans quelle que soit l'échelle -> épaisseur de masque en px constante
  - une parcelle n'est gardée sur un scan que si son contour tombe sur de l'encre
  - chaque parcelle n'est attribuée qu'à un seul scan (le mieux aligné) -> pas de doublons

    python3 tools/build_dataset.py --gpkg data/gis_export/genefoncier_prod.gpkg \
        --calib data/gis_export/calibration.csv --out data/cadastre_lines_v2
"""
import argparse, csv, hashlib, json, math, random, re, shutil, sys, unicodedata
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from carto import Sheet, mem_layer, burn, transform, darkness, dilate, erode, thick_line, classify_marks
from calibrate import S3857, load_feuilles, parcels_in, content_polygon

TILE = 512
STRIDE = 384
LINE_W = 3            # cœur de ligne (px) — trait d'encre mesuré ~4 px
IGNORE_BAND = 1       # px de part et d'autre du cœur
CROP_BAND = 12        # px ignorés le long du bord de découpe du scan (lignes coupées, faux bord)
INK_T = 20            # seuil d'encre (assombrissement / papier)
INK_NEAR = 2          # tolérance (px) entre tracé et encre
GAP_BRIDGE = 6        # comble les manques d'encre jusqu'à ~2*6 px le long d'une ligne
ACCEPT = 0.75         # part du contour d'une parcelle qui doit être sur l'encre
ZONE_CLOSE_M = 6.0    # fermeture de la zone annotée (m) : inclut les chemins entre parcelles
F = 8                 # facteur de la grille basse résolution (zone)
RURAL = {"labour", "prairie", "bois", "jardin", "marais", "pature", "verger", "vigne", "friche"}


def slug(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60]


# --------------------------------------------------------------------------------------------
G = {}


def init(gpkg):
    G["feuilles"] = {f["id"]: f for f in load_feuilles(gpkg)}
    G["pds"] = ogr.Open(str(gpkg)); G["pl"] = G["pds"].GetLayerByName("parcelles")


def candidates(raster, pairs):
    """Parcelles candidates d'un raster (union de ses couples feuille/scan), en SRS du raster."""
    sh = Sheet(raster); seen = {}
    for p in pairs:
        if p["mode"] == "feuille":
            f = G["feuilles"][p["feuille_id"]]
            ps = parcels_in(G["pl"], f["geom"], f["year"])
        else:
            zone = content_polygon(sh)
            ps = parcels_in(G["pl"], zone, sh.year) if zone is not None else []
        for g, t, i in ps:
            seen.setdefault(i, (transform(g, S3857, sh.srs), t))
    return sh, seen


def footprint(rp):
    sh = Sheet(rp); g = content_polygon(sh)
    return rp, (g.ExportToWkt() if g is not None else None), sh.year


def cluster(meta, fps):
    """Regroupe les copies d'une même feuille : même année et emprises quasi identiques
    (IoU > 0.4) ou l'une incluse dans l'autre à 80 % (recadrage) à résolution comparable."""
    rps = [rp for rp in meta if fps[rp][0]]
    geo = {rp: ogr.CreateGeometryFromWkt(fps[rp][0]) for rp in rps}
    parent = {rp: rp for rp in meta}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, a in enumerate(rps):
        for b in rps[i + 1:]:
            ya, yb = fps[a][1], fps[b][1]
            if ya and yb and abs(ya - yb) > 1: continue
            if not geo[a].Intersects(geo[b]): continue
            inter = geo[a].Intersection(geo[b]).GetArea()
            aa, ab = geo[a].GetArea(), geo[b].GetArea()
            same_res = max(meta[a]["res"], meta[b]["res"]) / min(meta[a]["res"], meta[b]["res"]) < 1.5
            if inter / (aa + ab - inter) > 0.4 or (same_res and inter / min(aa, ab) > 0.8):
                parent[find(a)] = find(b)
    groups = defaultdict(list)
    for rp in meta: groups[find(rp)].append(rp)
    return list(groups.values())


def offset_gt(sh, x0, y0, dy, dx):
    return sh.win_gt(x0 - dx, y0 - dy)


def phase_a(args):
    """Score d'encre par parcelle sur un raster."""
    raster, pairs, dy, dx = args
    sh, cand = candidates(raster, pairs)
    if not cand: return raster, {}
    ids = list(cand)
    _ds, lyr = mem_layer([], sh.srs)
    lyr.CreateField(ogr.FieldDefn("k", ogr.OFTInteger))
    for k, i in enumerate(ids):
        f = ogr.Feature(lyr.GetLayerDefn()); f.SetGeometry(cand[i][0].Boundary()); f.SetField("k", k + 1)
        lyr.CreateFeature(f)
    tot = np.zeros(len(ids) + 1); on = np.zeros(len(ids) + 1)
    B = 2048
    for y0 in range(0, sh.H, B):
        for x0 in range(0, sh.W, B):
            w, h = min(B, sh.W - x0), min(B, sh.H - y0)
            mem = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Int32)
            mem.SetGeoTransform(offset_gt(sh, x0, y0, dy, dx)); mem.SetProjection(sh.srs.ExportToWkt())
            gdal.RasterizeLayer(mem, [1], lyr, options=["ATTRIBUTE=k"])
            lab = mem.ReadAsArray()
            if not lab.any(): continue
            rgb, valid = sh.read(x0, y0, w, h)
            d = darkness(rgb)
            near = dilate(d > INK_T, INK_NEAR)
            m = (lab > 0) & valid
            tot += np.bincount(lab[m], minlength=len(tot))
            on += np.bincount(lab[m & near], minlength=len(tot))
    score = {ids[k - 1]: (on[k] / tot[k], int(tot[k])) for k in range(1, len(tot)) if tot[k] > 0}
    return raster, score


def phase_c(args):
    """Tuilage d'un raster."""
    raster, pairs, dy, dx, assigned, accepted, out_dir, sheet, seed, group = args
    sh, cand = candidates(raster, pairs)
    rng = random.Random(seed)
    acc = [cand[i] for i in accepted if i in cand]
    rej = [cand[i] for i in cand if i not in accepted]
    asg = [cand[i] for i in assigned if i in cand]
    if not asg: return []
    _a, lyr_acc = mem_layer([g.Boundary() for g, _ in acc], sh.srs)
    _r, lyr_rej = mem_layer([g.Boundary() for g, _ in rej], sh.srs)
    # zone annotée (grille basse résolution) : parcelles attribuées, fermeture morphologique
    lw, lh = math.ceil(sh.W / F), math.ceil(sh.H / F)
    lgt = offset_gt(sh, 0, 0, dy, dx); lgt = (lgt[0], lgt[1] * F, 0, lgt[3], 0, lgt[5] * F)
    _z, lyr_z = mem_layer([g for g, _ in asg], sh.srs, ogr.wkbMultiPolygon)
    _u, lyr_u = mem_layer([g for g, t in asg if t in RURAL], sh.srs, ogr.wkbMultiPolygon)
    zone = burn(lyr_z, lgt, lw, lh, sh.srs, all_touched=True)
    rc = max(1, round(ZONE_CLOSE_M / (sh.res * F)))
    zone = erode(dilate(zone, rc), rc)
    rural = burn(lyr_u, lgt, lw, lh, sh.srs, all_touched=True)
    rows = []
    name = f"{slug(sheet)}_{hashlib.md5(str(raster).encode()).hexdigest()[:6]}"  # unique par raster
    for y0 in range(0, sh.H - TILE + 1, STRIDE):
        for x0 in range(0, sh.W - TILE + 1, STRIDE):
            zt = zone[y0 // F:(y0 + TILE) // F, x0 // F:(x0 + TILE) // F]
            if zt.mean() < 0.5: continue
            rgb, valid = sh.read(x0, y0, TILE, TILE)
            if valid.mean() < 0.6: continue
            g_ = offset_gt(sh, x0, y0, dy, dx)
            core = burn(lyr_acc, g_, TILE, TILE, sh.srs)
            core_rej = burn(lyr_rej, g_, TILE, TILE, sh.srs)
            d = darkness(rgb)
            near = dilate(d > INK_T, INK_NEAR)
            pos = core & dilate(core & near, GAP_BRIDGE)
            unsure = core & ~pos
            line = thick_line(pos, LINE_W) & valid
            zfull = np.kron(zt, np.ones((F, F), bool))
            zfull = np.pad(zfull, ((0, TILE - zfull.shape[0]), (0, TILE - zfull.shape[1])), mode="edge")
            ign = (dilate(line, IGNORE_BAND) & ~line) | dilate(unsure, 2) | (dilate(core_rej, 2) & ~line) | ~zfull | ~valid
            # bord de découpe : la transition papier/vide et les lignes coupées ne doivent rien apprendre
            ign |= ~erode(valid, CROP_BAND)
            # traits non annotés droits (crayon, pointillés) : ignorés ; texte/symboles : négatifs renforcés
            un_lines, text = classify_marks(d, core | core_rej)
            ign |= un_lines & ~line
            text &= ~ign & ~dilate(line, 2)
            n_line = int(line.sum())
            if n_line < 150 and rng.random() > 0.05: continue  # garde ~5 % de tuiles quasi vides
            if ign.mean() > 0.5: continue
            mask = np.zeros((TILE, TILE), np.uint8); mask[text] = 50; mask[ign] = 100; mask[line] = 255
            tname = f"{name}_y{y0}_x{x0}"
            Image.fromarray(rgb.transpose(1, 2, 0)).save(out_dir / "images" / f"{tname}.jpg", quality=95)
            Image.fromarray(mask).save(out_dir / "labels" / f"{tname}.png", optimize=True)
            rt = rural[y0 // F:(y0 + TILE) // F, x0 // F:(x0 + TILE) // F]
            rows.append(dict(tile=tname, group=group, sheet=sheet, raster=str(raster), res_m=round(sh.res, 4),
                             line_frac=round(n_line / TILE ** 2, 4), ignore_frac=round(float(ign.mean()), 3),
                             text_frac=round(float(text.mean()), 4),
                             rural_frac=round(float(rt.mean()) if rt.size else 0, 3)))
    return rows


# --------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpkg", required=True); ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--min-hit", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--filter", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    rows = [r for r in csv.DictReader(open(a.calib)) if float(r["hit"] or 0) >= a.min_hit
            and (not a.filter or a.filter in r["raster"])]
    by_r = defaultdict(list)
    for r in rows: by_r[r["raster"]].append(r)
    # décalage du raster = celui du couple le mieux noté (en pratique 0 ou ±1 px)
    meta = {}
    for rp, ps in by_r.items():
        b = max(ps, key=lambda p: float(p["hit"]))
        meta[rp] = dict(pairs=ps, dy=round(float(b["dy"])), dx=round(float(b["dx"])), hit=float(b["hit"]),
                        res=float(b["res_m"]),
                        group=" ".join(x for x in [b["commune"] if b["mode"] == "feuille" else Path(rp).stem,
                                                    b["section"] or "", b["feuille_no"] or "", b["year"] or ""] if x))
    print(f"{len(meta)} rasters retenus (hit >= {a.min_hit})", flush=True)

    with Pool(a.jobs, initializer=init, initargs=(a.gpkg,)) as pool:
        # --- phase 0 : copies d'une même feuille (TIF / GPKG / COG...) -> on garde la meilleure
        fps = {rp: (w, y) for rp, w, y in pool.imap_unordered(footprint, list(meta))}
        # --- phase A : score d'encre par parcelle (sur toutes les copies)
        scores = {}
        for i, (rp, sc) in enumerate(pool.imap_unordered(
                phase_a, [(rp, m["pairs"], m["dy"], m["dx"]) for rp, m in meta.items()])):
            scores[rp] = sc
            ok = sum(v[0] >= ACCEPT and v[1] >= 30 for v in sc.values())
            print(f"  A [{i+1}/{len(meta)}] {Path(rp).name[:60]}: {ok}/{len(sc)} parcelles sur l'encre", flush=True)
        accepted = {rp: {i for i, (s, n) in scores[rp].items() if s >= ACCEPT and n >= 30} for rp in meta}
        # --- copies / recouvrements d'une même feuille (TIF, GPKG, COG, montages...) : même groupe
        #     pour le split train/val (anti-fuite). Le dédoublonnage des tuiles se fait en phase B.
        split_group = {}
        for k, grp_ in enumerate(cluster(meta, fps)):
            best = max(grp_, key=lambda rp: (len(accepted[rp]), -meta[rp]["res"]))
            for rp in grp_: split_group[rp] = meta[best]["group"] + f" #{k}"
        # --- phase B : attribution de chaque parcelle à un seul scan (celui qui en valide le plus)
        order = sorted(meta, key=lambda rp: (-len(accepted[rp]), meta[rp]["res"]))
        used, assigned = set(), {}
        for rp in order:
            assigned[rp] = accepted[rp] - used
            used |= assigned[rp]
        print(f"phase B : {len(used)} parcelles uniques attribuées", flush=True)
        # --- phase C : tuilage
        tmp = out / "_all"
        for sub in ("images", "labels"):
            (tmp / sub).mkdir(parents=True, exist_ok=True)
        jobs = [(rp, meta[rp]["pairs"], meta[rp]["dy"], meta[rp]["dx"], assigned[rp], accepted[rp], tmp,
                 meta[rp]["group"], i, split_group[rp]) for i, rp in enumerate(order) if len(assigned[rp]) >= 20]
        tiles = []
        for i, rws in enumerate(pool.imap_unordered(phase_c, jobs)):
            tiles += rws
            if rws: print(f"  C [{i+1}/{len(jobs)}] {rws[0]['sheet']}: {len(rws)} tuiles", flush=True)

    # --- split train/val PAR GROUPE (feuille), stratifié rural/urbain
    grp = defaultdict(list)
    for t in tiles: grp[t["group"]].append(t)
    stats = {g: np.mean([t["rural_frac"] for t in ts]) for g, ts in grp.items()}
    val = set()
    for is_rural in (True, False):
        gs = sorted((g for g in grp if (stats[g] >= 0.3) == is_rural),
                    key=lambda g: hashlib.md5(g.encode()).hexdigest())
        n_target = a.val_frac * sum(len(grp[g]) for g in gs); n = 0
        for g in gs:
            if n >= n_target: break
            val.add(g); n += len(grp[g])
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)
    for t in tiles:
        t["split"] = "val" if t["group"] in val else "train"
        shutil.move(tmp / "images" / f"{t['tile']}.jpg", out / "images" / t["split"] / f"{t['tile']}.jpg")
        shutil.move(tmp / "labels" / f"{t['tile']}.png", out / "labels" / t["split"] / f"{t['tile']}.png")
    shutil.rmtree(tmp, ignore_errors=True)
    with open(out / "tiles.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tiles[0].keys())); w.writeheader(); w.writerows(tiles)
    summary = dict(n_tiles=len(tiles), n_train=sum(t["split"] == "train" for t in tiles),
                   n_val=sum(t["split"] == "val" for t in tiles), n_groups=len(grp), val_groups=sorted(val),
                   n_parcelles=len(used), tile=TILE, line_w=LINE_W, ignore_band=IGNORE_BAND,
                   rural_tiles=sum(t["rural_frac"] >= 0.3 for t in tiles))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "val_groups"}, indent=2))


if __name__ == "__main__":
    main()
