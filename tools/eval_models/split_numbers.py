#!/usr/bin/env python3
"""Découpe des faces fusionnées grâce aux numéros de parcelle (sortie « texte » du modèle B).

Une face qui contient >= 2 groupes de texte est découpée par watershed sur la carte
de probabilité des traits (graines = groupes de texte). Une coupe n'est gardée que si
la frontière entre deux morceaux suit une crête (p moyen >= ``ridge``) : sinon les
morceaux sont refondus. Les faces sans numéro ne sont jamais touchées ni supprimées.

    .venv/bin/python tools/eval_models/split_numbers.py --model v3_fin --variant gaps_link \
        --text-model v3_fin_texte --ridge 0.15
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features as rfeat
from scipy import ndimage
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "refiner"))
import run_eval as RE  # noqa: E402


def text_groups(text, res, thr=0.5, join_m=0.8, min_m2=0.3, max_m2=40.0):
    """Étiquette des groupes de texte (chiffres d'un même numéro réunis)."""
    t = text > thr
    t = ndimage.binary_opening(t, iterations=1)
    j = max(1, int(round(join_m / res)))
    lab, n = ndimage.label(ndimage.binary_dilation(t, iterations=j))
    lab[~t] = 0
    area = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1)) * res * res
    bad = np.flatnonzero((area < min_m2) | (area > max_m2)) + 1
    lab[np.isin(lab, bad)] = 0
    return lab


def split_face(poly, prob, groups, transform, res, ridge, min_piece_m2):
    from skimage.segmentation import watershed

    inv = ~transform
    x0, y0, x1, y1 = poly.bounds
    c0, r0 = inv * (x0, y1)
    c1, r1 = inv * (x1, y0)
    pad = 3
    r0, c0 = max(0, int(r0) - pad), max(0, int(c0) - pad)
    r1, c1 = min(prob.shape[0], int(r1) + pad + 1), min(prob.shape[1], int(c1) + pad + 1)
    if r1 - r0 < 4 or c1 - c0 < 4:
        return None
    wt = transform * rasterio.Affine.translation(c0, r0)
    inside = rfeat.rasterize([(poly, 1)], out_shape=(r1 - r0, c1 - c0), transform=wt,
                             all_touched=True).astype(bool)
    g = groups[r0:r1, c0:c1] * inside
    ids = [i for i in np.unique(g) if i]
    if len(ids) < 2:
        return None
    markers = np.zeros(g.shape, np.int32)
    for k, i in enumerate(ids, 1):
        markers[g == i] = k
    p = prob[r0:r1, c0:c1]
    lab = watershed(p, markers, mask=inside)
    # force moyenne de la frontière entre chaque paire
    K = len(ids)
    s = np.zeros((K + 1, K + 1)); n = np.zeros((K + 1, K + 1))
    for a, b, pa, pb in ((lab[:, :-1], lab[:, 1:], p[:, :-1], p[:, 1:]),
                         (lab[:-1, :], lab[1:, :], p[:-1, :], p[1:, :])):
        m = (a != b) & (a > 0) & (b > 0)
        np.add.at(s, (a[m], b[m]), np.maximum(pa[m], pb[m]))
        np.add.at(n, (a[m], b[m]), 1)
    s = s + s.T; n = n + n.T
    parent = list(range(K + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a in range(1, K + 1):
        for b in range(a + 1, K + 1):
            if n[a, b] > 0 and s[a, b] / n[a, b] < ridge:
                parent[find(a)] = find(b)
    roots = np.array([find(i) for i in range(K + 1)])
    lab = roots[lab] * (lab > 0)
    if len(np.unique(lab[lab > 0])) < 2:
        return None
    pieces = []
    for geom, v in rfeat.shapes(lab.astype(np.int32), mask=lab > 0, transform=wt):
        pieces.append((v, shape(geom)))
    by = {}
    for v, gm in pieces:
        by.setdefault(v, []).append(gm)
    parts = [unary_union(v).buffer(0).intersection(poly) for v in by.values()]
    parts = [q for q in parts if not q.is_empty and q.area >= min_piece_m2]
    if len(parts) < 2:
        return None
    rest = poly.difference(unary_union(parts))
    if not rest.is_empty:  # bords non couverts -> au morceau voisin le plus long contact
        for r in (rest.geoms if hasattr(rest, "geoms") else [rest]):
            if r.area <= 0:
                continue
            k = max(range(len(parts)), key=lambda i: parts[i].intersection(r.buffer(res)).area)
            parts[k] = parts[k].union(r)
    out = []
    for q in parts:
        for x in (q.geoms if hasattr(q, "geoms") else [q]):
            if x.geom_type == "Polygon" and x.area > 0:
                out.append(x)
    return out if len(out) >= 2 else None


def run(job):
    model, variant, tmodel, s, ridge, tag = job
    from features import load_polys
    from georeferencing import polygon_refiner as pr
    from train_eval import score

    out = RE.EVAL / model / s
    rp = out / f"{tag}.score.json"
    if rp.exists():
        return json.loads(rp.read_text())
    src_geo = out / f"{variant}.geojson"
    tpath = RE.EVAL / tmodel / s / "text.tif"
    if not src_geo.exists() or not tpath.exists():
        return None
    with rasterio.open(RE.sheet_dir(model, s) / "prob.tif") as src:
        prob = src.read(1).astype(np.float32) / 255
        tr = src.transform
        res = abs(tr.a)
    with rasterio.open(tpath) as src:
        text = src.read(1).astype(np.float32) / 255
    groups = text_groups(text, res)
    polys, _ = load_polys(src_geo)
    new, nsplit = [], 0
    for q in polys:
        sp = split_face(q, prob, groups, tr, res, ridge, min_piece_m2=4.0)
        if sp:
            new.extend(sp); nsplit += 1
        else:
            new.append(q)
    (out / f"{tag}.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": mapping(g)} for g in new]}))
    gt, _ = load_polys(RE.REF / s / "gt.geojson")
    truth = pr.Truth.from_parcels(gt)
    r = dict(model=model, sheet=s, variant=tag, **score(new, truth), gaps={"split_faces": nsplit})
    r["simp10"] = score(pr.simplify_coverage(new, 10 * res), truth)
    rp.write_text(json.dumps(r, default=float))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v3_fin")
    ap.add_argument("--variant", default="gaps_link")
    ap.add_argument("--text-model", default="v3_fin_texte")
    ap.add_argument("--ridge", type=float, nargs="+", default=[0.15])
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    jobs = []
    for ridge in a.ridge:
        tag = f"{a.variant}_num{int(ridge * 100)}"
        jobs += [(a.model, a.variant, a.text_model, s, ridge, tag) for _, s in RE.val_sheets()]
    with Pool(a.jobs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(run, jobs):
            if r:
                print(f"{r['variant']:18} {r['sheet'][:34]:34} matched={r['matched']} under={r['under_seg']} "
                      f"over={r['over_seg']} split={r['gaps']['split_faces']}", flush=True)


if __name__ == "__main__":
    main()
