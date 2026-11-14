#!/usr/bin/env python3
"""Caractéristiques + étiquettes du correcteur, une feuille à la fois (cache .pkl).

    .venv/bin/python tools/refiner/features.py work_dirs/refiner [--jobs 6]
"""
import argparse
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from shapely.geometry import shape

sys.path.insert(0, "/home/lekalmar/PyCharmMiscProject/Code/Geenfoncier/GeneFoncier/terrapartus")
from georeferencing import polygon_refiner as pr  # noqa: E402


def load_polys(path, key=None):
    fc = json.loads(Path(path).read_text())["features"]
    polys, props = [], []
    for f in fc:
        g = shape(f["geometry"])
        for part in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
            if not part.is_valid:
                part = part.buffer(0)
            if part.geom_type == "Polygon" and not part.is_empty:
                polys.append(part)
                props.append(f.get("properties") or {})
    return polys, props


def process(d):
    d = Path(d)
    out = d / "features.pkl"
    if out.exists():
        return d.name, "cached"
    if not (d / "grass.geojson").exists() or not (d / "gt.geojson").exists():
        return d.name, "incomplete"
    t0 = time.time()
    gt_polys, _ = load_polys(d / "gt.geojson")
    if len(gt_polys) < 20:
        return d.name, f"gt too small ({len(gt_polys)})"
    faces, props = load_polys(d / "grass.geojson")
    prob = pr.ProbRaster.open(d / "prob.tif")
    truth = pr.Truth.from_parcels(gt_polys)
    cov = pr.build_coverage(faces, snap_tol=prob.res)
    origins = [p.get("origin") for p in props]
    rec = dict(
        sheet=d.name, res=prob.res,
        faces=cov.faces, pairs=cov.pairs,
        neighbours=cov.neighbours, shared_len=cov.shared_len, shared=cov.shared,
        Xe=pr.edge_features(cov, prob), Xf=pr.face_features(cov, prob, origins),
        ye=pr.edge_labels(cov, truth, tol=max(0.6, 4 * prob.res), step=prob.res),
        yf=pr.face_labels(cov, truth),
        truth=truth,
    )
    with open(out, "wb") as fh:
        pickle.dump(rec, fh)
    ye, yf = rec["ye"], rec["yf"]
    return d.name, (f"{len(faces)} faces, {len(cov.pairs)} edges | edges real/false/unk "
                    f"{(ye == 1).sum()}/{(ye == 0).sum()}/{(ye == -1).sum()} | faces parcel/noise/unk "
                    f"{(yf == 1).sum()}/{(yf == 0).sum()}/{(yf == -1).sum()} | {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    dirs = sorted(p for p in Path(a.root).iterdir() if p.is_dir() and (not a.only or a.only in p.name))
    with Pool(a.jobs) as pool:
        for name, msg in pool.imap_unordered(process, dirs):
            print(f"{name[:50]:50} {msg}", flush=True)


if __name__ == "__main__":
    main()
