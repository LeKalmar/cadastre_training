"""Retrouve le plan géoréférencé NON découpé dont chaque DCP est tiré.

Un DCP est un découpage (alpha) d'un GeoTIFF « GEOREF » : même taille de pixel et
même grille. Pour chaque DCP :
  1. candidats = TIF (hors DCP) de même taille de pixel dont l'emprise contient le DCP ;
  2. preuve par les pixels : on tire des points dans la partie opaque du DCP et on compare
     leurs couleurs à celles du candidat au même endroit L93 (écart moyen en niveaux) ;
  3. on garde le candidat le plus proche ; à égalité, le plus grand (le moins découpé).

    .venv/bin/python tools/zone/sources_dcp.py            # -> data/zones/amorce_dcp/sources.csv
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from amorcer_dcp import BASE, OUT  # noqa: E402


def index(racines):
    import rasterio
    out = []
    for r in racines:
        for p in glob.glob(os.path.join(r, "**", "*.tif"), recursive=True):
            try:
                with rasterio.open(p) as s:
                    t = s.transform
                    out.append({"path": p, "res": abs(t.a), "rot": abs(t.b) + abs(t.d),
                                "b": tuple(s.bounds), "W": s.width, "H": s.height, "n": s.count,
                                "dcp": os.path.basename(p).upper().startswith("DCP")})
            except Exception as e:
                print(f"illisible : {p} ({e})", flush=True)
    return out


def contient(b, c, tol):
    return b[0] - tol <= c[0] and b[1] - tol <= c[1] and b[2] + tol >= c[2] and b[3] + tol >= c[3]


def echantillon(dcp, n=3000, seed=0):
    """n points (x, y L93) dans la partie opaque du DCP + leur couleur RGB."""
    import rasterio
    from rasterio.enums import Resampling
    rng = np.random.default_rng(seed)
    with rasterio.open(dcp) as s:
        k = max(1, max(s.width, s.height) // 2000)
        h, w = s.height // k, s.width // k
        a = s.read(4, out_shape=(h, w), resampling=Resampling.nearest) if s.count >= 4 \
            else s.dataset_mask(out_shape=(h, w))
        rr, cc = np.nonzero(a > 250)
        if len(rr) == 0:
            return None
        i = rng.choice(len(rr), min(n, len(rr)), replace=False)
        rows = np.clip(rr[i] * k + k // 2, 0, s.height - 1)
        cols = np.clip(cc[i] * k + k // 2, 0, s.width - 1)
        xs, ys = rasterio.transform.xy(s.transform, rows, cols)
        rgb = np.array(list(s.sample(zip(xs, ys), indexes=[1, 2, 3])), dtype=np.float32)
    return np.array(xs), np.array(ys), rgb


def ecart(cand, xs, ys, rgb):
    import rasterio
    with rasterio.open(cand) as s:
        idx = [1, 2, 3] if s.count >= 3 else [1, 1, 1]
        v = np.array(list(s.sample(zip(xs, ys), indexes=idx)), dtype=np.float32)
    ok = v.sum(1) > 0  # hors nodata du candidat
    if ok.mean() < 0.9:
        return None
    return float(np.abs(v[ok] - rgb[ok]).mean())


def main():
    P = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    P.add_argument("racines", nargs="*", default=[BASE])
    P.add_argument("--out", default=os.path.join(OUT, "sources.csv"))
    P.add_argument("--ecart-max", type=float, default=12.0, help="écart RGB moyen max (niveaux)")
    a = P.parse_args()

    tifs = index(a.racines)
    dcps = [t for t in tifs if t["dcp"]]
    autres = [t for t in tifs if not t["dcp"]]
    print(f"{len(dcps)} DCP, {len(autres)} autres TIF", flush=True)
    lignes = []
    for d in sorted(dcps, key=lambda t: t["path"]):
        nom = os.path.basename(d["path"])
        cands = [t for t in autres if abs(t["res"] - d["res"]) / d["res"] < 0.02
                 and contient(t["b"], d["b"], 5 * d["res"])]
        if not cands:  # découpe rééchantillonnée : on élargit à toute taille de pixel
            cands = [t for t in autres if contient(t["b"], d["b"], 20.0)]
        smp = echantillon(d["path"])
        res = []
        for c in cands:
            e = ecart(c["path"], *smp) if smp else None
            if e is not None:
                res.append((e, -c["W"] * c["H"], c))
        res.sort(key=lambda r: (round(r[0], 0), r[1]))
        best = res[0] if res and res[0][0] <= a.ecart_max else None
        lignes.append({"dcp": d["path"], "source": best[2]["path"] if best else "",
                       "ecart": round(best[0], 2) if best else (round(res[0][0], 2) if res else ""),
                       "meme_res": bool(best and abs(best[2]["res"] - d["res"]) / d["res"] < 0.02),
                       "n_candidats": len(cands),
                       "autres": " | ".join(f"{os.path.basename(r[2]['path'])}={r[0]:.1f}" for r in res[1:4])})
        src = os.path.basename(best[2]["path"]) if best else "— aucune source"
        print(f"{nom[:40]:40} -> {src[:70]:70} écart {lignes[-1]['ecart']}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)
    print(f"\n{sum(bool(l['source']) for l in lignes)}/{len(lignes)} sources trouvées -> {a.out}")


if __name__ == "__main__":
    main()
