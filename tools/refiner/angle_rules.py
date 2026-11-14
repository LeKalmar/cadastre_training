#!/usr/bin/env python3
"""Règles de sommets « angle + déplacement + alternance » sur le réseau de limites.

Un sommet intérieur d'une limite (entre deux jonctions) est retiré quand :
  - plancher  : il est à moins de ``floor_px`` de la corde de ses voisins (toujours) ;
  - zigzag    : son virage est de sens opposé à celui d'un voisin (aller-retour)
                ET il est à moins de ``zig_px`` de la corde ;
  - bras courts : un de ses bras fait moins de ``short_px`` ET il est à moins de
                ``short_dev_px`` de la corde.
Les jonctions (degré ≠ 2) ne bougent jamais ; on retire d'abord le sommet le plus
proche de sa corde, puis on recalcule ses voisins. Un vrai angle faible a de longs
bras : son écart à la corde est grand, il est protégé.

    .venv/bin/python tools/refiner/angle_rules.py work_dirs/refiner
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union

sys.path.insert(0, str(Path(__file__).parent))
from train_eval import load, score  # noqa: E402


def _dev(p, v, n):
    """Distance de v à la corde p–n et virage signé (cross) en v."""
    d = n - p
    L = np.hypot(*d)
    a, b = v - p, n - v
    cross = a[0] * b[1] - a[1] * b[0]
    if L < 1e-9:
        return float(np.hypot(*(v - p))), cross
    return abs(d[0] * (v[1] - p[1]) - d[1] * (v[0] - p[0])) / L, cross


def clean_line(c, px, floor_px, zig_px, short_px, short_dev_px):
    pts = [np.asarray(x, float) for x in c]
    closed = len(pts) > 3 and np.allclose(pts[0], pts[-1])
    removed = 0
    while len(pts) > (4 if closed else 2):
        n = len(pts)
        idx = range(1, n - 1)
        dev, sgn, arm = {}, {}, {}
        for i in idx:
            d, cr = _dev(pts[i - 1], pts[i], pts[i + 1])
            dev[i], sgn[i] = d, np.sign(cr)
            arm[i] = min(np.hypot(*(pts[i] - pts[i - 1])), np.hypot(*(pts[i + 1] - pts[i])))
        best, best_d = None, None
        for i in idx:
            d = dev[i] / px
            alt = any(j in sgn and sgn[j] * sgn[i] < 0 for j in (i - 1, i + 1))
            ok = (d < floor_px
                  or (alt and d < zig_px)
                  or (arm[i] / px < short_px and d < short_dev_px))
            if ok and (best_d is None or d < best_d):
                best, best_d = i, d
        if best is None:
            break
        del pts[best]
        removed += 1
    return pts, removed


def absorb_slivers(faces, max_area=15.0, max_radius=1.1):
    """Fusionne les lamelles (petites ET fines) dans le voisin au plus long bord commun."""
    faces = [f for f in faces if not f.is_empty]
    alive = dict(enumerate(faces))
    tree = shapely.STRtree(faces)

    def thin(g):
        if g.area >= max_area:
            return False
        try:
            r = shapely.maximum_inscribed_circle(g, 0.02).length
        except Exception:
            return False
        return r < max_radius

    owner = {i: i for i in alive}

    def root(i):
        while owner[i] != i:
            i = owner[i]
        return i

    merged = 0
    for i in sorted(alive, key=lambda k: faces[k].area):
        if root(i) != i or not thin(alive[i]):
            continue
        g = alive[i]
        best, best_len = None, 0.0
        for j in tree.query(g.buffer(0.05)):
            j = root(int(j))
            if j == i or j not in alive:
                continue
            L = g.boundary.intersection(alive[j].boundary).length
            if L > best_len:
                best, best_len = j, L
        if best is None or best_len <= 0:
            continue
        u = unary_union([alive[best], g])
        if u.geom_type != "Polygon":
            u = u.buffer(0)
        if u.geom_type != "Polygon":
            continue
        alive[best] = u
        del alive[i]
        owner[i] = best
        merged += 1
    return list(alive.values()), merged


def apply_rules(faces, px, floor_px=1.0, zig_px=6.0, short_px=12.0, short_dev_px=4.0, min_area=5.0,
                absorb=None):
    if absorb:
        faces, _ = absorb_slivers(faces, **absorb)
    net = unary_union([f.boundary for f in faces])
    merged = linemerge(net)
    lines = list(getattr(merged, "geoms", [merged]))
    out, rem = [], 0
    for ln in lines:
        pts, r = clean_line(list(ln.coords), px, floor_px, zig_px, short_px, short_dev_px)
        rem += r
        if len(pts) >= 2:
            out.append(LineString(pts))
    polys = list(shapely.get_parts(shapely.polygonize(list(shapely.get_parts(unary_union(out))))))
    cover = unary_union(faces)
    shapely.prepare(cover)
    keep = [p for p in polys if p.area >= min_area and cover.contains(p.representative_point())]
    return keep, rem


VARIANTS = {
    "plancher1": dict(floor_px=1.0, zig_px=0, short_px=0, short_dev_px=0),
    "zig4": dict(floor_px=1.0, zig_px=4.0, short_px=0, short_dev_px=0),
    "zig6": dict(floor_px=1.0, zig_px=6.0, short_px=0, short_dev_px=0),
    "zig8": dict(floor_px=1.0, zig_px=8.0, short_px=0, short_dev_px=0),
    "zig6+court12/3": dict(floor_px=1.0, zig_px=6.0, short_px=12.0, short_dev_px=3.0),
    "zig6+court16/4": dict(floor_px=1.0, zig_px=6.0, short_px=16.0, short_dev_px=4.0),
    "zig8+court16/4": dict(floor_px=1.0, zig_px=8.0, short_px=16.0, short_dev_px=4.0),
    "plancher2+zig6+court16/4": dict(floor_px=2.0, zig_px=6.0, short_px=16.0, short_dev_px=4.0),
    "lamelles": dict(floor_px=0, zig_px=0, short_px=0, short_dev_px=0,
                     absorb=dict(max_area=15.0, max_radius=1.1)),
    "lamelles+zig6+court16/4": dict(floor_px=1.0, zig_px=6.0, short_px=16.0, short_dev_px=4.0,
                                    absorb=dict(max_area=15.0, max_radius=1.1)),
    "lamelles30+zig8+court16/4": dict(floor_px=1.0, zig_px=8.0, short_px=16.0, short_dev_px=4.0,
                                      absorb=dict(max_area=30.0, max_radius=1.1)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--only", default="", help="sous-chaîne des variantes à lancer")
    ap.add_argument("--out", default="angle_results.json")
    a = ap.parse_args()
    val = [r for r in load(a.root) if r["split"] == "val"]
    from georeferencing import polygon_refiner as pr

    results = []
    for r in val:
        t, px = r["truth"], r["res"]
        results.append(dict(sheet=r["sheet"], variant="grass", **score(r["faces"], t)))
        for k in (10, 12):
            results.append(dict(sheet=r["sheet"], variant=f"grass+simp{k}",
                                **score(pr.simplify_coverage(r["faces"], k * px), t)))
        for name, kw in VARIANTS.items():
            if a.only and a.only not in name:
                continue
            polys, _ = apply_rules(r["faces"], px, **kw)
            results.append(dict(sheet=r["sheet"], variant=name, **score(polys, t)))
        print(f"  {r['sheet'][:40]:40} fait", flush=True)

    Path(a.root, a.out).write_text(json.dumps(results, indent=1, default=float))
    keys = ["n_pred", "matched", "missed", "spurious", "over_seg", "under_seg", "vertices"]
    base_v = sum(x["vertices"] for x in results if x["variant"] == "grass")
    print(f"\n{'variante':26} " + " ".join(f"{k:>9}" for k in keys) + "   bF1@.3  bF1@.6  sommets  faux")
    for v in dict.fromkeys(x["variant"] for x in results):
        rows = [x for x in results if x["variant"] == v]
        tot = {k: sum(x[k] for x in rows) for k in keys}
        f3 = np.mean([x["bF1@0.3"] for x in rows]); f6 = np.mean([x["bF1@0.6"] for x in rows])
        fake = np.average([x["fake_v"] for x in rows], weights=[x["vertices"] or 1 for x in rows])
        print(f"{v:26} " + " ".join(f"{tot[k]:9d}" for k in keys)
              + f"   {f3:.4f}  {f6:.4f}  {100 * tot['vertices'] / base_v:5.1f}%  {100 * fake:4.1f}%")


if __name__ == "__main__":
    main()


def merge_thin_runs(faces, max_radius=1.6, max_shared=2.5):
    """Recolle entre eux les tronçons FINS d'une même bande (traits doubles) coupés
    par des « barreaux » : deux faces fines voisines par un bord COURT (< max_shared m).
    Deux bandes parallèles partagent un long bord : elles restent séparées, et une
    bande n'est jamais fondue dans une parcelle large."""
    faces = [f for f in faces if not f.is_empty]
    rad = [shapely.maximum_inscribed_circle(f, 0.02).length for f in faces]
    tree = shapely.STRtree(faces)
    parent = list(range(len(faces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    joins = 0
    for i, f in enumerate(faces):
        if rad[i] >= max_radius:
            continue
        for j in tree.query(f.buffer(0.05)):
            j = int(j)
            if j <= i or rad[j] >= max_radius:
                continue
            L = f.boundary.intersection(faces[j].boundary).length
            if 0 < L < max_shared and find(i) != find(j):
                parent[find(j)] = find(i)
                joins += 1
    groups = {}
    for i in range(len(faces)):
        groups.setdefault(find(i), []).append(faces[i])
    out = []
    for g in groups.values():
        u = g[0] if len(g) == 1 else unary_union(g)
        out.extend(shapely.get_parts(u if u.is_valid else u.buffer(0)))
    return out, joins
