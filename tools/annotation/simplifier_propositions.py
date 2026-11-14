#!/usr/bin/env python3
"""Simplifie SUR PLACE les propositions IA encore « à valider » et jamais retouchées.

Douglas-Peucker par arc (chaque limite commune simplifiée une seule fois, nœuds
fixes : voir georeferencing.polygonize._simplify_coverage_arcs). Les parcelles
retouchées, validées, corrigées ou rejetées ne bougent pas et figent leurs
sommets, donc leurs voisines restent collées à elles. Les identifiants sont
conservés (une session d'édition QGIS ouverte reste valable) ; geom et geom_ia
sont mis à jour ensemble, la proposition reste « non modifiée ».

    .venv/bin/python tools/annotation/simplifier_propositions.py --feuilles 'Douai 1908%' [--tol 0.3] [--essai]
"""
import argparse
import sys
from pathlib import Path

import shapely
from shapely import wkb
from shapely.geometry import MultiPolygon

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402
from georeferencing.polygonize import _simplify_coverage_arcs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feuilles", required=True, help="motif SQL LIKE sur le nom de feuille")
    ap.add_argument("--tol", type=float, default=0.3, help="tolérance (m)")
    ap.add_argument("--tol-px", type=float, default=0.0,
                    help="tolérance en pixels du scan de chaque feuille ; > 0 remplace --tol")
    ap.add_argument("--essai", action="store_true", help="calcule sans rien écrire")
    a = ap.parse_args()
    db = C.connect()
    cur = db.cursor()
    feuilles = cur.execute("select id, nom, raster from feuilles where nom like %s order by nom",
                           (a.feuilles,)).fetchall()
    tot0 = tot1 = nmaj = 0
    for fid, nom, raster in feuilles:
        tol = a.tol
        if a.tol_px:
            import rasterio
            try:
                with rasterio.open(raster) as src:
                    tol = a.tol_px * abs(src.transform.a)
            except Exception as e:
                print(f"{nom:32} scan illisible ({e}), feuille sautée", flush=True)
                continue
        rows = cur.execute(
            "select id, review_status = 'pending' and not geom_modifiee, ST_AsBinary(geom) "
            "from parcelles_pending where feuille_id=%s "
            "union all select null, false, ST_AsBinary(geom) from parcelles "
            "where feuille_id=%s and pending_id is null", (fid, fid)).fetchall()
        parts, owner, fixed = [], [], []
        for k, (pid, libre, g) in enumerate(rows):
            for p in getattr(wkb.loads(g), "geoms", [wkb.loads(g)]):
                if not libre:
                    fixed.append(len(parts))
                parts.append(p)
                owner.append(k)
        if not parts:
            continue
        out = _simplify_coverage_arcs(parts, tol, fixed=fixed)
        by = {}
        for k, p0, p1 in zip(owner, parts, out):
            by.setdefault(k, []).append((p0, p1))
        maj = []
        n0 = n1 = 0
        for k, pp in by.items():
            pid, libre, _ = rows[k]
            if not libre:
                continue
            c0 = sum(shapely.get_num_coordinates(p) for p, _ in pp)
            c1 = sum(shapely.get_num_coordinates(q) for _, q in pp)
            n0 += c0
            n1 += c1
            if c1 < c0:
                maj.append((MultiPolygon([q for _, q in pp]).wkb, pid))
        tot0 += n0
        tot1 += n1
        nmaj += len(maj)
        print(f"{nom:32} {len(maj):5} propositions simplifiées, sommets {n0} -> {n1} "
              f"({100 * (1 - n1 / max(1, n0)):.0f} % en moins)", flush=True)
        if a.essai or not maj:
            continue
        # le déclencheur interdit de toucher geom_ia : on le suspend le temps de la transaction
        cur.execute("alter table parcelles_pending disable trigger pending_avant")
        cur.executemany(
            "update parcelles_pending set geom = g.n, geom_ia = g.n, aire_m2 = ST_Area(g.n) "
            "from (select nettoyer_geom(ST_Multi(ST_GeomFromWKB(%s, 2154))) n) g "
            "where id = %s and review_status = 'pending' and not geom_modifiee", maj)
        cur.execute("alter table parcelles_pending enable trigger pending_avant")
        db.commit()
    print(f"TOTAL : {nmaj} propositions, sommets {tot0} -> {tot1} "
          f"({100 * (1 - tot1 / max(1, tot0)):.0f} % en moins){' [essai, rien écrit]' if a.essai else ''}")


if __name__ == "__main__":
    main()
