#!/usr/bin/env python3
"""Remplit la base d'annotation à partir de l'export GénéFoncier existant.

Pour chaque feuille du jeu cadastre_lines_v2 (tiles.csv, un raster par feuille,
le mieux calé) :
  * crée la feuille : emprise (feuille GénéFoncier, ou emprise du papier pour les
    scans calés en mode « scan »), scan de référence, décalage, split actuel ;
  * rattache chaque parcelle validée à LA feuille où son contour est le mieux
    dessiné (test de contraste trait/papier). Une parcelle candidate de plusieurs
    feuilles va à la meilleure ; son lien vaut ok / a_verifier / hors_feuille ;
  * les parcelles au lien douteux restent statut « a_verifier » : elles ne
    serviront à l'entraînement qu'une fois vérifiées dans QGIS.
Enregistre aussi le modèle de la nuit comme modèle actif.

    .venv/bin/python tools/annotation/init_from_export.py [--limit N]
"""
import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_geom
from shapely.affinity import translate
from shapely.geometry import mapping, shape

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "refiner"))
import common as C  # noqa: E402
from prepare import GPKG, MODEL, gpkg_geom, sheets, year_of  # noqa: E402


def feuille_rows(rp, m, con):
    """[(id, nom, commune, section, feuille_no, annee, geom_3857, source)]."""
    out = []
    nom = re.sub(r"\s*#\d+$", "", m["group"]).strip()
    for fid in m["feuilles"]:
        r = con.execute("select geom, commune, section, feuille_no, date_start from feuilles "
                        "where id=?", (fid,)).fetchone()
        if r:
            out.append((fid, nom, r[1], r[2], r[3], year_of(r[4]), gpkg_geom(r[0]), "export"))
    if not out and m.get("zone_mode"):
        fp = C.paper_footprint(rp)
        if fp is not None:
            with rasterio.open(rp) as src:
                g3857 = shape(transform_geom(src.crs, "EPSG:3857", mapping(fp)))
            out.append((C.uuid5(rp), nom, None, None, None, m.get("zone_year"), g3857, "scan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    con = sqlite3.connect(GPKG)
    db = C.connect()
    cur = db.cursor()
    if cur.execute("select count(*) from feuilles").fetchone()[0]:
        sys.exit("La base contient déjà des feuilles : import initial refusé (rien n'est écrasé).")

    todo = list(sheets().items())
    if a.limit:
        todo = todo[: a.limit]
    best = {}      # parcelle id -> (score, feuille_id, geom 2154 recalée, attributs)
    n_feuilles = 0
    for k, (rp, m) in enumerate(todo):
        t0 = time.time()
        rows = feuille_rows(rp, m, con)
        if not rows:
            print(f"[{k + 1}/{len(todo)}] {m['group'][:50]:50} sans emprise, ignorée", flush=True)
            continue
        with rasterio.open(rp) as src:
            crs, T = src.crs, src.transform
            res = abs(T.a)
            rgb = src.read([1, 2, 3])
            valid = src.dataset_mask() > 0
        dark, dark_on = C.darkness(rgb)
        del rgb
        dxm, dym = m["dx"] * res, -m["dy"] * res
        n_cand = 0
        for fid, nom, commune, section, fno, annee, g3857, source in rows:
            g2154 = shape(transform_geom("EPSG:3857", "EPSG:2154", mapping(g3857)))
            cur.execute(
                "insert into feuilles (id, nom, commune, section, feuille_no, annee, raster, dx, dy, "
                "statut, split, source, geom) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,'a_verifier',%s,%s,"
                "ST_GeomFromWKB(%s, 2154)) on conflict (id) do nothing",
                (fid, nom, commune, section, None if fno is None else str(fno), annee, rp,
                 m["dx"], m["dy"], m["split"] if m["split"] in ("train", "val") else None,
                 source, C.wkb(g2154)))
            n_feuilles += 1
            x0, y0, x1, y1 = g3857.bounds
            q = ("select p.id, p.geom, p.type, p.type2, p.parcelles_no, p.parcelles_bis, "
                 "p.par_main_id, p.date_start, p.date_end from parcelles p join rtree_parcelles_geom r "
                 "on p.fid = r.id where r.maxx >= ? and r.minx <= ? and r.maxy >= ? and r.miny <= ?")
            for pid, blob, typ, typ2, no, bis, main_id, ds, de in con.execute(q, (x0, x1, y0, y1)):
                y = year_of(ds)
                if annee and y and abs(y - annee) > 1:
                    continue
                g = gpkg_geom(blob)
                if g.is_empty or not g.point_on_surface().within(g3857):
                    continue
                g = translate(shape(transform_geom("EPSG:3857", crs, mapping(g))), dxm, dym)
                if not g.is_valid:
                    g = g.buffer(0)
                if g.is_empty:
                    continue
                score = C.line_contrast(dark, dark_on, T, g.boundary, valid)
                n_cand += 1
                s = -1.0 if score is None else score
                if pid not in best or s > best[pid][0]:
                    best[pid] = (s, fid, g, dict(type=typ, type2=typ2, no=no, bis=bis,
                                                 main=main_id, ds=ds, de=de))
        db.commit()
        print(f"[{k + 1}/{len(todo)}] {m['group'][:50]:50} {n_cand:5} parcelles candidates "
              f"{time.time() - t0:4.0f}s", flush=True)

    counts = {"ok": 0, "a_verifier": 0, "hors_feuille": 0}
    for pid, (s, fid, g, at) in best.items():
        lien = C.lien_of(None if s < 0 else s)
        counts[lien] += 1
        if lien == "hors_feuille":
            # dessinée sur aucune feuille connue : la rattacher de force serait
            # précisément l'erreur à éviter. Elle reste dans l'export GénéFoncier.
            continue
        cur.execute(
            "insert into parcelles (id, feuille_id, parcelles_no, parcelles_bis, type, type2, "
            "par_main_id, date_start, date_end, origin, statut, lien, contraste, geom) values "
            "(%s,%s,%s,%s,%s,%s,%s,%s,%s,'export',%s,%s,%s,ST_GeomFromWKB(%s, 2154))",
            (pid, fid, None if at["no"] is None else str(at["no"]), at["bis"], at["type"],
             at["type2"], at["main"], at["ds"], at["de"],
             "valide" if lien == "ok" else "a_verifier", lien, None if s < 0 else s, C.wkb(g)))
    if MODEL.exists():
        cur.execute(
            "insert into modeles (nom, chemin, sha256, encodeur, entraine_le, actif) values "
            "(%s,%s,%s,%s,to_timestamp(%s),true) on conflict (nom) do nothing",
            ("v2_dinov3_small_night", str(MODEL), C.sha256(MODEL),
             "tu-convnext_small.dinov3_lvd1689m", MODEL.stat().st_mtime))
    db.commit()
    print(f"\n{n_feuilles} feuilles, {len(best)} parcelles candidates : {counts} "
          f"(les {counts['hors_feuille']} « hors_feuille » ne sont pas importées)")


if __name__ == "__main__":
    main()
