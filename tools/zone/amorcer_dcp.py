"""Amorce de la zone utile des feuilles à partir des TIF découpés à la main (« DCP »).

Chaque DCP est un GeoTIFF RGBA dont l'alpha est la découpe faite pour GénéFoncier.
Ces découpes sont des compromis (décalages du géoréférencement) : on en fait des
PROPOSITIONS de zone (feuille_zones, origine 'dcp', statut 'a_verifier'), à corriger
dans QGIS pour coller au dessin de la feuille.

    .venv/bin/python tools/zone/amorcer_dcp.py                      # aperçu : GeoJSON + rapport
    .venv/bin/python tools/zone/amorcer_dcp.py --lancer             # écrit en base

Rattachement par le plan NON découpé d'où vient chaque DCP (sources.csv, produit par
sources_dcp.py : pixels identiques), dans cet ordre :
  1. feuille dont raster/autres_rasters = plan source ;
  2. feuille dont raster = le DCP lui-même -> son raster devient le plan source (pixels et
     géoréférencement identiques : dx/dy inchangés), le DCP passe dans autres_rasters ;
  3. feuille GénéFoncier (source 'export') dont l'emprise est la découpe (IoU >= 0,9) mais
     liée à un autre scan (assemblage) -> le plan source s'ajoute à autres_rasters ;
  4. sinon nouvelle feuille (statut 'a_verifier', raster = plan source, emprise du papier).
Tout reste « à vérifier » : rien n'entre dans un entraînement avant validation dans QGIS.
Une proposition de zone déjà retouchée ou validée n'est jamais écrasée.
"""
import argparse
import csv
import glob
import re
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "annotation"))
import common as C  # noqa: E402

BASE = "/home/lekalmar/BACKUP WINDOWS FILES/Travail/Cartographie/INDEXATION CADASTRE"
OUT = os.path.join(C.ROOT, "data", "zones", "amorce_dcp")
SCHEMA_DEBUT = "-- ---------------------------------------------------------------- zone utile des feuilles"


def trouver(racines):  # utilisé par sources_dcp.py via glob direct ; gardé pour usage manuel
    vus, out = set(), []
    for r in racines:
        for p in glob.glob(os.path.join(r, "**", "DCP*.tif"), recursive=True):
            k = os.path.realpath(p)
            if k not in vus:
                vus.add(k)
                out.append(p)
    return sorted(out)


def decoupe(path, cote_max=3000, trou_min_m2=200.0, frag_min=0.01):
    """Polygone L93 (MultiPolygon shapely) de l'alpha du DCP, et taille de pixel (m)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import shapes
    from shapely.geometry import MultiPolygon, Polygon, shape
    from shapely.ops import transform as tr, unary_union

    with rasterio.open(path) as src:
        k = max(1, int(np.ceil(max(src.width, src.height) / cote_max)))
        h, w = max(1, src.height // k), max(1, src.width // k)
        if src.count >= 4:
            a = src.read(4, out_shape=(h, w), resampling=Resampling.nearest)
        else:
            a = src.dataset_mask(out_shape=(h, w))
        T = src.transform * src.transform.scale(src.width / w, src.height / h)
        crs = src.crs
    m = (a > 127).astype(np.uint8)
    polys = [shape(g) for g, v in shapes(m, mask=m.astype(bool), transform=T) if v]
    if not polys:
        return None, abs(T.a)
    g = unary_union(polys)
    if crs is not None and crs.to_epsg() not in (None, C.SRID):
        from pyproj import Transformer
        f = Transformer.from_crs(crs, f"EPSG:{C.SRID}", always_xy=True).transform
        g = tr(f, g)
    parts = list(g.geoms) if hasattr(g, "geoms") else [g]
    tot = sum(p.area for p in parts)
    garde = []
    for p in parts:
        if p.area < frag_min * tot:
            continue
        trous = [i for i in p.interiors if Polygon(i).area >= trou_min_m2]
        garde.append(Polygon(p.exterior, trous))
    px = abs(T.a)
    g = MultiPolygon([q for q in (p.simplify(px, preserve_topology=True).buffer(0) for p in garde)
                      for q in (q.geoms if hasattr(q, "geoms") else [q]) if not q.is_empty])
    return g, px


def feuille_existante(cur, dcp, source, g):
    """(id, nom, cas) ou None. cas : 'source', 'dcp' (raster à remplacer), 'export'."""
    r = cur.execute("SELECT id, nom FROM feuilles WHERE raster = %(s)s "
                    "OR %(s)s = ANY(string_to_array(autres_rasters, ';')) ORDER BY raster = %(s)s DESC",
                    {"s": source}).fetchone()
    if r:
        return r[0], r[1], "source"
    r = cur.execute("SELECT id, nom FROM feuilles WHERE raster = %s", (dcp,)).fetchone()
    if r:
        return r[0], r[1], "dcp"
    r = cur.execute(
        "WITH d AS (SELECT ST_GeomFromWKB(%s, 2154) AS g) "
        "SELECT f.id, f.nom, ST_Area(ST_Intersection(f.geom, d.g)) / ST_Area(ST_Union(f.geom, d.g)) iou "
        "FROM feuilles f, d WHERE f.source = 'export' AND f.geom && d.g ORDER BY iou DESC LIMIT 1",
        (C.wkb(g),)).fetchone()
    if r and r[2] >= 0.9:
        return r[0], r[1], "export"
    return None


def identite(dcp, source):
    """commune, section, n° de feuille, année, nom lisible, tirés des noms de fichier."""
    b = re.sub(r"\.tif$", "", os.path.basename(dcp), flags=re.I)
    b = re.sub(r"^DCP[\s_-]*", "", b)
    an = re.findall(r"(?<!\d)(1[78]\d\d)(?!\d)", b) or re.findall(r"(?<!\d)(1[78]\d\d)(?!\d)", source)
    annee = int(an[-1]) if an else None
    t = [x.strip() for x in re.split(r"\s*_\s*|\s+-\s*|-\s+", b) if x.strip() and x.strip() != str(annee)]
    commune = t[0] if t else None
    section = next((x for x in t[1:] if re.fullmatch(r"[A-Z]{1,2}", x)), None)
    num = next((x for x in t[1:] if x.isdigit()), None)
    nom = " ".join(x for x in [commune, section, num] if x) + (f" {annee}" if annee else "") + " (DCP)"
    return commune, section, num, annee, nom


def creer_feuille(cur, dcp, source):
    import rasterio
    from rasterio.warp import transform_geom
    from shapely.geometry import mapping, shape
    commune, section, num, annee, nom = identite(dcp, source)
    fp = C.paper_footprint(source)
    with rasterio.open(source) as s:
        fp = shape(transform_geom(s.crs, f"EPSG:{C.SRID}", mapping(fp)))
    fid = cur.execute(
        "INSERT INTO feuilles (nom, commune, section, feuille_no, annee, raster, autres_rasters, statut, "
        "source, note, geom) VALUES (%s,%s,%s,%s,%s,%s,%s,'a_verifier','scan',%s,ST_GeomFromWKB(%s, 2154)) "
        "RETURNING id",
        (nom, commune, section, num, annee, source, dcp,
         "Créée depuis un DCP (amorcer_dcp.py) : identité et zone à vérifier avant tout entraînement.",
         C.wkb(fp))).fetchone()[0]
    return fid, nom


def ajouter_autre(cur, fid, chemin):
    cur.execute("UPDATE feuilles SET autres_rasters = CASE WHEN autres_rasters IS NULL OR autres_rasters = '' "
                "THEN %(c)s ELSE autres_rasters || ';' || %(c)s END, modifie_le = now() WHERE id = %(f)s "
                "AND NOT (%(c)s = ANY(string_to_array(coalesce(autres_rasters, ''), ';')))",
                {"c": chemin, "f": fid})


def creer_table(cur):
    sql = open(os.path.join(C.WORK, "schema.sql"), encoding="utf-8").read()
    i = sql.find(SCHEMA_DEBUT)
    if i < 0:
        sys.exit("schema.sql ne contient pas le bloc feuille_zones")
    cur.execute(sql[i:])


def main():
    P = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    P.add_argument("--sources", default=os.path.join(OUT, "sources.csv"), help="sortie de sources_dcp.py")
    P.add_argument("--out", default=OUT)
    P.add_argument("--lancer", action="store_true",
                   help="écrit en base (feuilles manquantes, rasters sources, propositions de zone)")
    a = P.parse_args()

    from shapely.geometry import mapping

    os.makedirs(a.out, exist_ok=True)
    sources = [r for r in csv.DictReader(open(a.sources, encoding="utf-8")) if r["source"]]
    print(f"{len(sources)} DCP avec plan source", flush=True)
    con = C.connect()
    cur = con.cursor()
    if a.lancer:
        creer_table(cur)
    feats, rapport, n_zones = [], [], 0
    for i, r in enumerate(sources, 1):
        dcp, source = r["dcp"], r["source"]
        nom_dcp = os.path.basename(dcp)
        g, px = decoupe(dcp)
        if g is None or g.is_empty:
            rapport.append({"dcp": dcp, "source": source, "action": "alpha vide"})
            continue
        ex = feuille_existante(cur, dcp, source, g)
        if ex:
            fid, nom, cas = ex
            action = {"source": "feuille existante", "dcp": "raster DCP -> plan source",
                      "export": "feuille GénéFoncier, plan source ajouté"}[cas]
            if a.lancer and cas == "dcp":
                cur.execute("UPDATE feuilles SET raster = %s, modifie_le = now() WHERE id = %s", (source, fid))
                ajouter_autre(cur, fid, dcp)
            elif a.lancer and cas == "export":
                ajouter_autre(cur, fid, source)
        else:
            nom = identite(dcp, source)[4]
            fid, action = None, "nouvelle feuille"
            if a.lancer:
                fid, nom = creer_feuille(cur, dcp, source)
        if a.lancer:
            n_zones += cur.execute(
                "INSERT INTO feuille_zones (feuille_id, origine, source, raster, geom_origine, geom, note) "
                "VALUES (%s, 'dcp', %s, %s, ST_GeomFromWKB(%s, 2154), ST_GeomFromWKB(%s, 2154), "
                "'Découpe GénéFoncier (compromis de géoréférencement) : à recaler sur le dessin.') "
                "ON CONFLICT (feuille_id, source) WHERE source IS NOT NULL DO UPDATE "
                "SET geom = EXCLUDED.geom, geom_origine = EXCLUDED.geom_origine, raster = EXCLUDED.raster "
                "WHERE feuille_zones.statut = 'a_verifier' AND feuille_zones.geom = feuille_zones.geom_origine",
                (fid, dcp, source, C.wkb(g), C.wkb(g))).rowcount
            con.commit()  # feuille par feuille : une erreur n'annule pas les précédentes
        ligne = {"dcp": dcp, "source": source, "action": action, "feuille": nom,
                 "feuille_id": str(fid or ""), "aire_ha": round(g.area / 1e4, 2), "px_m": round(px, 3)}
        rapport.append(ligne)
        feats.append({"type": "Feature", "geometry": mapping(g), "properties": ligne})
        print(f"[{i}/{len(sources)}] {nom_dcp[:38]:38} {action:40} {nom[:45]}", flush=True)
    if a.lancer:
        print(f"{n_zones} propositions de zone écrites ou mises à jour")
    gj = os.path.join(a.out, "amorce_dcp.geojson")
    with open(gj, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection",
                   "crs": {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{C.SRID}"}},
                   "features": feats}, fh)
    rp = os.path.join(a.out, "rapport.csv")
    with open(rp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["dcp", "source", "action", "feuille", "feuille_id", "aire_ha", "px_m"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rapport)
    from collections import Counter
    print("\n" + "\n".join(f"{v:3} {k}" for k, v in Counter(l["action"] for l in rapport).items()))
    print(f"-> {gj}\n-> {rp}")


if __name__ == "__main__":
    main()
