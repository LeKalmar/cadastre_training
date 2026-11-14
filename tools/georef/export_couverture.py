"""Exporte la base cadastre_annotation pour la carte « Couverture des extractions »
(artefact claude.ai). Deux fichiers à envoyer ensuite par le bouton « Mettre à jour les
données » de la carte :

    .venv/bin/python tools/georef/export_couverture.py ~/couverture
        -> feuilles.json   emprises des feuilles + compteurs de parcelles (~0,5 Mo)
        -> parcelles.txt   contours des parcelles, binaire gzip encodé en base64

Fond communal (rarement utile : il est déjà dans la page), avec GDAL :
    .venv-georef/bin/python tools/georef/export_couverture.py ~/couverture --communes

Format de parcelles.txt, une fois décodé (base64 puis gzip) : uint32 nombre d'anneaux, puis
pour chaque anneau des entiers « varint zigzag » : catégorie (0 validée, 1 à vérifier,
2 proposition IA en attente), indice de feuille dans feuilles.json, nombre de sommets,
premier sommet en écart au premier sommet de l'anneau précédent, puis écarts successifs.
Coordonnées Lambert-93 arrondies au mètre, anneau fermé implicite, trous ignorés.
"""
import argparse
import base64
import gzip
import json
import os
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMUNES_GPKG = ("/home/lekalmar/BACKUP WINDOWS FILES/Travail/Cartographie/INDEXATION CADASTRE/"
                 "Cadastre XIX/COMMUNES_2026.gpkg")
TOL_FEUILLE, TOL_PARCELLE, TOL_COMMUNE = 6, 1.5, 25   # simplification (m)


def export_base(out):
    sys.path.insert(0, str(ROOT / "tools" / "annotation"))
    import shapely
    from common import connect

    con = connect()
    cur = con.cursor()
    cur.execute(f"""
    with p as (select feuille_id,
                 count(*) filter (where statut='valide') v,
                 count(*) filter (where statut='a_verifier') av
               from parcelles group by 1),
         q as (select feuille_id,
                 count(*) filter (where review_status='pending') pe,
                 count(*) filter (where review_status='approved') ap,
                 count(*) filter (where review_status='corrected') co,
                 count(*) filter (where review_status='rejected') rj
               from parcelles_pending group by 1),
         e as (select feuille_id, max(lance_le) last from extractions group by 1)
    select f.id, f.nom, f.commune, f.section, f.feuille_no, f.annee, f.statut, f.source,
           coalesce(p.v,0), coalesce(p.av,0), coalesce(q.pe,0), coalesce(q.ap,0), coalesce(q.co,0),
           coalesce(q.rj,0), to_char(e.last,'YYYY-MM-DD'), f.priorite,
           ST_AsBinary(ST_SimplifyPreserveTopology(f.geom, {TOL_FEUILLE})), ST_Area(f.geom)
    from feuilles f left join p on p.feuille_id=f.id left join q on q.feuille_id=f.id
    left join e on e.feuille_id=f.id order by f.commune nulls last, f.annee, f.nom""")
    feuilles, fidx = [], {}
    for r in cur.fetchall():
        g = shapely.from_wkb(bytes(r[16]))
        rings = [[[round(x), round(y)] for x, y in poly.exterior.coords]
                 for poly in getattr(g, "geoms", [g]) if not poly.is_empty]
        fidx[r[0]] = len(feuilles)
        feuilles.append(dict(nom=r[1], commune=r[2], section=r[3], no=r[4], annee=r[5], statut=r[6],
                             source=r[7], v=r[8], av=r[9], pe=r[10], ap=r[11], co=r[12], rj=r[13],
                             ext=r[14], prio=r[15], ha=round(r[17] / 1e4, 1), rings=rings))

    buf = bytearray()

    def varint(n):
        n = (n << 1) ^ (n >> 63)
        while n >= 0x80:
            buf.append((n & 0x7F) | 0x80)
            n >>= 7
        buf.append(n)

    n_rings, prev = 0, (0, 0)
    for sql in (
        f"select feuille_id, case when statut='valide' then 0 else 1 end, "
        f"ST_AsBinary(ST_SimplifyPreserveTopology(geom,{TOL_PARCELLE})) from parcelles "
        f"where statut<>'rejete' order by feuille_id",
        f"select feuille_id, 2, ST_AsBinary(ST_SimplifyPreserveTopology(geom,{TOL_PARCELLE})) "
        f"from parcelles_pending where review_status='pending' order by feuille_id",
    ):
        with con.cursor(name="c") as sc:
            sc.itersize = 20000
            sc.execute(sql)
            for fid, cat, wkb in sc:
                g = shapely.from_wkb(bytes(wkb))
                for poly in getattr(g, "geoms", [g]):
                    if poly.is_empty:
                        continue
                    cs = [(round(x), round(y)) for x, y in poly.exterior.coords]
                    pts = [cs[0]] + [c for a, c in zip(cs, cs[1:]) if c != a]
                    if len(pts) < 4:
                        continue
                    pts = pts[:-1]
                    varint(cat); varint(fidx[fid]); varint(len(pts))
                    varint(pts[0][0] - prev[0]); varint(pts[0][1] - prev[1])
                    prev = pts[0]
                    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                        varint(x1 - x0); varint(y1 - y0)
                    n_rings += 1
        con.commit()

    date = time.strftime("%Y-%m-%dT%H:%M")
    (out / "feuilles.json").write_text(json.dumps(
        {"date": date, "base": "cadastre_annotation", "anneaux": n_rings, "feuilles": feuilles},
        ensure_ascii=False, separators=(",", ":")))
    raw = gzip.compress(struct.pack("<I", n_rings) + bytes(buf), 9)
    (out / "parcelles.txt").write_bytes(base64.b64encode(raw))
    for name in ("feuilles.json", "parcelles.txt"):
        print(f"{out / name}  ({os.path.getsize(out / name) / 1e6:.1f} Mo)")
    print(f"{len(feuilles)} feuilles, {n_rings} contours de parcelles, état du {date}")


def export_communes(out):
    import shapely
    from osgeo import ogr, osr
    ogr.UseExceptions()
    ds = ogr.Open(COMMUNES_GPKG)
    lyr = ds.GetLayer(0)
    src, dst = lyr.GetSpatialRef(), osr.SpatialReference()
    dst.ImportFromEPSG(2154)
    for s in (src, dst):
        s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tr = osr.CoordinateTransformation(src, dst)
    lyr.SetAttributeFilter("code_insee_du_departement in ('59','62')")
    communes = []
    for f in lyr:
        g = f.GetGeometryRef().Clone()
        g.Transform(tr)
        sg = shapely.from_wkb(bytes(g.ExportToWkb())).simplify(TOL_COMMUNE)
        rings = [[[round(x), round(y)] for x, y in p.exterior.coords] for p in getattr(sg, "geoms", [sg])]
        c = sg.representative_point()
        communes.append(dict(n=f.GetField("nom_officiel"), i=f.GetField("code_insee"),
                             c=[round(c.x), round(c.y)], a=round(sg.area / 1e6, 1), rings=rings))
    (out / "communes.json").write_text(json.dumps(communes, ensure_ascii=False, separators=(",", ":")))
    print(f"{out / 'communes.json'} : {len(communes)} communes (Nord, Pas-de-Calais)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="dossier de sortie")
    ap.add_argument("--communes", action="store_true", help="exporte le fond communal (GDAL requis)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    export_communes(a.out) if a.communes else export_base(a.out)
