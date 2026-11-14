"""Calages « à vérifier » : table georef_verif de la base cadastre_annotation.

run.py classe « à vérifier » les feuilles qu'il a placées sans pouvoir confirmer le
calage (cadastre actuel trop différent, une seule feuille d'appui, cotes douteuses).
Ce module recopie ces feuilles dans georef_verif (emprise du GeoTIFF, motif, mesures)
pour qu'on les juge dans QGIS (extension « Annotation cadastrale », bouton « Calages »).
importer_lot.py appelle sync() à chaque passage et extrait les feuilles jugées « bon ».

    .venv/bin/python tools/georef/verif.py data/georef_nord data/georef_ad59     # synchronise
    .venv/bin/python tools/georef/verif.py --bilan                               # décompte

Une feuille recalée depuis la décision (emprise déplacée de plus de 0,5 m) repasse
« a_verifier » : la décision portait sur l'ancien calage.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "annotation"))

A_VERIFIER = "à vérifier"
TOL_DEPLACEMENT = 0.5


def emprise(tif):
    """Emprise L93 (xmin, ymin, xmax, ymax) lue dans les balises GeoTIFF, sans GDAL."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(tif) as im:
            sx, sy = im.tag_v2[33550][:2]
            _, _, _, x0, y0, _ = im.tag_v2[33922][:6]
            W, H = im.size
    except Exception:
        return None
    return [x0, y0 - sy * H, x0 + sx * W, y0]


def lire(outs):
    """Feuilles « à vérifier » des dossiers de sortie de run.py."""
    rows = []
    for out in outs:
        lot = os.path.basename(os.path.normpath(out))
        for cj in sorted(glob.glob(os.path.join(out, "*", "*", "calage.json"))):
            meta = json.load(open(cj))
            for f in meta["feuilles"]:
                tif = f.get("geotiff")
                if not f["statut"].startswith(A_VERIFIER) or not tif or not os.path.exists(tif):
                    continue
                bb = emprise(tif)
                if bb is None:
                    continue
                rows.append(dict(geotiff=tif, lot=lot, commune=meta["commune"], nom=meta.get("nom"),
                                 annee=meta.get("annee"), feuille=f["feuille"], statut_calage=f["statut"],
                                 pas_m=f.get("pas_m"), z=f.get("z"), ecart_m=f.get("ecart_cadastre_m"),
                                 appui=f.get("appui"), resolution_m=f.get("resolution_m"), bb=bb))
    return rows


def sync(outs, db=None):
    """Met georef_verif à jour d'après les calage.json ; renvoie le nombre de feuilles ajoutées."""
    import common as C
    rows = lire(outs)
    lots = [os.path.basename(os.path.normpath(o)) for o in outs]
    own = db is None
    db = db or C.connect()
    ajout = 0
    try:
        cur = db.cursor()
        for r in rows:
            env = "ST_Multi(ST_MakeEnvelope(%s, %s, %s, %s, 2154))"
            cur.execute(f"""
                INSERT INTO georef_verif (geotiff, lot, commune, nom, annee, feuille, statut_calage, pas_m, z,
                                          ecart_m, appui, resolution_m, geom)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, {env})
                ON CONFLICT (geotiff) DO UPDATE SET
                    statut_calage = EXCLUDED.statut_calage, pas_m = EXCLUDED.pas_m, z = EXCLUDED.z,
                    ecart_m = EXCLUDED.ecart_m, appui = EXCLUDED.appui, resolution_m = EXCLUDED.resolution_m,
                    present = true, vu_le = now(), geom = EXCLUDED.geom,
                    decision = CASE WHEN ST_HausdorffDistance(georef_verif.geom, EXCLUDED.geom) > {TOL_DEPLACEMENT}
                                    THEN 'a_verifier' ELSE georef_verif.decision END,
                    note = CASE WHEN ST_HausdorffDistance(georef_verif.geom, EXCLUDED.geom) > {TOL_DEPLACEMENT}
                                     AND georef_verif.decision <> 'a_verifier'
                                THEN concat_ws(' | ', georef_verif.note,
                                     'recalée le ' || to_char(now(), 'DD/MM/YYYY') ||
                                     ' : décision « ' || georef_verif.decision || ' » à refaire')
                                ELSE georef_verif.note END
                RETURNING (xmax = 0)""",
                (r["geotiff"], r["lot"], r["commune"], r["nom"], r["annee"], r["feuille"], r["statut_calage"],
                 r["pas_m"], r["z"], r["ecart_m"], r["appui"], r["resolution_m"], *r["bb"]))
            nouveau, = cur.fetchone()
            ajout += bool(nouveau)
        # feuilles qui ne sont plus « à vérifier » (recalées « ok » ou autre) : gardées, marquées absentes
        cur.execute("UPDATE georef_verif SET present = false WHERE lot = ANY(%s) AND present "
                    "AND NOT (geotiff = ANY(%s))", (lots, [r["geotiff"] for r in rows]))
        db.commit()
    finally:
        if own:
            db.close()
    return ajout


def decisions(db=None):
    """{geotiff: 'bon' | 'faux'} pour les feuilles jugées."""
    import common as C
    own = db is None
    db = db or C.connect()
    try:
        return dict(db.execute("SELECT geotiff, decision FROM georef_verif WHERE decision <> 'a_verifier' "
                               "AND present").fetchall())
    finally:
        if own:
            db.close()


def bilan():
    import common as C
    with C.connect() as db:
        for lot, dec, n in db.execute("SELECT lot, decision, count(*) FROM georef_verif WHERE present "
                                      "GROUP BY 1, 2 ORDER BY 1, 2").fetchall():
            print(f"{lot:14} {dec:11} {n}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--bilan"]:
        bilan()
    elif len(sys.argv) > 1:
        a = sync(sys.argv[1:])
        print(f"{a} feuille(s) ajoutée(s)")
        bilan()
    else:
        print(__doc__)
