"""Prépare une feuille pour la validation « par parcelle principale » dans QGIS.

Sur les propositions IA EN ATTENTE de la feuille (parcelles_pending, review_status = 'pending') :
  1. type visible : classe majoritaire du modèle de typologie dans le polygone (bande de 3 px du
     bord exclue) -> typology_type + typology_conf ; polygone mixte (classe majoritaire < --mixte)
     signalé dans review_note (« à découper ? ») ;
  2. regroupement en parcelles principales (classifieur d'arêtes + union-find, regrouper.py)
     -> mains_pending + parcelles_pending.main_pending ;
  3. type des parties non bâties : « cour » dans une parcelle principale qui contient du bâti,
     sinon vide (labour, pré… ne se voient pas : matrices ou lettre, plus tard).
Rien n'est écrit sans --lancer (aperçu GPKG). Une proposition déjà revue n'est jamais touchée ;
relancer remplace les parcelles principales proposées encore en attente de la feuille.

    .venv/bin/python tools/typologie/proposer.py --feuille "Douai 1908 – Media 06" --out data/propositions/media06.gpkg
    .venv/bin/python tools/typologie/proposer.py --feuille "Douai 1908 – Media 06" --lancer
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
import timm
import torch
from rasterio.features import rasterize
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "annotation"))
import common as C  # noqa: E402
from predire import TYPE_DE_CLASSE, charger, probabilites  # noqa: E402
from regrouper import UF, proba_paires  # noqa: E402


def classe_majoritaire(g, carte, T, nc, bord=3):
    m = rasterize([(g, 1)], out_shape=carte.shape, transform=T, dtype="uint8") > 0
    m2 = ndimage.binary_erosion(m, iterations=bord)
    m = m2 if m2.sum() >= 20 else m
    if not m.any():
        return None, 0.0
    h = np.bincount(carte[m].ravel(), minlength=nc)
    k = int(h.argmax())
    return k, float(h[k] / h.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feuille", required=True)
    ap.add_argument("--modele-typo", default="work_dirs/typologie_v1/best_model.pth")
    ap.add_argument("--modele-aretes", default="work_dirs/regroupement_v1/best_model.pth")
    ap.add_argument("--seuil", type=float, default=0.5)
    ap.add_argument("--mixte", type=float, default=0.7)
    ap.add_argument("--out", help="aperçu GPKG")
    ap.add_argument("--lancer", action="store_true", help="écrire dans la base")
    a = ap.parse_args()
    con = C.connect()
    cur = con.cursor()
    fid, nom, raster = cur.execute(
        "SELECT id, nom, raster FROM feuilles WHERE id::text = %s OR nom = %s LIMIT 1", (a.feuille, a.feuille)).fetchone()
    rows = cur.execute("SELECT id, ST_AsBinary(geom), typology_type FROM parcelles_pending WHERE feuille_id = %s "
                       "AND review_status = 'pending' AND developpement IS NULL", (fid,)).fetchall()
    if not rows:
        sys.exit("Aucune proposition en attente sur cette feuille.")
    ids = [str(r[0]) for r in rows]
    polys = [C.geom_of(r[1]) for r in rows]
    dev = "cuda"
    model, classes = charger(a.modele_typo, dev)
    nc = len(classes)
    e = torch.load(a.modele_aretes, map_location=dev)
    m_ar = timm.create_model(e["config"]["arch"], pretrained=False, in_chans=6, num_classes=1).to(dev)
    m_ar.load_state_dict(e["model"])
    with rasterio.open(raster) as src:
        b = np.array([p.bounds for p in polys])
        inv = ~src.transform
        c0, r0 = inv * (b[:, 0].min(), b[:, 3].max())
        c1, r1 = inv * (b[:, 2].max(), b[:, 1].min())
        x0, y0 = max(0, int(c0) - 16), max(0, int(r0) - 16)
        x1, y1 = min(src.width, int(c1) + 16), min(src.height, int(r1) + 16)
        carte = probabilites(model, nc, src, x0, y0, x1, y1, dev).argmax(0).astype(np.uint8)
        T = src.transform * rasterio.Affine.translation(x0, y0)
        pp = proba_paires(m_ar, src, polys, 0.3, 24, 160, 3, dev)
    invT = ~T
    typ = []
    for g in polys:
        bx0, by0, bx1, by1 = g.bounds
        cc0, rr0 = invT * (bx0, by1)
        cc1, rr1 = invT * (bx1, by0)
        cc0, rr0 = max(0, int(cc0) - 2), max(0, int(rr0) - 2)
        cc1, rr1 = min(carte.shape[1], int(cc1) + 3), min(carte.shape[0], int(rr1) + 3)
        k, part = classe_majoritaire(g, carte[rr0:rr1, cc0:cc1], T * rasterio.Affine.translation(cc0, rr0), nc) \
            if cc1 > cc0 and rr1 > rr0 else (None, 0.0)
        typ.append((k, part))
    uf = UF(len(polys))
    conf_min = {}
    for (i, j), p in sorted(pp, key=lambda t: -t[1]):
        if p >= a.seuil and uf.find(i) != uf.find(j):
            ri, rj = uf.find(i), uf.find(j)
            c = min(conf_min.get(ri, 1.0), conf_min.get(rj, 1.0), p)
            uf.union(i, j)
            conf_min[uf.find(i)] = c
    grp = [uf.find(i) for i in range(len(polys))]
    bati = {g for g, (k, _) in zip(grp, typ) if k in (1, 2)}
    types = []
    for g, (k, part) in zip(grp, typ):
        t = TYPE_DE_CLASSE.get(k) if k is not None else None
        if k == 0:
            t = "cour" if g in bati else None
        types.append(t)
    n_groupes = len(set(grp))
    mixtes = sum(part < a.mixte for _, part in typ)
    print(f"{nom} : {len(polys)} propositions en attente -> {n_groupes} parcelles principales ; "
          f"types : " + ", ".join(f"{t or 'vide'} {types.count(t)}" for t in sorted(set(types), key=str))
          + f" ; {mixtes} polygone(s) mixte(s)")
    if a.out:
        import geopandas as gpd
        gdf = gpd.GeoDataFrame({"id": ids, "type": types, "classe": [classes[k] if k is not None else None for k, _ in typ],
                                "part": [round(p, 2) for _, p in typ], "groupe": grp},
                               geometry=polys, crs=f"EPSG:{C.SRID}")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(a.out, layer="typologies", driver="GPKG")
        gdf.dissolve("groupe").to_file(a.out, layer="principales", driver="GPKG")
        print(f"aperçu -> {a.out}")
    if a.lancer:
        # anciennes propositions de parcelles principales encore en attente : remplacées
        cur.execute("UPDATE parcelles_pending SET main_pending = NULL WHERE feuille_id = %s AND review_status = 'pending'", (fid,))
        cur.execute("DELETE FROM mains_pending m WHERE m.feuille_id = %s AND m.review_status = 'pending' AND NOT EXISTS "
                    "(SELECT 1 FROM parcelles_pending p WHERE p.main_pending = m.id)", (fid,))
        mid = {}
        for g in sorted(set(grp)):
            mid[g] = cur.execute("INSERT INTO mains_pending (feuille_id, confiance, source) VALUES (%s, %s, %s) RETURNING id",
                                 (fid, conf_min.get(g), f"proposer.py {Path(a.modele_aretes).parent.name}")).fetchone()[0]
        for i, pid in enumerate(ids):
            k, part = typ[i]
            note = "typologie mixte : à découper ?" if part < a.mixte else None
            cur.execute("UPDATE parcelles_pending SET typology_type = coalesce(%s, typology_type), typology_conf = %s, "
                        "main_pending = %s, review_note = CASE WHEN %s::text IS NULL THEN review_note "
                        "ELSE concat_ws(' | ', nullif(review_note, ''), %s::text) END "
                        "WHERE id = %s AND review_status = 'pending'",
                        (types[i], round(part, 3), mid[grp[i]], note, note, pid))
        for m in mid.values():
            cur.execute("SELECT main_recalculer(%s)", (m,))
        con.commit()
        print(f"écrit : {len(mid)} parcelles principales proposées, {len(ids)} propositions typées")


if __name__ == "__main__":
    main()
