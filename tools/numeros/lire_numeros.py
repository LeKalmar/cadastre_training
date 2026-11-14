"""Lit les numéros de parcelles d'une feuille (OCR) et les rattache aux polygones.

RapidOCR (détection + reconnaissance + classifieur d'orientation) par tuiles de --tuile px
recouvrantes, à résolution native. On garde les lectures qui ressemblent à un numéro de
parcelle (chiffres, éventuellement suivis de « bis » ou d'une lettre), en corrigeant les
confusions classiques (O->0, l/I->1, S->5…). Chaque numéro va au polygone qui contient le
centre de sa boîte ; doublons de tuiles fusionnés.

Avec --verite : rappel = part des parcelles principales (par_main_id, sinon la typologie seule)
dont le numéro GénéFoncier est retrouvé dans l'une de leurs typologies ; précision = part des
numéros rattachés qui sont justes.

    .venv/bin/python tools/numeros/lire_numeros.py --feuille "Arras C 0 1838" --verite --out data/numeros/arras_c.gpkg
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from shapely import STRtree
from shapely.geometry import Point, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "annotation"))
import common as C  # noqa: E402

CONFUSIONS = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1", "i": "1", "|": "1",
                            "!": "1", "S": "5", "s": "5", "Z": "2", "z": "2", "B": "8", "g": "9", "q": "9",
                            "G": "6", "b": "6", "T": "7"})
NUM = re.compile(r"^(\d{1,4})(bis|ter|[a-z])?$")
_OCR = None


def ocr():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR(det_limit_side_len=1024, det_limit_type="max", det_model_path="")
    return _OCR


def normaliser(t):
    t = re.sub(r"[\s.,'`°_\-]+", "", t)
    m = re.match(r"^(.*?)(bis|ter)?$", t, re.I)
    corps, suff = m.group(1), (m.group(2) or "").lower()
    corps = corps.translate(CONFUSIONS) if not corps.isdigit() else corps
    t = corps + suff
    m = NUM.match(t)
    if not m or m.group(1).startswith("0"):
        return None
    return m.group(1) + (m.group(2) or "")


def lire_feuille(raster, emprise, tuile, recouv, score_min):
    """[(numero, score, polygone L93 de la boîte)] sur l'emprise (bounds L93) du scan."""
    out = []
    with rasterio.open(raster) as src:
        inv = ~src.transform
        c0, r0 = inv * (emprise[0], emprise[3])
        c1, r1 = inv * (emprise[2], emprise[1])
        c0, r0 = max(0, int(c0)), max(0, int(r0))
        c1, r1 = min(src.width, int(c1)), min(src.height, int(r1))
        pas = tuile - recouv
        for y in range(r0, max(r0 + 1, r1 - recouv), pas):
            for x in range(c0, max(c0 + 1, c1 - recouv), pas):
                w = Window(x, y, min(tuile, src.width - x), min(tuile, src.height - y))
                if (src.read_masks(1, window=w) > 0).mean() < 0.3:
                    continue
                img = np.ascontiguousarray(np.transpose(src.read([1, 2, 3], window=w), (1, 2, 0))[..., ::-1])
                res, _ = ocr()(img, use_cls=True)
                for box, texte, score in res or []:
                    n = normaliser(texte)
                    if n is None or float(score) < score_min:
                        continue
                    pts = [src.transform * (x + float(px), y + float(py)) for px, py in box]
                    out.append((n, float(score), Polygon(pts)))
    # doublons (recouvrement des tuiles) : même numéro, boîtes qui se touchent -> la meilleure
    out.sort(key=lambda t: -t[1])
    garde = []
    for n, s, b in out:
        if any(n == n2 and b.intersects(b2) for n2, _, b2 in garde):
            continue
        garde.append((n, s, b))
    return garde


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feuille", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tuile", type=int, default=1024)
    ap.add_argument("--recouv", type=int, default=160)
    ap.add_argument("--score-min", type=float, default=0.5)
    ap.add_argument("--verite", action="store_true")
    a = ap.parse_args()

    import geopandas as gpd

    cur = C.connect().cursor()
    fid, nom, raster = cur.execute(
        "SELECT id, nom, raster FROM feuilles WHERE id::text = %s OR nom = %s "
        "ORDER BY (SELECT count(*) FROM parcelles p WHERE p.feuille_id = feuilles.id) DESC LIMIT 1",
        (a.feuille, a.feuille)).fetchone()
    rows = cur.execute("SELECT id, coalesce(par_main_id::text, id::text), parcelles_no, parcelles_bis, "
                       "ST_AsBinary(geom) FROM parcelles WHERE feuille_id = %s AND statut = 'valide' "
                       "AND developpement IS NULL", (fid,)).fetchall()
    ids = [str(r[0]) for r in rows]
    mains = [r[1] for r in rows]
    vrais = [(str(r[2]).strip() + (str(r[3]).strip() if r[3] else "")) if r[2] else None for r in rows]
    polys = [C.geom_of(r[4]) for r in rows]
    tree = STRtree(polys)
    emprise = np.array([p.bounds for p in polys])
    emprise = (emprise[:, 0].min(), emprise[:, 1].min(), emprise[:, 2].max(), emprise[:, 3].max())
    lus = lire_feuille(raster, emprise, a.tuile, a.recouv, a.score_min)
    feats = []
    par_poly = defaultdict(list)
    for n, s, b in lus:
        c = b.centroid
        cible = next((int(i) for i in tree.query(c) if polys[int(i)].contains(Point(c.x, c.y))), None)
        feats.append({"numero": n, "score": round(s, 3), "typologie": ids[cible] if cible is not None else None,
                      "vrai": vrais[cible] if cible is not None else None, "geometry": b})
        if cible is not None:
            par_poly[cible].append(n)
    gdf = gpd.GeoDataFrame(feats, crs=f"EPSG:{C.SRID}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(a.out, layer="numeros", driver="GPKG")
    print(f"{nom} : {len(lus)} numéros lus, {sum(f['typologie'] is not None for f in feats)} dans une parcelle -> {a.out}")
    if a.verite:
        vrai_main = {}
        for m, v in zip(mains, vrais):
            if v:
                vrai_main.setdefault(m, v)
        trouve = defaultdict(set)
        for i, ns in par_poly.items():
            trouve[mains[i]].update(ns)
        ok = sum(v in trouve[m] for m, v in vrai_main.items())
        dans = [f for f in feats if f["typologie"] is not None and f["vrai"]]
        justes = sum(f["numero"] == f["vrai"] for f in dans)
        print(f"Rappel : numéro retrouvé dans {ok}/{len(vrai_main)} parcelles principales = {ok / max(len(vrai_main), 1):.1%}")
        print(f"Précision : {justes}/{len(dans)} numéros rattachés justes = {justes / max(len(dans), 1):.1%}")


if __name__ == "__main__":
    main()
