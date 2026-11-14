"""Typologie visible d'une feuille : sous-polygones typés de chaque parcelle.

1. Le modèle de segmentation (train_seg.py) passe sur la feuille par tuiles de 512 px
   recouvrantes (moyenne des probabilités, bords de tuile atténués).
2. Chaque parcelle (validée, ou proposition IA en attente) est découpée selon la classe
   prédite : régions lissées (ouverture/fermeture de --lisse px), régions de moins de
   --min-m2 fondues dans la classe voisine dominante, contour simplifié à --simplifie px.
3. Sortie GPKG (rien n'est écrit en base) : couche « typologies » (type, classe, confiance,
   parcelle = id de la parcelle d'origine, comme par_main_id) ; avec --verite, comparaison
   aux typologies GénéFoncier de la feuille (part de surface bien typée).

    .venv/bin/python tools/typologie/predire.py --feuille "Arras C 0 1838" --modele work_dirs/typologie_v1/best_model.pth \
        --out data/typologie_pred/arras_c.gpkg --verite
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from rasterio.features import rasterize, shapes
from rasterio.windows import Window
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "annotation"))
import common as C  # noqa: E402

TILE, STRIDE = 512, 384
TYPE_DE_CLASSE = {0: None, 1: "batiment", 2: "bat-public", 3: "eau", 4: "jardin"}
# classe 0 « non bâti » : le type précis (cour, labour…) ne se voit pas ; on garde celui de la
# parcelle s'il existe, sinon rien. Classe 4 « végétation » : « jardin » par défaut (bois, verger
# se distinguent par la lettre, phase OCR).
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


def charger(chemin, dev):
    e = torch.load(chemin, map_location=dev)
    cfg = e["config"]
    m = smp.Unet(cfg["encoder"], encoder_weights=None, classes=len(cfg["classes"]),
                 decoder_attention_type="scse").to(dev)
    m.load_state_dict(e["model"])
    return m.eval(), cfg["classes"]


@torch.no_grad()
def probabilites(model, nc, src, x0, y0, x1, y1, dev, batch=8):
    """Probabilités (nc, H, W) sur la fenêtre [x0:x1, y0:y1] du scan."""
    W, H = x1 - x0, y1 - y0
    acc = np.zeros((nc, H, W), np.float32)
    poids = np.zeros((H, W), np.float32)
    r = np.minimum(np.arange(TILE) + 1, TILE - np.arange(TILE)).astype(np.float32)
    w2 = np.minimum.outer(np.minimum(r / 64, 1), np.minimum(r / 64, 1))   # atténuation des bords
    pos = [(x, y) for y in range(0, max(1, H - TILE) + STRIDE, STRIDE) for x in range(0, max(1, W - TILE) + STRIDE, STRIDE)]
    pos = sorted({(min(x, max(0, W - TILE)), min(y, max(0, H - TILE))) for x, y in pos})
    for i in range(0, len(pos), batch):
        lot = pos[i:i + batch]
        xs = []
        for x, y in lot:
            a = src.read([1, 2, 3], window=Window(x0 + x, y0 + y, TILE, TILE), boundless=True, fill_value=255)
            xs.append((a.astype(np.float32) / 255 - MEAN) / STD)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.softmax(model(torch.from_numpy(np.stack(xs)).to(dev)).float(), 1).cpu().numpy()
        for (x, y), pp in zip(lot, p):
            h, w = min(TILE, H - y), min(TILE, W - x)
            acc[:, y:y + h, x:x + w] += pp[:, :h, :w] * w2[:h, :w]
            poids[y:y + h, x:x + w] += w2[:h, :w]
    return acc / np.maximum(poids, 1e-6)


def decouper(parcelle, carte, conf, T, px, lisse, min_m2, simplifie):
    """[(geom, classe, confiance)] : la parcelle découpée selon la carte de classes."""
    inside = rasterize([(parcelle, 1)], out_shape=carte.shape, transform=T, dtype="uint8") > 0
    if inside.sum() == 0:
        return []
    cl = np.where(inside, carte, 255).astype(np.uint8)
    for k in np.unique(cl[inside]):                    # lissage par classe
        m = cl == k
        m2 = ndimage.binary_closing(ndimage.binary_opening(m, iterations=lisse), iterations=lisse) & inside
        cl[m & ~m2] = 255
    # pixels perdus au lissage -> classe voisine la plus proche
    trou = inside & (cl == 255)
    if trou.any() and (cl != 255).any():
        idx = ndimage.distance_transform_edt(cl == 255, return_distances=False, return_indices=True)
        cl[trou] = cl[idx[0][trou], idx[1][trou]]
    out = []
    for g, v in shapes(cl, mask=inside & (cl != 255), transform=T):
        geom = shape(g).intersection(parcelle)
        if geom.is_empty:
            continue
        out.append([geom, int(v)])
    # petites régions fondues dans la voisine qui partage le plus de contour
    out.sort(key=lambda t: t[0].area)
    fusion = []
    for g, k in out:
        if g.area < min_m2 and len(out) > 1:
            voisins = [f for f in fusion + [o for o in out if o[0] is not g and o not in fusion]
                       if f[0].touches(g) or f[0].distance(g) < px]
            if voisins:
                v = max(voisins, key=lambda f: f[0].boundary.intersection(g.buffer(px)).length)
                v[0] = unary_union([v[0], g])
                continue
        fusion.append([g, k])
    res = []
    for g, k in fusion:
        g = g.simplify(simplifie * px, preserve_topology=True).buffer(0)
        if g.is_empty:
            continue
        m = rasterize([(g, 1)], out_shape=carte.shape, transform=T, dtype="uint8") > 0
        res.append((g, k, float(conf[m].mean()) if m.any() else 0.0))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feuille", required=True, help="nom ou id de la feuille")
    ap.add_argument("--modele", default="work_dirs/typologie_v1/best_model.pth")
    ap.add_argument("--out", required=True, help="fichier .gpkg")
    ap.add_argument("--source", choices=["valide", "pending"], default="valide",
                    help="parcelles à découper : validées (évaluation) ou propositions IA en attente")
    ap.add_argument("--verite", action="store_true", help="comparer aux typologies GénéFoncier de la feuille")
    ap.add_argument("--lisse", type=int, default=2)
    ap.add_argument("--min-m2", type=float, default=4.0)
    ap.add_argument("--simplifie", type=float, default=1.5, help="tolérance de simplification (px)")
    a = ap.parse_args()

    import geopandas as gpd

    cur = C.connect().cursor()
    f = cur.execute("SELECT id, nom, raster FROM feuilles WHERE id::text = %s OR nom = %s "
                    "ORDER BY (SELECT count(*) FROM parcelles p WHERE p.feuille_id = feuilles.id) DESC LIMIT 1",
                    (a.feuille, a.feuille)).fetchone()
    if not f:
        sys.exit("Feuille introuvable.")
    fid, nom, raster = f
    if a.source == "valide":
        # parcelle principale = union des typologies de même par_main_id (sinon la typologie seule)
        rows = cur.execute(
            "SELECT coalesce(par_main_id::text, id::text), ST_AsBinary(ST_Union(geom)), "
            "string_agg(DISTINCT type, ',') FROM parcelles WHERE feuille_id = %s AND statut = 'valide' "
            "AND developpement IS NULL GROUP BY 1", (fid,)).fetchall()
    else:
        rows = cur.execute("SELECT id::text, ST_AsBinary(geom), typology_type FROM parcelles_pending "
                           "WHERE feuille_id = %s AND review_status = 'pending'", (fid,)).fetchall()
    parcelles = [(pid, C.geom_of(g), t) for pid, g, t in rows]
    print(f"{nom} : {len(parcelles)} parcelles", flush=True)
    dev = "cuda"
    model, classes = charger(a.modele, dev)
    nc = len(classes)
    feats = []
    with rasterio.open(raster) as src:
        px = abs(src.transform.a)
        tout = unary_union([g for _, g, _ in parcelles]).bounds
        inv = ~src.transform
        c0, r0 = inv * (tout[0], tout[3])
        c1, r1 = inv * (tout[2], tout[1])
        x0, y0 = max(0, int(c0) - 16), max(0, int(r0) - 16)
        x1, y1 = min(src.width, int(c1) + 16), min(src.height, int(r1) + 16)
        prob = probabilites(model, nc, src, x0, y0, x1, y1, dev)
        T = src.transform * rasterio.Affine.translation(x0, y0)
    carte, conf = prob.argmax(0).astype(np.uint8), prob.max(0)
    del prob
    invT = ~T
    for pid, g, t in parcelles:
        # fenêtre de la parcelle seule (rasteriser sur toute la feuille serait très lent)
        bx0, by0, bx1, by1 = g.bounds
        c0, r0 = invT * (bx0, by1)
        c1, r1 = invT * (bx1, by0)
        c0, r0 = max(0, int(c0) - 2), max(0, int(r0) - 2)
        c1, r1 = min(carte.shape[1], int(c1) + 3), min(carte.shape[0], int(r1) + 3)
        if c1 <= c0 or r1 <= r0:
            continue
        Tw = T * rasterio.Affine.translation(c0, r0)
        for geom, k, cf in decouper(g, carte[r0:r1, c0:c1], conf[r0:r1, c0:c1], Tw, px, a.lisse, a.min_m2, a.simplifie):
            feats.append({"parcelle": pid, "classe": classes[k], "type": TYPE_DE_CLASSE.get(k),
                          "confiance": round(cf, 3), "aire_m2": round(geom.area, 1), "geometry": geom})
    gdf = gpd.GeoDataFrame(feats, crs=f"EPSG:{C.SRID}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(a.out, layer="typologies", driver="GPKG")
    print(f"{len(gdf)} typologies -> {a.out}")
    print(gdf.groupby("classe")["aire_m2"].agg(["count", "sum"]).round(0).to_string())

    if a.verite:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from build_dataset import TYPES, classe as classe_de
        vt = [(C.geom_of(g), classe_de(t, t2, True)) for t, t2, g in cur.execute(
            "SELECT type, type2, ST_AsBinary(geom) FROM parcelles WHERE feuille_id = %s AND statut = 'valide' "
            "AND lien = 'ok' AND developpement IS NULL", (fid,))]
        vt = [(g, k) for g, k in vt if k is not None]
        from shapely import STRtree
        tot, bien = np.zeros(nc), np.zeros(nc)
        hs = list(gdf.geometry)
        ks = [classes.index(c) for c in gdf["classe"]]
        arbre = STRtree(hs)
        for g, k in vt:
            tot[k] += g.area
            for i in arbre.query(g):
                if ks[int(i)] == k:
                    bien[k] += hs[int(i)].intersection(g).area
        print("\nPart de la surface vraie bien typée (rappel surfacique) :")
        for k, c in enumerate(classes):
            if tot[k] > 0:
                print(f"  {c:12} {bien[k] / tot[k]:6.1%}  sur {tot[k] / 1e4:.2f} ha")
        _ = TYPES


if __name__ == "__main__":
    main()
