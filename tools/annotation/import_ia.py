#!/usr/bin/env python3
"""Fait tourner l'IA sur une feuille et dépose ses polygones dans parcelles_pending.

Rien n'entre dans la vérité terrain sans validation manuelle : chaque polygone
arrive « pending » ; c'est l'approbation dans QGIS qui le promeut en parcelle.

Chaîne = celle de extract_cadastre_task : U-Net (GPU) -> papier + polygonisation
nettoyée -> GRASS v.clean -> simplification finale de la couverture.
Une parcelle déjà validée sur la feuille n'est pas reproposée (> 50 % recouverte).

    # feuille existante (depuis QGIS, action « Lancer l'IA ») :
    .venv/bin/python tools/annotation/import_ia.py --feuille-id <uuid>
    # nouveau scan géoréférencé :
    .venv/bin/python tools/annotation/import_ia.py --raster scan.tif --nom "Douai 1908 A 2" \\
        [--commune Douai --annee 1908]
    # réutiliser un masque déjà calculé (pas de GPU) :
    ... --mask mask.tif
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_geom
from shapely import STRtree
from shapely.geometry import MultiPolygon, mapping, shape

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402
from georeferencing import grass_vectorize, inference  # noqa: E402

sys.path.insert(0, str(Path(__file__).parents[1] / "eval_models"))


def upscaled_copy(raster, target_res, out):
    """Copie agrandie (cubique) du scan quand il est plus grossier que ``target_res``
    (traits parallèles trop serrés en pixels pour le modèle, ex. Douai 0,25 m/px)."""
    from rasterio.enums import Resampling

    with rasterio.open(raster) as src:
        res = abs(src.transform.a)
        if not target_res or res <= target_res * 1.02:
            return raster, 1.0
        f = res / target_res
        H, W = int(round(src.height * f)), int(round(src.width * f))
        data = src.read(out_shape=(src.count, H, W), resampling=Resampling.cubic)
        prof = src.profile
        prof.pop("photometric", None)
        prof.update(width=W, height=H, transform=src.transform * rasterio.Affine.scale(src.width / W, src.height / H),
                    compress="deflate", tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
        msk = src.read_masks(1, out_shape=(H, W), resampling=Resampling.nearest) if src.count < 4 else None
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(data)
            if msk is not None and (msk == 0).any():
                dst.write_mask(msk)
    return str(out), f


def close_mask_gaps(mask_path, prob_path, out_path, mode):
    """Raccord des bouts de traits (gaps.py) avant polygonisation."""
    from gaps import close_gaps

    with rasterio.open(mask_path) as src:
        m = src.read(1) > 0
        valid = src.dataset_mask() > 0
        prof = src.profile
        res = abs(src.transform.a)
    with rasterio.open(prob_path) as src:
        p = src.read(1).astype(np.float32)
    out, st = close_gaps(m, p, res, valid=valid, low=None if mode == "link" else 0.12)
    prof.update(nodata=None)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write((out * 255).astype(np.uint8), 1)
            dst.write_mask(valid)
    return str(out_path), st


def simplify_coverage(polys, tol):
    """Simplification de la couverture entière : chaque limite commune est
    simplifiée une seule fois, les nœuds restent en place. Entrée rendue telle
    quelle en cas d'échec."""
    import shapely

    try:
        out = shapely.coverage_simplify(np.array(polys, dtype=object), tol)
    except Exception:
        return polys
    out = [g if g.is_valid else g.buffer(0) for g in out]
    return out if all(g.geom_type == "Polygon" and not g.is_empty for g in out) else polys


def ensure_paper_mask(mask_path, raster_path):
    """Un masque ancien (sans masque papier interne) reçoit celui du scan."""
    with rasterio.open(mask_path) as m:
        if m.nodata is None:
            return mask_path
        binary, prof = m.read(1), inference._mask_profile(m.profile)
    with rasterio.open(raster_path) as s:
        img = np.transpose(s.read([1, 2, 3]), (1, 2, 0))
        valid = s.dataset_mask()
    paper = inference._paper_mask(img, valid)
    del img
    binary[~paper] = 0
    outline = inference._footprint_outline(paper.astype(np.uint8) * 255, 3)
    if outline is not None:
        binary[outline] = 255
    prof["nodata"] = None
    out = Path(mask_path).with_name(Path(mask_path).stem + "_papier.tif")
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(binary, 1)
            dst.write_mask(paper.astype(np.uint8) * 255)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feuille-id")
    ap.add_argument("--raster")
    ap.add_argument("--nom")
    ap.add_argument("--commune")
    ap.add_argument("--annee", type=int)
    ap.add_argument("--model", help="checkpoint .pth (défaut : modèle actif du registre)")
    ap.add_argument("--mask", help="masque déjà calculé (saute le U-Net)")
    ap.add_argument("--seuil", type=float, default=0.30)
    ap.add_argument("--simplify-m", type=float, default=None,
                    help="simplification par arc en MÈTRES (ancien réglage : 0.3) ; 0 = aucune ; "
                         "non précisé = --simplify-arc-px")
    ap.add_argument("--simplify-arc-px", type=float, default=3.0,
                    help="simplification par arc (Douglas-Peucker sur chaque limite commune), tolérance en "
                         "PIXELS du scan : le trait mesure ~4 px à toute échelle. 3 px = meilleur réglage "
                         "sur les 10 feuilles de validation (eval du 11/11/2026)")
    ap.add_argument("--simplify-px", type=float, default=0.0,
                    help="ANCIENNE simplification finale coverage_simplify (px), garde trop de sommets ; 0 = non")
    ap.add_argument("--agrandir", type=float, default=0.0, metavar="M_PX",
                    help="agrandit le scan jusqu'à cette résolution avant l'IA (ex. 0.13 pour Douai) ; 0 = non")
    ap.add_argument("--trous", choices=["aucun", "link", "complet"], default="link",
                    help="fermeture des trous des traits : raccord des bouts (link), + hystérésis (complet)")
    ap.add_argument("--close-iter", type=int, default=0,
                    help="fermeture morphologique avant polygonisation (0 quand les trous sont fermés)")
    ap.add_argument("--jonction-px", type=float, default=6.0,
                    help="fusion des nœuds de jonction proches (px) ; 0 = auto")
    ap.add_argument("--remplacer", action="store_true",
                    help="supprime d'abord les propositions encore « pending » de la feuille")
    a = ap.parse_args()

    db = C.connect()
    cur = db.cursor()
    t0 = time.time()

    # --- feuille
    if a.feuille_id:
        row = cur.execute("select nom, raster from feuilles where id=%s", (a.feuille_id,)).fetchone()
        if not row:
            sys.exit(f"Feuille {a.feuille_id} introuvable.")
        nom, raster = row
        raster = a.raster or raster
        feuille_id = a.feuille_id
        if not raster:
            sys.exit("Cette feuille n'a pas de scan (champ raster).")
    else:
        if not a.raster or not a.nom:
            sys.exit("Donner --feuille-id, ou --raster et --nom pour une nouvelle feuille.")
        raster, nom = a.raster, a.nom
        fp = C.paper_footprint(raster)
        with rasterio.open(raster) as src:
            fp = shape(transform_geom(src.crs, "EPSG:2154", mapping(fp)))
        feuille_id = cur.execute(
            "insert into feuilles (nom, commune, annee, raster, statut, source, geom) values "
            "(%s,%s,%s,%s,'en_cours','ia',ST_GeomFromWKB(%s, 2154)) returning id",
            (nom, a.commune, a.annee, raster, C.wkb(fp))).fetchone()[0]
        db.commit()
        print(f"Feuille créée : {nom} ({feuille_id})", flush=True)

    with rasterio.open(raster) as src:
        if src.crs is None or src.crs.to_epsg() != 2154:
            sys.exit(f"Le scan doit être géoréférencé en EPSG:2154 (trouvé : {src.crs}).")

    # --- modèle
    model_id, model_path = C.active_model(cur)
    if a.model:
        model_path = a.model
        row = cur.execute("select id from modeles where chemin=%s", (model_path,)).fetchone()
        model_id = str(row[0]) if row else None
    if not a.mask and not model_path:
        sys.exit("Aucun modèle : passer --model ou en activer un dans la table modeles.")

    # --- U-Net + polygonisation
    work = C.WORK / "extractions" / str(feuille_id) / datetime.now().strftime("%Y%m%d-%H%M%S")
    work.mkdir(parents=True, exist_ok=True)
    seg_raster, facteur = raster, 1.0
    prob = None
    if a.mask:
        mask = ensure_paper_mask(a.mask, raster)
        seg = {"mask": a.mask}
    else:
        mask = str(work / "mask.tif")
        prob = str(work / "prob.tif")
        if a.agrandir:
            seg_raster, facteur = upscaled_copy(raster, a.agrandir, work / "scan_agrandi.tif")
            if facteur > 1:
                print(f"  scan agrandi x{facteur:.2f}", flush=True)

        def progress(done, total):
            if done == total or done % max(1, total // 10) == 0:
                print(f"  U-Net {done}/{total} tuiles", flush=True)

        seg = inference.segment_geotiff(seg_raster, mask, model_path,
                                        params=inference.InferenceParams(threshold=a.seuil),
                                        progress_cb=progress, prob_output_path=prob)
        seg = dict(seg or {}, agrandissement=round(facteur, 3))
    if prob and a.trous != "aucun":
        mask, st = close_mask_gaps(mask, prob, work / "mask_trous.tif", a.trous)
        seg["trous"] = st
        print(f"  trous : {st.get('linked', 0)} bouts raccordés", flush=True)
    if a.simplify_m is None:
        with rasterio.open(raster) as src:      # pixels du scan d'origine, pas du scan agrandi
            a.simplify_m = a.simplify_arc_px * abs(src.transform.a)
        print(f"  simplification : {a.simplify_arc_px:g} px = {a.simplify_m:.2f} m", flush=True)
    print("  polygonisation (GRASS)...", flush=True)
    pkw = dict(line_close_iter=a.close_iter)
    if a.simplify_m:
        pkw.update(simplify_method="arcs", simplify_dp_m=a.simplify_m)
    else:
        pkw["simplify_method"] = "coverage"
    if a.jonction_px:
        pkw["junction_merge_px"] = a.jonction_px
    feats, rapport = grass_vectorize.grass_vectorize_mask(mask, target_srid=2154, **pkw)
    for tmp in (work / "scan_agrandi.tif", work / "prob.tif"):
        tmp.unlink(missing_ok=True)
    polys = []
    origines = []
    for f in feats:
        g = shape(f["geometry"])
        for part in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
            polys.append(part)
            origines.append((f.get("properties") or {}).get("origin"))
    if a.simplify_m and polys:
        # GRASS (v.clean) peut rajouter des sommets : on repasse la simplification par arc
        from georeferencing.polygonize import _simplify_coverage_arcs
        polys = _simplify_coverage_arcs(polys, a.simplify_m)
    elif a.simplify_px and polys:
        res = abs(rasterio.open(mask).transform.a)   # px du masque (agrandi le cas échéant)
        simplified = simplify_coverage(polys, a.simplify_px * res)
        if len(simplified) == len(polys):
            polys = simplified

    # --- ne pas reproposer ce qui est déjà validé
    valides = [C.geom_of(r[0]) for r in cur.execute(
        "select ST_AsBinary(geom) from parcelles where feuille_id=%s and statut='valide' "
        "union all select ST_AsBinary(geom) from parcelles_pending where feuille_id=%s "
        "and review_status='pending' and geom_modifiee",
        (feuille_id, feuille_id))]
    skipped = 0
    if valides:
        tree = STRtree(valides)
        keep = []
        for g, o in zip(polys, origines):
            covered = sum(g.intersection(valides[int(j)]).area for j in tree.query(g))
            if covered > 0.5 * g.area:
                skipped += 1
            else:
                keep.append((g, o))
    else:
        keep = list(zip(polys, origines))

    # --- enregistrement
    if a.remplacer:
        n = cur.execute("delete from parcelles_pending where feuille_id=%s and review_status='pending' "
                        "and not geom_modifiee",   # une retouche manuelle n'est jamais jetée
                        (feuille_id,)).rowcount
        print(f"  {n} anciennes propositions « pending » supprimées", flush=True)
    extraction_id = cur.execute(
        "insert into extractions (feuille_id, modele_id, seuil, moteur, params, rapport, nb_propositions) "
        "values (%s,%s,%s,%s,%s,%s,%s) returning id",
        (feuille_id, model_id, a.seuil, rapport.get("engine"),
         json.dumps({"simplify_m": a.simplify_m, "simplify_arc_px": a.simplify_arc_px, "simplify_px": a.simplify_px, "model": model_path, "mask": mask,
                    "agrandir": a.agrandir, "trous": a.trous, "close_iter": a.close_iter,
                    "jonction_px": a.jonction_px}),
         json.dumps({"segmentation": seg, "polygonisation": rapport}, default=str),
         len(keep))).fetchone()[0]
    with cur.copy("copy parcelles_pending (feuille_id, extraction_id, origin, origine_grass, geom) "
                  "from stdin") as cp:
        for g, o in keep:
            multi = g if g.geom_type == "MultiPolygon" else MultiPolygon([g])
            cp.write_row((feuille_id, extraction_id, "ai", o, "SRID=2154;" + multi.wkt))
    cur.execute("update feuilles set statut='en_cours' where id=%s and statut='a_verifier'",
                (feuille_id,))
    db.commit()
    print(f"{nom} : {len(keep)} propositions à valider (extraction {extraction_id}), "
          f"{skipped} déjà validées non reproposées, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
