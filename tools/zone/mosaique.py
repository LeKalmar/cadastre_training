"""Mosaïque sans recouvrement des feuilles d'une commune (et d'une année), à partir des zones utiles.

Chaque zone est tracée fidèlement au dessin de SA feuille (feuille_zones). Entre feuilles voisines,
le géoréférencement laisse des recouvrements et des trous de quelques mètres : c'est ici qu'on fait
les compromis, automatiquement.
  1. Recouvrement de deux zones : la part couverte par les parcelles validées d'une seule des deux
     feuilles lui revient ; le reste est partagé par la ligne médiane (chaque point va à la feuille
     dans laquelle il est le plus profond : distance au bord de sa zone).
  2. Trous entre zones voisines : signalés (couche « trous »), non comblés.
  3. Avec --decouper : chaque plan est découpé par sa zone finale, comme les DCP (GeoTIFF RGBA,
     alpha = découpe), puis un VRT par commune et année (si GDAL est disponible).

Sortie : <out>/<commune>_<annee>.gpkg (couches zones_finales, recouvrements, trous) [+ TIF, VRT].
Par défaut seules les zones « valide » ; --avec-a-verifier pour un aperçu avec les propositions.

    .venv/bin/python tools/zone/mosaique.py --commune Lille --annee 1881 --out data/mosaiques [--decouper]
"""
import argparse
import re
import sys
import unicodedata
from itertools import combinations
from pathlib import Path

import numpy as np
from rasterio.features import rasterize, shapes
from rasterio.transform import from_origin
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "annotation"))
import common as C  # noqa: E402


def slug(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def mediane(za, zb, zone, pas):
    """Partage `zone` (partie commune de za et zb) : (part de a, part de b) par la ligne médiane."""
    x0, y0, x1, y1 = zone.buffer(pas * 4).bounds
    W, H = int(np.ceil((x1 - x0) / pas)), int(np.ceil((y1 - y0) / pas))
    if W * H > 40_000_000:                      # recouvrement énorme : pas de calcul fin
        return (zone, None) if za.area >= zb.area else (None, zone)
    T = from_origin(x0, y1, pas, pas)
    ma = rasterize([(za, 1)], out_shape=(H, W), transform=T, dtype="uint8") > 0
    mb = rasterize([(zb, 1)], out_shape=(H, W), transform=T, dtype="uint8") > 0
    mz = rasterize([(zone, 1)], out_shape=(H, W), transform=T, dtype="uint8") > 0
    da, db = ndimage.distance_transform_edt(ma), ndimage.distance_transform_edt(mb)
    va = (mz & (da >= db)).astype(np.uint8)
    pa = unary_union([shape(g) for g, v in shapes(va, mask=va > 0, transform=T)]) if va.any() else None
    pa = pa.intersection(zone) if pa is not None else None
    pb = zone.difference(pa) if pa is not None else zone
    return (pa if pa is not None and not pa.is_empty else None), (pb if not pb.is_empty else None)


def arbitrer(zones, parcelles, pas, min_m2):
    """zones {fid: geom} -> zones finales sans recouvrement, et couche des recouvrements traités."""
    fin = dict(zones)
    rec = []
    for a, b in combinations(sorted(zones), 2):
        za, zb = fin[a], fin[b]
        if not za.intersects(zb):
            continue
        o = za.intersection(zb)
        if o.area < min_m2:
            continue
        pa, pb = parcelles.get(a), parcelles.get(b)
        a_seul = o.intersection(pa).difference(pb) if pa is not None and pb is not None else (
            o.intersection(pa) if pa is not None else None)
        b_seul = o.intersection(pb).difference(pa) if pb is not None and pa is not None else (
            o.intersection(pb) if pb is not None else None)
        reste = o
        for g in (a_seul, b_seul):
            if g is not None and not g.is_empty:
                reste = reste.difference(g)
        ma, mb = mediane(za, zb, reste, pas) if not reste.is_empty and reste.area >= min_m2 else (None, None)
        part_a = [g for g in (a_seul, ma) if g is not None and not g.is_empty]
        part_b = [g for g in (b_seul, mb) if g is not None and not g.is_empty]
        part_a = unary_union(part_a) if part_a else None
        part_b = unary_union(part_b) if part_b else None
        # chacun garde sa part du recouvrement et cède celle de l'autre
        fin[a] = za.difference(part_b) if part_b is not None else za
        fin[b] = zb.difference(part_a) if part_a is not None else zb
        if part_a is None and part_b is None:      # recouvrement minuscule non partagé
            fin[b] = zb.difference(o)
        rec.append({"feuille_a": a, "feuille_b": b, "aire_m2": round(o.area, 1),
                    "par_parcelles_m2": round(sum(g.area for g in (a_seul, b_seul) if g is not None), 1),
                    "geometry": o})
    for k in fin:
        fin[k] = fin[k].buffer(0)
    return fin, rec


def trous(fin, pas_max_m):
    """Trous entre zones voisines : espaces de moins de pas_max_m de large, entourés de zones."""
    u = unary_union(list(fin.values()))
    ferme = u.buffer(pas_max_m).buffer(-pas_max_m)
    t = ferme.difference(u)
    return [g for g in getattr(t, "geoms", [t]) if not g.is_empty and g.area > 1.0]


def decouper(raster, zone, dest):
    import rasterio
    from rasterio.mask import mask as rmask
    with rasterio.open(raster) as src:
        rgb, T = rmask(src, [zone], crop=True, indexes=[1, 2, 3], filled=True, nodata=0)
        alpha = rasterize([(zone, 255)], out_shape=rgb.shape[1:], transform=T, dtype="uint8")
        prof = src.profile.copy()
        prof.update(count=4, height=rgb.shape[1], width=rgb.shape[2], transform=T, dtype="uint8",
                    compress="deflate", tiled=True, photometric="RGB", nodata=None)
        with rasterio.open(dest, "w", **prof) as dst:
            dst.write(rgb, [1, 2, 3])
            dst.write(alpha, 4)
            dst.colorinterp = [rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green,
                               rasterio.enums.ColorInterp.blue, rasterio.enums.ColorInterp.alpha]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commune", required=True)
    ap.add_argument("--annee", type=int)
    ap.add_argument("--out", default=str(C.ROOT / "data" / "mosaiques"))
    ap.add_argument("--avec-a-verifier", action="store_true")
    ap.add_argument("--pas", type=float, default=0.5, help="résolution (m) du partage médian")
    ap.add_argument("--min-m2", type=float, default=1.0)
    ap.add_argument("--trou-max-m", type=float, default=15.0)
    ap.add_argument("--decouper", action="store_true")
    a = ap.parse_args()

    import geopandas as gpd

    statuts = ["valide", "a_verifier"] if a.avec_a_verifier else ["valide"]
    cur = C.connect().cursor()
    q = ("SELECT DISTINCT ON (f.id) f.id, f.nom, coalesce(z.raster, f.raster), ST_AsBinary(z.geom) "
         "FROM feuille_zones z JOIN feuilles f ON f.id = z.feuille_id "
         "WHERE z.statut = ANY(%s) AND f.commune ILIKE %s")
    args = [statuts, a.commune]
    if a.annee:
        q += " AND f.annee = %s"
        args.append(a.annee)
    rows = cur.execute(q + " ORDER BY f.id, (z.statut = 'valide') DESC, z.modifie_le DESC", args).fetchall()
    if not rows:
        sys.exit("Aucune zone pour cette commune (zones validées dans QGIS : touches Z, V).")
    zones = {str(r[0]): C.geom_of(r[3]) for r in rows}
    info = {str(r[0]): (r[1], r[2]) for r in rows}
    parcelles = {}
    for fid in zones:
        g = cur.execute("SELECT ST_AsBinary(ST_Union(geom)) FROM parcelles WHERE feuille_id = %s "
                        "AND statut = 'valide' AND developpement IS NULL", (fid,)).fetchone()[0]
        if g:
            parcelles[fid] = C.geom_of(g)
    fin, rec = arbitrer(zones, parcelles, a.pas, a.min_m2)
    tr = trous(fin, a.trou_max_m)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = f"{slug(a.commune)}_{a.annee or 'toutes'}"
    gpkg = out / f"{base}.gpkg"
    crs = f"EPSG:{C.SRID}"
    gpd.GeoDataFrame([{"feuille_id": k, "feuille": info[k][0], "aire_ha": round(g.area / 1e4, 3),
                       "geometry": g} for k, g in fin.items()], crs=crs).to_file(gpkg, layer="zones_finales")
    if rec:
        gpd.GeoDataFrame(rec, crs=crs).to_file(gpkg, layer="recouvrements")
    if tr:
        gpd.GeoDataFrame([{"aire_m2": round(g.area, 1), "geometry": g} for g in tr], crs=crs).to_file(
            gpkg, layer="trous")
    print(f"{len(fin)} feuilles, {len(rec)} recouvrements arbitrés "
          f"({sum(r['aire_m2'] for r in rec):.0f} m²), {len(tr)} trous ({sum(g.area for g in tr):.0f} m²) -> {gpkg}")
    if a.decouper:
        tifs = []
        for k, g in fin.items():
            nom, raster = info[k]
            if not raster or not Path(raster).exists():
                print(f"  plan introuvable : {nom}")
                continue
            dest = out / base / f"MOS_{slug(nom)}.tif"
            dest.parent.mkdir(parents=True, exist_ok=True)
            decouper(raster, g, dest)
            tifs.append(str(dest))
            print(f"  découpé : {dest.name}", flush=True)
        try:
            from osgeo import gdal
            gdal.UseExceptions()
            gdal.BuildVRT(str(out / f"{base}.vrt"), tifs, options=gdal.BuildVRTOptions(addAlpha=False))
            print(f"VRT -> {out / f'{base}.vrt'}")
        except ImportError:
            print("GDAL absent de ce venv : VRT à faire avec .venv-georef "
                  f"(gdal.BuildVRT) ou QGIS (Raster > Divers > Construire un raster virtuel).")


if __name__ == "__main__":
    main()
