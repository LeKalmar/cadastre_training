"""Briques communes du projet d'annotation (base PostGIS « cadastre_annotation »).

Le schéma est dans annotation_parcelles/schema.sql. Connexion par le service
PostgreSQL ``cadastre_annotation`` (~/.pg_service.conf + ~/.pgpass) : aucun mot
de passe dans le code ni dans le projet QGIS.
"""
import hashlib
import sys
import uuid
from pathlib import Path

import numpy as np

REPO = "/home/lekalmar/PyCharmMiscProject/Code/Geenfoncier/GeneFoncier/terrapartus"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "annotation_parcelles"
SERVICE = "service=cadastre_annotation"
SRID = 2154

# Contraste trait / papier : un point du contour est « sur un trait » s'il est plus
# sombre de CONTRAST_T niveaux que le plus clair des deux côtés, à SIDE_PX px
# perpendiculairement. Insensible au jaunissement et aux taches du papier.
CONTRAST_T, SIDE_PX, INK_NEAR, MIN_SAMPLES = 25, 6, 2, 30
LIEN_OK, LIEN_DOUTE = 0.75, 0.35

# Motifs de rejet dont les faux traits deviennent des négatifs renforcés (masque 50).
NEGATIVE_MOTIFS = ("bruit", "texte", "hachures", "double_trait", "marge_cadre", "sur_segmentation")


def connect(autocommit=False):
    import psycopg

    return psycopg.connect(SERVICE, autocommit=autocommit)


def uuid5(text):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, text))


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def wkb(geom):
    """Géométrie shapely -> WKB (à passer à ST_GeomFromWKB(%s, 2154))."""
    import shapely
    from shapely.geometry import MultiPolygon

    if geom.geom_type == "Polygon":
        geom = MultiPolygon([geom])
    return shapely.to_wkb(geom)


def geom_of(blob):
    import shapely

    return shapely.from_wkb(bytes(blob)) if blob is not None else None


def darkness(rgb):
    """(3,H,W) uint8 -> assombrissement uint8 et sa version max-filtrée (±INK_NEAR)."""
    from scipy import ndimage

    dark = ((765 - rgb.sum(axis=0, dtype=np.uint16)) // 3).astype(np.uint8)
    return dark, ndimage.maximum_filter(dark, size=2 * INK_NEAR + 1)


def line_contrast(dark, dark_on, transform, boundary, valid=None):
    """Part (0..1) des points de ``boundary`` posés sur un trait du scan, ou None
    si trop peu de points tombent sur le scan."""
    import shapely

    inv = ~transform
    res = abs(transform.a)
    H, W = dark.shape
    hits, n_ok = [], 0
    for part in getattr(boundary, "geoms", [boundary]):
        L = part.length
        n = int(L / res)
        if n < 4:
            continue
        dist = np.linspace(res, L - res, n)
        p = shapely.get_coordinates(shapely.line_interpolate_point(part, dist))
        a = shapely.get_coordinates(shapely.line_interpolate_point(part, dist - res))
        b = shapely.get_coordinates(shapely.line_interpolate_point(part, dist + res))
        t = b - a
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        nrm = np.column_stack([-t[:, 1], t[:, 0]]) * SIDE_PX * res

        def px(xy):
            c = (inv.a * xy[:, 0] + inv.b * xy[:, 1] + inv.c).astype(np.int64)
            r = (inv.d * xy[:, 0] + inv.e * xy[:, 1] + inv.f).astype(np.int64)
            return r, c, (r >= 0) & (r < H) & (c >= 0) & (c < W)

        r0, c0, k0 = px(p)
        r1, c1, k1 = px(p + nrm)
        r2, c2, k2 = px(p - nrm)
        k = k0 & k1 & k2
        if valid is not None:
            k[k] &= valid[r0[k], c0[k]]
        if not k.any():
            continue
        on = dark_on[r0[k], c0[k]].astype(np.int16)
        side = np.minimum(dark[r1[k], c1[k]], dark[r2[k], c2[k]]).astype(np.int16)
        hits.append((on - side) > CONTRAST_T)
        n_ok += int(k.sum())
    if n_ok < MIN_SAMPLES:
        return None
    return float(np.concatenate(hits).mean())


def lien_of(score):
    if score is None:
        return "a_verifier"
    return "ok" if score >= LIEN_OK else ("a_verifier" if score >= LIEN_DOUTE else "hors_feuille")


def paper_footprint(raster_path, step=8):
    """Emprise du papier (Polygon, CRS du raster) : zone valide sans fond noir."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import shapes
    from shapely.geometry import shape

    from georeferencing.inference import _paper_mask

    with rasterio.open(raster_path) as src:
        h, w = max(1, src.height // step), max(1, src.width // step)
        rgb = src.read([1, 2, 3], out_shape=(3, h, w), resampling=Resampling.average)
        valid = src.dataset_mask(out_shape=(h, w))
        T = src.transform * src.transform.scale(src.width / w, src.height / h)
    paper = _paper_mask(np.transpose(rgb, (1, 2, 0)), valid, block=2, open_px=4)
    polys = [shape(g) for g, v in shapes(paper.astype(np.uint8), mask=paper, transform=T) if v]
    if not polys:
        return None
    return max(polys, key=lambda g: g.area).simplify(abs(T.a)).buffer(0)


def active_model(cur):
    """(id, chemin) du modèle actif du registre, ou (None, None)."""
    row = cur.execute("SELECT id, chemin FROM modeles WHERE actif").fetchone()
    return (str(row[0]), row[1]) if row else (None, None)
