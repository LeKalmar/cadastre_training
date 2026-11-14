"""
Fonctions communes : appariement raster <-> feuille, rasterisation des contours,
mesure d'alignement et d'épaisseur d'encre. Dépendances : GDAL, numpy, PIL
(le python système installé avec QGIS suffit).
"""
import re
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr, osr
from PIL import Image, ImageFilter

gdal.UseExceptions(); ogr.UseExceptions()

YEAR_RE = re.compile(r"(?<!\d)(1[78]\d\d)(?!\d)")


def srs_of(wkt_or_epsg):
    s = osr.SpatialReference()
    if isinstance(wkt_or_epsg, int): s.ImportFromEPSG(wkt_or_epsg)
    else: s.ImportFromWkt(wkt_or_epsg)
    s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return s


# ---------------------------------------------------------------------------
# Raster
# ---------------------------------------------------------------------------
class Sheet:
    """Un GeoTIFF géoréférencé + son emprise utile (zones non noires)."""

    THUMB = 600

    def __init__(self, path):
        self.path = Path(path)
        self.ds = gdal.OpenEx(str(path), gdal.OF_RASTER)
        self.W, self.H = self.ds.RasterXSize, self.ds.RasterYSize
        self.gt = self.ds.GetGeoTransform()
        self.srs = srs_of(self.ds.GetProjection())
        self.res = abs(self.gt[1])
        m = YEAR_RE.findall(self.path.name) or YEAR_RE.findall(str(self.path.parent))
        self.year = int(m[-1]) if m else None
        self._content = None

    @property
    def content(self):
        """Masque booléen (miniature) des pixels portant de l'image (pas le fond noir/transparent)."""
        if self._content is None:
            tw = self.THUMB; th = max(1, round(tw * self.H / self.W))
            a = self.ds.ReadAsArray(buf_xsize=tw, buf_ysize=th).astype(np.int32)
            if a.ndim == 2: a = a[None]
            ok = a[:3].sum(0) > 45
            if a.shape[0] == 4: ok &= a[3] > 0
            self._content = ok
        return self._content

    def thumb_gt(self):
        th, tw = self.content.shape
        g = self.gt
        return (g[0], g[1] * self.W / tw, g[2], g[3], g[4], g[5] * self.H / th)

    def read(self, x, y, w, h):
        a = self.ds.ReadAsArray(x, y, w, h)
        if a.ndim == 2: a = np.repeat(a[None], 3, 0)
        rgb = a[:3].astype(np.uint8)
        valid = np.ones((h, w), bool) if a.shape[0] < 4 else a[3] > 0
        valid &= rgb.astype(np.int32).sum(0) > 45
        return rgb, valid

    def win_gt(self, x, y):
        g = self.gt
        return (g[0] + x * g[1], g[1], 0, g[3] + y * g[5], 0, g[5])


# ---------------------------------------------------------------------------
# Vecteurs
# ---------------------------------------------------------------------------
def mem_layer(geoms, srs, gtype=ogr.wkbMultiLineString):
    ds = ogr.GetDriverByName("MEM").CreateDataSource("")
    lyr = ds.CreateLayer("l", srs, gtype)
    for g in geoms:
        f = ogr.Feature(lyr.GetLayerDefn()); f.SetGeometry(g); lyr.CreateFeature(f)
    return ds, lyr


def burn(lyr, gt, w, h, srs, all_touched=False):
    """Rasterise une couche dans une grille (w,h,gt) -> bool."""
    mem = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    mem.SetGeoTransform(gt); mem.SetProjection(srs.ExportToWkt())
    opts = ["ALL_TOUCHED=TRUE"] if all_touched else []
    gdal.RasterizeLayer(mem, [1], lyr, burn_values=[1], options=opts)
    return mem.ReadAsArray().astype(bool)


def transform(geom, src, dst):
    g = geom.Clone(); g.Transform(osr.CoordinateTransformation(src, dst)); return g


# ---------------------------------------------------------------------------
# Morphologie (PIL, sans scipy/opencv)
# ---------------------------------------------------------------------------
def dilate(mask, r):
    if r <= 0: return mask.copy()
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.array(im.filter(ImageFilter.MaxFilter(2 * r + 1))) > 0


def erode(mask, r):
    if r <= 0: return mask.copy()
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.array(im.filter(ImageFilter.MinFilter(2 * r + 1))) > 0


def thick_line(core, width):
    """Trait d'épaisseur `width` px centré sur un squelette 1 px (disque discret)."""
    if width <= 1: return core.copy()
    r = (width - 1) / 2
    im = Image.fromarray(core.astype(np.uint8) * 255)
    if r == int(r):
        return np.array(im.filter(ImageFilter.MaxFilter(int(2 * r + 1)))) > 0
    # largeur paire : dilatation 2r+1 décalée d'un demi-pixel -> union de deux carrés
    k = int(2 * r) + 1  # ex: width 4 -> r=1.5 -> k=4 (PIL refuse les tailles paires)
    a = np.array(im.filter(ImageFilter.MaxFilter(k - 1))) > 0
    b = np.zeros_like(a); b[1:, 1:] = a[:-1, :-1]
    return a | b


# ---------------------------------------------------------------------------
# Mesures image <-> vecteur
# ---------------------------------------------------------------------------
def darkness(rgb):
    """Encre = sombre. Normalisé localement (fond du papier ~ 0)."""
    g = rgb.astype(np.float32).mean(0)
    d = 255.0 - g
    return d - np.median(d)


def best_shift(dark, core, max_shift=12):
    """Décalage (dy,dx) qui maximise l'encre sous le tracé vecteur (corrélation FFT).
    Retourne (dy, dx, gain) ; gain = contraste au pic / médiane des décalages."""
    h, w = dark.shape
    F = np.fft.rfft2(dark); G = np.fft.rfft2(core.astype(np.float32))
    cc = np.fft.irfft2(F * np.conj(G), s=(h, w)) / max(core.sum(), 1)
    ys = np.r_[0:max_shift + 1, h - max_shift:h]; xs = np.r_[0:max_shift + 1, w - max_shift:w]
    sub = cc[np.ix_(ys, xs)]
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    dy = int(ys[iy]); dx = int(xs[ix])
    dy = dy - h if dy > h // 2 else dy; dx = dx - w if dx > w // 2 else dx
    peak = sub.max(); base = np.median(sub)
    return dy, dx, float(peak - base), float(peak)


def shift(mask, dy, dx):
    out = np.zeros_like(mask)
    h, w = mask.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = mask[ys, xs]
    return out


def ink_profile(dark, core, kmax=10):
    """Encre moyenne en fonction de la distance (px) au tracé : anneaux de dilatation."""
    prof, prev, cur = [], np.zeros_like(core), core
    for k in range(kmax + 1):
        if k: cur = dilate(prev, 1)
        ring = cur & ~prev
        prof.append(float(dark[ring].mean()) if ring.any() else np.nan)
        prev = cur
    return np.array(prof)


def ink_width(prof):
    """Largeur de trait (px) = 2 * distance à mi-hauteur + 1."""
    peak, bg = prof[0], np.nanmedian(prof[-4:])
    if not np.isfinite(peak) or peak - bg < 5: return np.nan
    half = bg + (peak - bg) / 2
    for k in range(1, len(prof)):
        if prof[k] < half:
            # interpolation linéaire entre k-1 et k
            t = (prof[k - 1] - half) / max(prof[k - 1] - prof[k], 1e-6)
            return 2 * (k - 1 + t) + 1
    return np.nan


# ---------------------------------------------------------------------------
# Détection de lignes non annotées (crayon, pointillés de chemins...)
# ---------------------------------------------------------------------------
_RIDGE_CACHE = {}


def _ridge_kernels(shape, length=41, offset=4, n_angles=12):
    """Noyaux orientés : +1/L sur un segment, -1/(2L) sur deux segments parallèles à ±offset px.
    Réponse forte = trait fin et long (indépendant des aplats de couleur et du texte court)."""
    key = (shape, length, offset, n_angles)
    if key in _RIDGE_CACHE: return _RIDGE_CACHE[key]
    h, w = shape; ks = []
    t = np.arange(length) - length // 2
    for a in np.linspace(0, np.pi, n_angles, endpoint=False):
        k = np.zeros(shape, np.float32)
        c, s = np.cos(a), np.sin(a)
        for sign, wgt in ((0, 1.0 / length), (1, -0.5 / length), (-1, -0.5 / length)):
            xs = np.round(t * c - sign * offset * s).astype(int) % w
            ys = np.round(t * s + sign * offset * c).astype(int) % h
            np.add.at(k, (ys, xs), wgt)
        ks.append(np.fft.rfft2(k))
    _RIDGE_CACHE[key] = ks
    return ks


def line_response(dark, with_angle=False):
    """Réponse max sur les orientations (même unité que `darkness`) ; option : indice d'orientation."""
    F = np.fft.rfft2(dark.astype(np.float32))
    R = np.stack([np.fft.irfft2(F * k, s=dark.shape) for k in _ridge_kernels(dark.shape)])
    return (R.max(0), R.argmax(0)) if with_angle else R.max(0)


def unannotated_lines(dark, annotated, thr=7.0, min_len=80, max_round=0.3):
    """Traits longs et fins NON annotés (crayon, pointillés...) -> masque à ignorer.
    Garde les composantes allongées (périmètre/2 >= min_len, compacité 4πA/P² < max_round) :
    le texte et les taches, compacts, restent des exemples négatifs."""
    cand = (line_response(dark) > thr) & ~dilate(annotated, 4)
    cand = dilate(erode(cand, 1), 2)
    if not cand.any(): return cand
    h, w = cand.shape
    mem = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    mem.SetGeoTransform((0, 1, 0, 0, 0, -1)); mem.GetRasterBand(1).WriteArray(cand.astype(np.uint8))
    vds = ogr.GetDriverByName("MEM").CreateDataSource(""); vl = vds.CreateLayer("c", None, ogr.wkbPolygon)
    vl.CreateField(ogr.FieldDefn("v", ogr.OFTInteger))
    gdal.Polygonize(mem.GetRasterBand(1), mem.GetRasterBand(1), vl, 0, ["8CONNECTED=8"])
    keep = []
    for f in vl:
        g = f.GetGeometryRef(); a = g.GetArea(); p = g.Boundary().Length()
        if p / 2 >= min_len and 4 * np.pi * a / (p * p) < max_round:
            keep.append(g.Clone())
    if not keep: return np.zeros_like(cand)
    _ds, lyr = mem_layer(keep, None, ogr.wkbPolygon)
    return burn_plain(lyr, w, h)


def burn_plain(lyr, w, h):
    mem = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    mem.SetGeoTransform((0, 1, 0, 0, 0, -1))
    gdal.RasterizeLayer(mem, [1], lyr, burn_values=[1], options=["ALL_TOUCHED=TRUE"])
    return mem.ReadAsArray().astype(bool)


# ---------------------------------------------------------------------------
# Classement des marques d'encre non annotées : traits droits (à ignorer) vs texte/symboles
# (négatifs renforcés : le modèle doit apprendre à ne PAS tracer dessus)
# ---------------------------------------------------------------------------
def _components(mask):
    h, w = mask.shape
    mem = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    mem.SetGeoTransform((0, 1, 0, 0, 0, -1)); mem.GetRasterBand(1).WriteArray(mask.astype(np.uint8))
    vds = ogr.GetDriverByName("MEM").CreateDataSource(""); vl = vds.CreateLayer("c", None, ogr.wkbPolygon)
    vl.CreateField(ogr.FieldDefn("v", ogr.OFTInteger))
    gdal.Polygonize(mem.GetRasterBand(1), mem.GetRasterBand(1), vl, 0, ["8CONNECTED=8"])
    return [f.GetGeometryRef().Clone() for f in vl if f.GetField("v") == 1]


def _straight_thin(g, min_len=80):
    """Longueur = diamètre de l'enveloppe convexe ; largeur = aire(enveloppe) / longueur."""
    hull = g.ConvexHull(); ring = hull.GetGeometryRef(0)
    pts = np.array(ring.GetPoints())[:, :2] if ring and ring.GetPointCount() else np.zeros((0, 2))
    if len(pts) < 3: return False
    d = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1)).max()
    width = hull.GetArea() / max(d, 1)
    return d >= min_len and width <= max(16.0, 0.08 * d)


def _burn_geoms(geoms, shape):
    if not geoms: return np.zeros(shape, bool)
    _ds, lyr = mem_layer(geoms, None, ogr.wkbPolygon)
    return burn_plain(lyr, shape[1], shape[0])


def _coherent(comp_mask, resp, ang, thr, n_angles=12, min_len=60, min_coh=0.55, long_len=300, long_coh=0.45):
    """Composante 'linéaire' : longue et dont les pixels de crête partagent une orientation (±15°).
    Au-delà de `long_len` px (plus long qu'un mot), on tolère les croix/doubles traits des chemins."""
    ys, xs = np.nonzero(comp_mask)
    if len(ys) == 0: return False
    span = np.hypot(ys.max() - ys.min(), xs.max() - xs.min())
    if span < min_len: return False
    if len(ys) / span > 40: return False   # aplat hachuré (bâtiments) : surface, pas une ligne
    a = ang[comp_mask & (resp > thr)]
    if a.size < 20: return False
    h = np.bincount(a, minlength=n_angles).astype(float)
    best = max(h[i] + h[(i - 1) % n_angles] + h[(i + 1) % n_angles] for i in range(n_angles))
    return best / h.sum() >= (long_coh if span >= long_len else min_coh)


def classify_marks(dark, annotated, ridge_thr=7.0, ink_thr=30.0, min_text_px=40, max_text_px=5000):
    """Retourne (lignes_non_annotees, texte) — deux masques booléens.
    - lignes non annotées (crayon, pointillés) : composantes de crête longues et d'orientation cohérente
    - texte/symboles : encre franche de la taille d'un glyphe/mot, hors voisinage des lignes"""
    near_ann = dilate(annotated, 4)
    resp, ang = line_response(dark, with_angle=True)
    cand = dilate(erode((resp > ridge_thr) & ~near_ann, 1), 2)
    lines = np.zeros(dark.shape, bool)
    for g in (_components(cand) if cand.any() else []):
        cm = _burn_geoms([g], dark.shape) & cand
        if _coherent(cm, resp, ang, ridge_thr): lines |= cm
    ink = (dark > ink_thr) & ~near_ann & ~dilate(lines, 1)
    text = np.zeros(dark.shape, bool)
    if ink.any():
        # glyphes / chiffres / symboles ; les aplats (lavis) restent du fond simple
        comps = [g for g in _components(dilate(ink, 1)) if min_text_px <= g.GetArea() <= max_text_px]
        text = _burn_geoms(comps, dark.shape) & dilate(ink, 1) & ~near_ann
    return lines, text
