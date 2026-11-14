"""Communes du Nord : rapprochement dossier -> code INSEE actuel, et cadastre
actuel (parcelles Etalab) en Lambert 93, mis en cache.

Seuls le code du département et le code INSEE partent vers les services publics
(geo.api.gouv.fr, cadastre.data.gouv.fr).
"""
import gzip
import json
import os
import re
import unicodedata
import urllib.request

import numpy as np
from osgeo import osr

UA = {"User-Agent": "georef-cadastre"}
# communes absorbées : nom du dossier -> commune actuelle
ALIAS = {
    # noms des dossiers AD59 (cotes P 31) -> nom officiel actuel
    "aix": "aix-en-pevele", "aubry": "aubry-du-hainaut", "doulieu": "le-doulieu", "favril": "le-favril",
    "flamengrie": "la-flamengrie", "hellemmes-lille": "lille",
    "petite-synthe": "dunkerque", "rosendael": "dunkerque", "malo-les-bains": "dunkerque",
    "mardyck": "dunkerque", "flers-lez-lille": "villeneuve-d-ascq", "annappes": "villeneuve-d-ascq",
    "ascq": "villeneuve-d-ascq", "lomme": "lille", "hellemmes": "lille", "fives": "lille",
    "wazemmes": "lille", "esquermes": "lille", "moulins-lille": "lille",
    "coudekerque": "teteghem-coudekerque-village", "teteghem": "teteghem-coudekerque-village",
    "templeuve": "templeuve-en-pevele", "thivencelles": "conde-sur-l-escaut", "rombies-et-marchipont": "rombies-et-marchipont",
    "saint-pol-sur-mer": "dunkerque", "fort-mardyck": "dunkerque",
}

_T = None


def to_l93(lon, lat):
    global _T
    if _T is None:
        a, b = osr.SpatialReference(), osr.SpatialReference()
        a.ImportFromEPSG(4326)
        b.ImportFromEPSG(2154)
        a.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        b.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        _T = osr.CoordinateTransformation(a, b)
    pts = np.column_stack([np.ravel(lon), np.ravel(lat)])
    out = np.array(_T.TransformPoints(pts.tolist()))[:, :2]
    return out


def norm(name):
    s = name.lower().replace("œ", "oe").replace("æ", "ae")   # Mons-en-Barœul : la ligature n'a pas de forme NFD
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    s = re.sub(r"-sans-date$", "", s)
    m = re.match(r"^(.*)-(le|la|les|l)$", s)          # « doulieu-le » -> « le-doulieu »
    if m:
        s = f"{m.group(2)}-{m.group(1)}"
    return s


def _get(url, timeout=120):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def departement(cache, dep="59"):
    p = os.path.join(cache, f"communes_{dep}.json")
    if not os.path.exists(p):
        os.makedirs(cache, exist_ok=True)
        d = _get(f"https://geo.api.gouv.fr/departements/{dep}/communes?fields=nom,code&format=json")
        open(p, "wb").write(d)
    return {norm(c["nom"]): c for c in json.load(open(p))}


def resolve(folder, cache):
    """« 59211_Esquerchin » -> (code INSEE actuel, nom) ou None."""
    coms = departement(cache)
    name = norm(re.sub(r"^\d+\s*[_-]\s*", "", folder))
    name = ALIAS.get(name, name)
    for cand in (name, re.sub(r"^(le|la|les|l)-", "", name)):
        if cand in coms:
            return coms[cand]["code"], coms[cand]["nom"]
    return None


def parcels(code, cache):
    """Anneaux des parcelles actuelles (L93) : (liste d'array Nx2, bbox par anneau)."""
    p = os.path.join(cache, f"parcelles_{code}.npz")
    if not os.path.exists(p):
        url = ("https://cadastre.data.gouv.fr/data/etalab-cadastre/latest/geojson/communes/"
               f"{code[:2]}/{code}/cadastre-{code}-parcelles.json.gz")
        feats = json.loads(gzip.decompress(_get(url, 300)))["features"]
        rings = []
        for f in feats:
            g = f["geometry"]
            polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
            for poly in polys:
                for ring in poly:
                    rings.append(np.asarray(ring, float)[:, :2])
        allp = to_l93(np.concatenate(rings)[:, 0], np.concatenate(rings)[:, 1])
        lens = np.array([len(r) for r in rings])
        np.savez_compressed(p, pts=allp.astype(np.float64), lens=lens)
    d = np.load(p)
    rings = np.split(d["pts"], np.cumsum(d["lens"])[:-1])
    bb = np.array([[r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max()] for r in rings])
    return rings, bb


def l93_local_frame(E, N):
    """Échelle de L93 et gisement du nord vrai (rad, sens trigo) au point (E, N)."""
    a, b = osr.SpatialReference(), osr.SpatialReference()
    a.ImportFromEPSG(2154)
    b.ImportFromEPSG(4326)
    a.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    b.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    lon, lat, _ = osr.CoordinateTransformation(a, b).TransformPoint(float(E), float(N))
    dlat = 0.01
    p = to_l93([lon, lon], [lat - dlat, lat + dlat])
    v = p[1] - p[0]
    # longueur d'un arc de méridien de 2·dlat (ellipsoïde GRS80)
    a_, e2 = 6378137.0, 0.00669438002290
    phi = np.radians(lat)
    M = a_ * (1 - e2) / (1 - e2 * np.sin(phi) ** 2) ** 1.5
    k = float(np.hypot(*v) / (M * np.radians(2 * dlat)))
    gamma = float(np.arctan2(-v[0], v[1]))      # rotation trigo du nord vrai dans L93
    return k, gamma
