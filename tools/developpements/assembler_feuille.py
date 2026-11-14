#!/usr/bin/env python3
"""Assemble une feuille pour GénéFoncier : développements remis à leur place réelle.

Un « développement » est un agrandissement dessiné en marge de la feuille (souvent
au 1/1000 sur une feuille au 1/2000). L'annotation le laisse là où il est dessiné ;
ce script, lui, produit ce que GénéFoncier affiche :

  0. si l'annotation a marqué les développements (touche 0 de l'extension QGIS :
     dev_role / dev_groupe), ces marques font foi ; la détection automatique est
     quand même lancée et comparée aux marques (c'est sa mesure de qualité) ;
  1. détection : chaque bloc de parcelles validées est comparé aux polygones plus
     petits de la feuille (validés ou en attente) ; un bloc qui retrouve sa forme en
     réduction (recouvrement élevé) est un développement, et ce polygone son
     emplacement réel ;
  2. calage : similitude (échelle, rotation, position), refaite sans les parcelles
     de contexte (dessinées dans le développement mais hors de son contour réel),
     puis affinée sur les bords sans se laisser tirer par une parcelle manquante ;
  3. polygones : déplacés par cette similitude (l'intérieur garde la forme du
     dessin et reste superposé au raster), découpés sur le contour réel,
     interstices fins rendus au voisin : aucun chevauchement avec le plan
     principal ; une vraie zone vide (parcelle non validée dans le
     développement) est signalée dans la couche non_couvert, jamais comblée ;
  4. raster : le dessin du développement est rééchantillonné à son emplacement
     réel avec la même transformation, puis effacé de la marge (transparent) :
     un seul fichier par feuille.

Lecture seule sur la base d'annotation ; rien n'est écrit ailleurs que dans --out.

    .venv/bin/python tools/developpements/assembler_feuille.py --feuille 'Douai 1908 – Media 02' \\
        --out data/genefoncier_feuilles

Sorties : <feuille>.tif (GeoTIFF JPEG tuilé, masque interne, aperçus),
<feuille>.gpkg (couches parcelles, developpements, exclues, non_couvert),
<feuille>_controle.jpg, <feuille>.json (rapport).
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import rasterio
import shapely
from rasterio import features as rfeat
from rasterio.enums import Resampling
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from shapely import affinity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "annotation"))

MIN_PARCELLES = 5        # un développement compte au moins tant de parcelles
BLOC_JOINT_M = 1.5       # parcelles à moins de 2 x cette distance = même bloc
ECHELLE_MIN, ECHELLE_MAX = 1.4, 10.0   # rapport dessin / réel d'un développement
IOU_CANDIDAT = 0.80      # recouvrement minimal pour examiner un appariement
IOU_ACCEPTE = 0.95       # recouvrement minimal après calage sans le contexte
PART_GARDEE_MIN = 0.70   # le calage doit garder au moins cette part du bloc
DEDANS_MIN = 0.5         # une parcelle moins dedans que ça est du contexte
INTERSTICE_PX = 4.0      # interstice de largeur moyenne inférieure (px du scan, côté dessin) : rendu au voisin
TRONQUE_PX = 4.0         # calage sur les bords : un écart au-delà (px du scan) ne tire plus


# ----------------------------------------------------------------------------- géométrie
def slug(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60]


def parts(g):
    return [p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon" and not p.is_empty]


def outline(polys, joint=BLOC_JOINT_M):
    b = shapely.unary_union([g.buffer(joint) for g in polys]).buffer(-joint)
    return shapely.Polygon(max(parts(b), key=lambda g: g.area).exterior)


def blocks(polys, joint=BLOC_JOINT_M):
    """Groupes de parcelles contiguës : [(contour du bloc, indices des parcelles)]."""
    u = shapely.unary_union([g.buffer(joint) for g in polys]).buffer(-joint)
    comps = sorted(parts(u), key=lambda g: -g.area)
    reps = shapely.points([g.representative_point().coords[0] for g in polys])
    out = []
    for c in comps:
        shapely.prepare(c)
        idx = np.flatnonzero(shapely.contains(c, reps))
        if len(idx):
            out.append((shapely.Polygon(c.exterior), idx))
    return out


def _rect(g):
    c = np.array(g.minimum_rotated_rectangle.exterior.coords)[:4]
    e1, e2 = c[1] - c[0], c[2] - c[1]
    l1, l2 = np.hypot(*e1), np.hypot(*e2)
    if l1 < l2:
        e1, l1, l2 = e2, l2, l1
    return np.degrees(np.arctan2(e1[1], e1[0])), l1 / max(l2, 1e-9)


class Similitude:
    """y = s·R(x − o) + o + t  (paramètres : log s, angle en degrés, tx, ty)."""

    def __init__(self, p, o):
        self.p, self.o = [float(v) for v in p], (float(o[0]), float(o[1]))

    @property
    def s(self):
        return float(np.exp(self.p[0]))

    @property
    def angle(self):
        return ((self.p[1] + 180) % 360) - 180

    def _R(self):
        a = np.radians(self.p[1])
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    def fwd(self, c):
        o, t = np.array(self.o), np.array(self.p[2:4])
        return self.s * (c - o) @ self._R().T + o + t

    def inv(self, c):
        o, t = np.array(self.o), np.array(self.p[2:4])
        return (c - o - t) @ self._R() / self.s + o

    def geom(self, g):
        return shapely.transform(g, self.fwd)

    def matrix(self):
        """[a, b, d, e, xoff, yoff] de shapely.affinity.affine_transform."""
        M = self.s * self._R()
        off = self.fwd(np.zeros((1, 2)))[0]
        return [M[0, 0], M[0, 1], M[1, 0], M[1, 1], off[0], off[1]]


def _iou(a, b):
    u = a.union(b).area
    return a.intersection(b).area / u if u else 0.0


def fit(dev, target, p0=None):
    """Meilleure similitude de ``dev`` sur ``target`` (maximise le recouvrement)."""
    o = dev.centroid.coords[0]
    if p0 is None:
        a0, el = _rect(dev)
        a1, _ = _rect(target)
        s0 = np.sqrt(target.area / dev.area)
        starts = [[np.log(s0), a1 - a0 + add, target.centroid.x - o[0], target.centroid.y - o[1]]
                  for add in ((0, 180) if el > 1.5 else (0, 90, 180, 270))]
    else:
        starts = [list(p0)]
    best = (0.0, starts[0])
    for p in starts:
        f = lambda q: -_iou(Similitude(q, o).geom(dev), target)  # noqa: E731
        v0 = -f(p)
        if v0 < 0.5 and p0 is None:
            best = max(best, (v0, p), key=lambda t: t[0])
            continue
        simplex = np.array([p] + [np.array(p) + d for d in
                                  ([.02, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2])])
        r = minimize(f, p, method="Nelder-Mead",
                     options=dict(xatol=1e-4, fatol=1e-6, maxiter=1500, initial_simplex=simplex))
        best = max(best, (-r.fun, list(r.x)), key=lambda t: t[0])
    return best[0], Similitude(best[1], o)


def fit_edges(dev, target, sim, tau):
    """Affine le calage sur les BORDS, sans se laisser tirer par ce qui manque.

    Le recouvrement de surfaces est biaisé dès qu'une parcelle du développement n'a
    pas été validée (ou qu'une parcelle de contexte dépasse) : tout le bloc glisse
    pour compenser. Ici chaque point de bord compte par sa distance à l'autre
    contour, plafonnée à ``tau`` : le bord d'un manque, loin de tout, pèse un coût
    constant et ne déplace plus rien ; les bords communs, eux, se superposent."""
    o = sim.o
    pd = _sample(dev.exterior, max(0.2, dev.exterior.length / 600))
    pt = shapely.points(_sample(target.exterior, max(0.1, target.exterior.length / 600)))
    tb = target.exterior

    def cost(q):
        S = Similitude(q, o)
        moved = S.fwd(pd)
        d1 = np.minimum(shapely.distance(shapely.points(moved), tb), tau)
        d2 = np.minimum(shapely.distance(pt, shapely.LinearRing(moved)), tau)
        return float((d1 ** 2).mean() + (d2 ** 2).mean())

    p = list(sim.p)
    simplex = np.array([p] + [np.array(p) + d for d in
                              ([.01, 0, 0, 0], [0, .5, 0, 0], [0, 0, tau, 0], [0, 0, 0, tau])])
    r = minimize(cost, p, method="Nelder-Mead",
                 options=dict(xatol=1e-5, fatol=1e-9, maxiter=3000, initial_simplex=simplex))
    return Similitude(r.x, o) if r.fun <= cost(p) else sim


def _sample(line, step):
    n = max(8, int(line.length / step))
    return shapely.get_coordinates(shapely.line_interpolate_point(line, np.linspace(0, line.length, n, endpoint=False)))


def _proj(pts, line):
    return shapely.get_coordinates(shapely.line_interpolate_point(
        line, shapely.line_locate_point(line, shapely.points(pts))))


def border_gap(a, b, step=0.25):
    d = np.r_[shapely.distance(shapely.points(_sample(a, step)), b),
              shapely.distance(shapely.points(_sample(b, step)), a)]
    return dict(moyen=round(float(d.mean()), 2), p95=round(float(np.percentile(d, 95)), 2),
                max=round(float(d.max()), 2))


class Warp:
    """Déformation douce (plaque mince) qui colle le bord du bloc déplacé au contour
    réel ; les écarts au-delà de ``cap`` ne sont pas forcés. ``fwd`` pour les
    polygones, ``inv`` pour le raster. Inactive si ``cap`` vaut 0."""

    def __init__(self, moved_outline, target, cap):
        tb = target.exterior
        step = max(0.5, moved_outline.exterior.length / 500)
        P = _sample(moved_outline.exterior, step)
        Tc = np.array(tb.coords)[:-1]
        sp = np.r_[P, _proj(Tc, moved_outline.exterior)]
        dp = np.r_[_proj(P, tb), Tc]
        _, iu = np.unique(np.round(sp, 2), axis=0, return_index=True)
        sp, dp = sp[iu], dp[iu]
        ok = np.hypot(*(dp - sp).T) <= cap
        self.n_forces, self.n_laisses = int(ok.sum()), int((~ok).sum())
        self.active = cap > 0 and ok.sum() >= 8
        if self.active:
            self._f = RBFInterpolator(sp[ok], (dp - sp)[ok], kernel="thin_plate_spline", smoothing=0.5)

    @staticmethod
    def _apply(f, c):
        # mêmes points -> même résultat : anneaux fermés, voisins toujours collés
        uq, iv = np.unique(c, axis=0, return_inverse=True)
        return (uq + f(uq))[iv.ravel()]

    def fwd(self, c):
        return self._apply(self._f, c) if self.active else c

    def inv(self, c):
        """Inverse exact de ``fwd`` (point fixe) : le raster suit la MÊME déformation
        que les polygones, au centimètre."""
        if not self.active:
            return c
        x = c.copy()
        for _ in range(30):
            nx = c - self._f(x)
            if np.abs(nx - x).max() < 1e-3:
                x = nx
                break
            x = nx
        return x


# ----------------------------------------------------------------------------- détection
def detect(valides, candidats, res, log=print):
    """Développements d'une feuille.

    ``valides`` : parcelles validées (là où elles sont dessinées).
    ``candidats`` : [(origine, géométrie)] des emplacements réels possibles.
    Renvoie [dict(bloc, idx, garde, contexte, cible, cible_origine, sim, iou, ...)].
    """
    out = []
    used = set()
    for contour, idx in blocks(valides):
        if len(idx) < MIN_PARCELLES:
            continue
        inside = set(int(i) for i in idx)
        best = None
        for k, (orig, c) in enumerate(candidats):
            if k in used or c.area < 100:
                continue
            ratio = np.sqrt(contour.area / c.area)
            if not ECHELLE_MIN <= ratio <= ECHELLE_MAX:
                continue
            if orig[0] == "valide" and orig[1] in inside:
                continue
            iou, sim = fit(contour, c)
            if iou >= IOU_CANDIDAT and (best is None or iou > best[0]):
                best = (iou, sim, k)
        if best is None:
            continue
        iou0, sim, k = best
        orig, target = candidats[k]
        d = caler(valides, idx, target, sim, res)
        if d is None:
            continue
        ok = d["iou"] >= IOU_ACCEPTE and d["part_gardee"] >= PART_GARDEE_MIN
        log(f"  bloc de {len(idx)} parcelles ({contour.area / 1e4:.2f} ha) : recouvrement {100 * iou0:.1f} % -> "
            f"{100 * d['iou']:.1f} % sans {len(d['contexte'])} parcelle(s) de contexte, échelle 1/{1 / d['sim'].s:.2f}, "
            f"rotation {d['sim'].angle:+.2f}° -> {'DÉVELOPPEMENT' if ok else 'écarté'}")
        if ok:
            used.add(k)
            out.append(dict(d, contour=contour, cible_origine=orig, iou0=iou0, source="auto"))
    return out


def caler(valides, idx, target, sim, res, contexte=()):
    """Calage d'un bloc (indices ``idx``) sur son emplacement réel ``target``.

    Part de la similitude ``sim`` (ou la cherche si None), écarte les parcelles de
    contexte (moins de la moitié dans le contour réel), puis affine sur les bords.
    ``contexte`` : parcelles déjà connues comme contexte (marquées à l'annotation)."""
    garde = [int(i) for i in idx if int(i) not in set(contexte)]
    contexte = [int(i) for i in contexte]
    if len(garde) < 1:
        return None
    if sim is None:
        _, sim = fit(outline([valides[i] for i in garde]), target)
    for _ in range(8):
        moved = [sim.geom(valides[i]) for i in garde]
        drop = [j for j, m in enumerate(moved) if m.intersection(target).area / m.area < DEDANS_MIN]
        if not drop:
            break
        for j in sorted(drop, reverse=True):
            contexte.append(garde.pop(j))
        if not garde:
            return None
        _, sim = fit(outline([valides[i] for i in garde]), target, sim.p)
    # calage final sur les bords (insensible aux parcelles manquantes), puis
    # dernier tri du contexte avec cette position
    for _ in range(3):
        sim = fit_edges(outline([valides[i] for i in garde]), target, sim, TRONQUE_PX * res)
        moved = [sim.geom(valides[i]) for i in garde]
        drop = [j for j, m in enumerate(moved) if m.intersection(target).area / m.area < DEDANS_MIN]
        if not drop:
            break
        for j in sorted(drop, reverse=True):
            contexte.append(garde.pop(j))
        if not garde:
            return None
    iou = _iou(sim.geom(outline([valides[i] for i in garde])), target)
    tous = [int(i) for i in idx]
    kept = sum(valides[i].area for i in garde) / sum(valides[i].area for i in tous)
    return dict(idx=tous, garde=garde, contexte=contexte, cible=target, sim=sim, iou0=iou, iou=iou,
                part_gardee=kept, contour=outline([valides[i] for i in tous]))


def from_labels(valides, roles, groupes, emplacements, res, log=print):
    """Développements marqués à l'annotation (dev_role / dev_groupe) : les parcelles
    et l'emplacement réel sont donnés, seul le calage est calculé."""
    out = []
    for g in sorted({gr for r, gr in zip(roles, groupes) if r == "parcelle" and gr is not None}):
        idx = [i for i, (r, gr) in enumerate(zip(roles, groupes)) if gr == g and r in ("parcelle", "contexte")]
        ctx = [i for i in idx if roles[i] == "contexte"]
        cible = emplacements.get(g)
        if cible is None:
            log(f"  développement n° {g} : {len(idx)} parcelles marquées mais pas d'emplacement réel marqué -> ignoré")
            continue
        d = caler(valides, idx, cible, None, res, contexte=ctx)
        if d is None:
            log(f"  développement n° {g} : calage impossible -> ignoré")
            continue
        auto_ctx = len(d["contexte"]) - len(ctx)
        log(f"  développement n° {g} (marqué) : {len(idx)} parcelles, recouvrement {100 * d['iou']:.1f} %, "
            f"échelle 1/{1 / d['sim'].s:.2f}, rotation {d['sim'].angle:+.2f}°"
            + (f", {auto_ctx} parcelle(s) hors du contour réel écartée(s) en plus" if auto_ctx else "")
            + ("  ⚠ recouvrement faible : vérifier le marquage" if d["iou"] < IOU_ACCEPTE else ""))
        out.append(dict(d, cible_origine=("marque", g), source="annotation", groupe=g))
    return out


def compare(auto, marques, log=print):
    """La détection automatique retrouve-t-elle ce qui a été marqué à la main ?"""
    ok = 0
    for m in marques:
        best = max(auto, key=lambda a: len(set(a["idx"]) & set(m["idx"])), default=None)
        if best is None or not set(best["idx"]) & set(m["idx"]):
            log(f"  détection automatique : développement n° {m['groupe']} NON trouvé")
            continue
        j = len(set(best["garde"]) & set(m["garde"])) / len(set(best["garde"]) | set(m["garde"]))
        c = _iou(best["cible"], m["cible"])
        bon = j >= 0.9 and c >= 0.9
        ok += bon
        log(f"  détection automatique : développement n° {m['groupe']} {'retrouvé' if bon else 'DIFFÉRENT'} "
            f"(mêmes parcelles à {100 * j:.0f} %, même emplacement à {100 * c:.0f} %)")
    faux = [a for a in auto if not any(set(a["idx"]) & set(m["idx"]) for m in marques)]
    if faux:
        log(f"  détection automatique : {len(faux)} développement(s) trouvé(s) en plus, non marqué(s)")
    return dict(marques=len(marques), retrouves=ok, en_plus=len(faux))


# ----------------------------------------------------------------------------- polygones
def transport(dev, valides, cap, res, log=print):
    """Parcelles du développement à leur place réelle, sans chevauchement.

    Renvoie (polygones par indice de parcelle, zones non couvertes, stats, warp)."""
    sim, target = dev["sim"], dev["cible"]
    garde = dev["garde"]
    moved_outline = sim.geom(outline([valides[i] for i in garde]))
    warp = Warp(moved_outline, target, cap)
    geoms = np.array([valides[i] for i in garde], dtype=object)
    moved = shapely.transform(geoms, lambda c: warp.fwd(sim.fwd(c)))
    moved = [g if g.is_valid else g.buffer(0) for g in moved]
    avant = border_gap(moved_outline.exterior, target.exterior)
    apres = border_gap(outline(moved, 0.3).exterior, target.exterior)

    # découpe sur le contour réel, puis suppression des recouvrements internes
    # (une parcelle ne garde que ce que les précédentes n'ont pas pris : les plus
    # grandes d'abord, pour que les lamelles aillent aux petites)
    clipped = {}
    taken = None
    for j in sorted(range(len(garde)), key=lambda j: -moved[j].area):
        g = moved[j].intersection(target)
        if taken is not None:
            g = g.difference(taken)
        pp = [p for p in parts(g) if p.area > 0.05]
        if not pp:
            continue
        g = max(pp, key=lambda p: p.area)
        clipped[garde[j]] = g
        taken = g if taken is None else taken.union(g)
    perdues = [i for i in garde if i not in clipped]

    # interstices entre le bloc et le contour réel : les fins vont au voisin
    gaps = parts(target.difference(taken)) if taken is not None else [target]
    non_couvert, rendus = [], 0
    keys = list(clipped)
    for gap in gaps:
        if gap.area < 0.01:
            continue
        # largeur moyenne de l'interstice ; au-delà c'est une vraie zone vide (parcelle
        # non validée dans le développement, même étroite comme un sentier) : on la signale
        if 2 * gap.area / gap.length > INTERSTICE_PX * res * sim.s:
            non_couvert.append(gap)
            continue
        tree = shapely.STRtree([clipped[k] for k in keys])
        near = [keys[int(j)] for j in tree.query(gap.buffer(0.05))]
        if not near:
            non_couvert.append(gap)
            continue
        owner = max(near, key=lambda k: gap.buffer(0.05).intersection(clipped[k]).area)
        u = clipped[owner].union(gap.buffer(0.01)).buffer(-0.01).intersection(target)
        pp = parts(u)
        if pp:
            clipped[owner] = max(pp, key=lambda p: p.area)
            rendus += 1
    a0 = np.array([sim.geom(valides[i]).area for i in clipped])
    a1 = np.array([clipped[i].area for i in clipped])
    u = shapely.unary_union(list(clipped.values()))
    stats = dict(bord_avant=avant, bord_apres=apres, points_forces=warp.n_forces, points_laisses=warp.n_laisses,
                 interstices_rendus=rendus, parcelles_perdues=len(perdues),
                 variation_aire_mediane=round(float(np.median(abs(a1 / a0 - 1))), 4),
                 variation_aire_max=round(float(abs(a1 / a0 - 1).max()), 4),
                 chevauchement_interne_m2=round(float(sum(a1) - u.area), 3),
                 couverture_cible=round(float(u.area / target.area), 4),
                 non_couvert_m2=round(float(sum(g.area for g in non_couvert)), 1))
    return clipped, non_couvert, stats, warp


# ----------------------------------------------------------------------------- raster
def assemble_raster(raster, out_path, devs, warps, valides, facteur_max, log=print):
    """Feuille unique : chaque développement redessiné à sa place réelle, effacé de la marge."""
    with rasterio.open(raster) as src:
        img = np.ascontiguousarray(np.transpose(src.read([1, 2, 3]), (1, 2, 0)))
        mask = src.dataset_mask()
        T0, crs, res = src.transform, src.crs, abs(src.transform.a)
    k = 1
    if devs:
        k = int(min(facteur_max, max(1, round(max(1 / d["sim"].s for d in devs)))))
    H, W = img.shape[0] * k, img.shape[1] * k
    T = T0 * rasterio.Affine.scale(1 / k)
    if k > 1:
        out = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
        omask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        out, omask = img.copy(), mask.copy()
    inv0 = ~T0

    # 1. effacer les dessins de la marge (tout le bloc, parcelles de contexte comprises)
    for d in devs:
        zone = outline([valides[i] for i in d["idx"]]).buffer(2 * res)
        m = rfeat.rasterize([(zone, 1)], out_shape=(H, W), transform=T, dtype="uint8").astype(bool)
        out[m] = 0
        omask[m] = 0
    # 2. redessiner chaque développement à sa place réelle
    for d, warp in zip(devs, warps):
        sim, target = d["sim"], d["cible"]
        x0, y0, x1, y1 = target.bounds
        c0, r1 = (~T) * (x0, y0)
        c1, r0 = (~T) * (x1, y1)
        c0, r0 = max(0, int(np.floor(c0)) - 2), max(0, int(np.floor(r0)) - 2)
        c1, r1 = min(W, int(np.ceil(c1)) + 2), min(H, int(np.ceil(r1)) + 2)
        h, w = r1 - r0, c1 - c0
        # carte destination -> source, calculée sur une grille grossière puis interpolée
        g = 8
        gy, gx = np.arange(0, h + g, g), np.arange(0, w + g, g)
        GX, GY = np.meshgrid(gx, gy)
        wx, wy = T * (c0 + GX.ravel() + 0.5, r0 + GY.ravel() + 0.5)
        srcw = sim.inv(warp.inv(np.c_[wx, wy]))
        sx, sy = inv0 * (srcw[:, 0], srcw[:, 1])
        # interpolation bilinéaire exacte de la grille : le pixel j lit la maille j / g
        fx, fy = np.meshgrid(np.arange(w, dtype=np.float32) / g, np.arange(h, dtype=np.float32) / g)
        mapx = cv2.remap((np.asarray(sx) - 0.5).reshape(GX.shape).astype(np.float32), fx, fy, cv2.INTER_LINEAR)
        mapy = cv2.remap((np.asarray(sy) - 0.5).reshape(GX.shape).astype(np.float32), fx, fy, cv2.INTER_LINEAR)
        del fx, fy
        # pixels source par pixel de sortie : > 1 = réduction, on lisse d'abord
        ratio = 1 / (sim.s * k)
        xs0, xs1 = int(max(0, mapx.min() - 4)), int(min(img.shape[1], mapx.max() + 5))
        ys0, ys1 = int(max(0, mapy.min() - 4)), int(min(img.shape[0], mapy.max() + 5))
        crop = img[ys0:ys1, xs0:xs1]
        if ratio > 1.3:
            crop = cv2.GaussianBlur(crop, (0, 0), 0.45 * ratio)
        patch = cv2.remap(crop, mapx - xs0, mapy - ys0, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        Tw = T * rasterio.Affine.translation(c0, r0)
        m = rfeat.rasterize([(target, 1)], out_shape=(h, w), transform=Tw, dtype="uint8").astype(bool)
        out[r0:r1, c0:c1][m] = patch[m]
        omask[r0:r1, c0:c1][m] = 255
        log(f"  raster : {int(m.sum())} px redessinés à l'emplacement réel "
            f"({ratio:.2f} px du scan par px de sortie)")
    out[omask == 0] = 0

    prof = dict(driver="GTiff", dtype="uint8", count=3, width=W, height=H, crs=crs, transform=T,
                tiled=True, blockxsize=512, blockysize=512, compress="jpeg", photometric="ycbcr",
                jpeg_quality=85, BIGTIFF="IF_SAFER")
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(np.transpose(out, (2, 0, 1)))
            dst.write_mask(omask)
            dst.build_overviews([2, 4, 8, 16, 32], Resampling.average)
    return out, omask, T, k


# ----------------------------------------------------------------------------- contrôle
def _draw(img, inv, g, col, th):
    for p in parts(g):
        c = np.array(p.exterior.coords)
        x, y = inv * (c[:, 0], c[:, 1])
        cv2.polylines(img, [np.stack([x, y], 1).astype(np.int32)], True, col, th, cv2.LINE_AA)


def control_image(path, out, T, principales, transportees, devs):
    Wt = 1800

    def title(txt):
        t = np.full((56, Wt, 3), 255, np.uint8)
        cv2.putText(t, txt, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
        return t

    f = max(1, out.shape[1] // Wt)
    ov = cv2.resize(out, (out.shape[1] // f, out.shape[0] // f), interpolation=cv2.INTER_AREA)
    inv = ~(T * rasterio.Affine.scale(f))
    for g in principales:
        _draw(ov, inv, g, (0, 150, 0), 1)
    for g in transportees:
        _draw(ov, inv, g, (240, 120, 0), 1)
    for d in devs:
        _draw(ov, inv, d["cible"], (220, 0, 180), 2)
    rows = [title("Feuille assemblee - vert : plan principal | orange : parcelles transportees | magenta : emplacement reel"),
            cv2.resize(ov, (Wt, int(ov.shape[0] * Wt / ov.shape[1])))]
    for n, d in enumerate(devs, 1):
        x0, y0, x1, y1 = d["cible"].buffer(0.15 * np.sqrt(d["cible"].area) + 5).bounds
        c0, r1 = (~T) * (x0, y0)
        c1, r0 = (~T) * (x1, y1)
        c0, r0, c1, r1 = [int(max(0, v)) for v in (c0, r0, c1, r1)]
        z = out[r0:r1, c0:c1]
        sc = min(3.0, Wt / z.shape[1])
        z = np.ascontiguousarray(cv2.resize(z, None, fx=sc, fy=sc,
                                            interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC))
        zi = ~(T * rasterio.Affine.translation(c0, r0) * rasterio.Affine.scale(1 / sc))
        plain = z.copy()
        for g in transportees:
            _draw(z, zi, g, (240, 120, 0), 2)
        for g in principales:
            _draw(z, zi, g, (0, 150, 0), 2)
        for im, lab in ((plain, f"Developpement {n} : raster assemble seul"),
                        (z, f"Developpement {n} : raster assemble + polygones")):
            rows += [title(lab), cv2.resize(im, (Wt, int(im.shape[0] * Wt / im.shape[1])))]
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


# ----------------------------------------------------------------------------- principal
def run(feuille, out_dir, cap=0.0, facteur_max=2, controle_auto=True, log=print):
    import common as C
    import geopandas as gpd

    cur = C.connect().cursor()
    rows = cur.execute("select id, nom, raster from feuilles where nom like %s or id::text = %s order by nom",
                       (feuille, feuille)).fetchall()
    if len(rows) != 1:
        sys.exit(f"{len(rows)} feuille(s) pour « {feuille} » : préciser le nom exact ou l'identifiant.")
    fid, nom, raster = rows[0]
    log(f"{nom}")
    attrs, valides, roles, groupes = [], [], [], []
    emplacements = {}          # n° de développement -> emplacement réel marqué
    for pid, no, bis, typ, role, grp, wkb_ in cur.execute(
            "select id, parcelles_no, parcelles_bis, type, dev_role, dev_groupe, ST_AsBinary(geom) "
            "from parcelles where feuille_id=%s and statut='valide' order by id", (fid,)):
        for p in parts(shapely.from_wkb(wkb_)):
            p = p if p.is_valid else p.buffer(0)
            if role == "emplacement":          # fausse parcelle : jamais une parcelle de sortie
                if grp is not None:
                    emplacements[grp] = p if grp not in emplacements else emplacements[grp].union(p)
                continue
            attrs.append(dict(parcelle_id=str(pid), parcelles_no=no, parcelles_bis=bis, type=typ))
            valides.append(p)
            roles.append(role)
            groupes.append(grp)
    # un emplacement marqué mais resté hors des parcelles validées compte aussi
    for grp, wkb_ in cur.execute(
            "select dev_groupe, ST_AsBinary(geom) from parcelles_pending where feuille_id=%s "
            "and dev_role='emplacement' and dev_groupe is not null "
            "and review_status not in ('approved', 'corrected')", (fid,)):
        for p in parts(shapely.from_wkb(wkb_)):
            emplacements.setdefault(grp, p if p.is_valid else p.buffer(0))
    emplacements = {g: max(parts(e), key=lambda q: q.area) for g, e in emplacements.items()}
    candidats = [(("valide", i), g) for i, g in enumerate(valides)]
    candidats += [(("marque", g), e) for g, e in emplacements.items()]
    for pid, wkb_ in cur.execute("select id, ST_AsBinary(geom) from parcelles_pending "
                                 "where feuille_id=%s and review_status='pending'", (fid,)):
        candidats += [(("attente", str(pid)), p) for p in parts(shapely.from_wkb(wkb_))]
    marque = any(r == "parcelle" for r in roles)
    log(f"  {len(valides)} parcelles validées, {len(candidats) - len(valides)} contours candidats, "
        f"{'développements marqués à l\'annotation' if marque else 'aucun développement marqué : détection automatique'}")

    with rasterio.open(raster) as src:
        res = abs(src.transform.a)
    accord = None
    if marque:
        devs = from_labels(valides, roles, groupes, emplacements, res, log)
        if controle_auto:
            accord = compare(detect(valides, candidats, res, lambda *a: None), devs, log)
    else:
        devs = detect(valides, candidats, res, log)
    transportees, warps, non_couvert, exclues = {}, [], [], []
    report = dict(feuille=nom, feuille_id=str(fid), raster=raster,
                  source="annotation" if marque else "détection automatique",
                  accord_detection_automatique=accord, developpements=[])
    dev_of = {}
    for n, d in enumerate(devs, 1):
        clipped, trous, stats, warp = transport(d, valides, cap, res, log)
        warps.append(warp)
        for i, g in clipped.items():
            transportees[i] = g
            dev_of[i] = n
        non_couvert += [(n, g) for g in trous]
        exclues += [(n, i) for i in d["contexte"]]
        report["developpements"].append(dict(
            n=n, groupe=d.get("groupe"), source=d["source"], parcelles=len(d["idx"]), transportees=len(clipped), contexte=len(d["contexte"]),
            echelle=round(1 / d["sim"].s, 4), rotation_deg=round(d["sim"].angle, 3),
            recouvrement_initial=round(d["iou0"], 4), recouvrement=round(d["iou"], 4),
            cible=list(d["cible_origine"]), cible_m2=round(d["cible"].area, 1),
            matrice=[round(v, 8) for v in d["sim"].matrix()], **stats))
        log(f"  développement {n} : {len(clipped)} parcelles transportées, bord {stats['bord_avant']['max']} m -> "
            f"{stats['bord_apres']['max']} m au pire, {stats['interstices_rendus']} interstices rendus, "
            f"non couvert {stats['non_couvert_m2']} m², aire ±{100 * stats['variation_aire_mediane']:.1f} % (médiane)")

    # couche finale : plan principal + parcelles transportées
    in_dev = {i for d in devs for i in d["idx"]}
    # parcelles marquées « développement » dont le calage a échoué : ni transportées ni laissées en marge
    in_dev |= {i for i, r in enumerate(roles) if r in ("parcelle", "contexte")}
    cibles_valides = {d["cible_origine"][1] for d in devs if d["cible_origine"][0] == "valide"}
    recs = []
    for i, g in enumerate(valides):
        if i in in_dev or i in cibles_valides:
            continue
        recs.append(dict(attrs[i], origine="principal", developpement=None, geometry=g))
    principales = [r["geometry"] for r in recs]
    for i, g in transportees.items():
        recs.append(dict(attrs[i], origine="developpement", developpement=dev_of[i], geometry=g))
    # contrôle : rien de transporté ne recouvre le plan principal
    tree = shapely.STRtree(principales) if principales else None
    chev = 0.0
    for g in transportees.values():
        for j in (tree.query(g) if tree is not None else []):
            chev += g.intersection(principales[int(j)]).area
    report.update(parcelles_principales=len(principales), parcelles_transportees=len(transportees),
                  chevauchement_avec_plan_principal_m2=round(chev, 3))
    log(f"  chevauchement entre parcelles transportées et plan principal : {chev:.2f} m²")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / slug(nom)
    gpkg = base.with_suffix(".gpkg")
    gpkg.unlink(missing_ok=True)
    gpd.GeoDataFrame(recs, crs=2154).to_file(gpkg, layer="parcelles", driver="GPKG")
    if devs:
        gpd.GeoDataFrame(
            [dict(n=n, role=role, echelle=round(1 / d["sim"].s, 4), rotation_deg=round(d["sim"].angle, 3),
                  recouvrement=round(d["iou"], 4), geometry=g)
             for n, d in enumerate(devs, 1) for role, g in (("dessin", d["contour"]), ("emplacement_reel", d["cible"]))],
            crs=2154).to_file(gpkg, layer="developpements", driver="GPKG")
    if exclues:
        gpd.GeoDataFrame([dict(attrs[i], developpement=n, motif="contexte (hors du contour réel)",
                               geometry=valides[i]) for n, i in exclues], crs=2154
                         ).to_file(gpkg, layer="exclues", driver="GPKG")
    if non_couvert:
        gpd.GeoDataFrame([dict(developpement=n, aire_m2=round(g.area, 1), geometry=g) for n, g in non_couvert],
                         crs=2154).to_file(gpkg, layer="non_couvert", driver="GPKG")

    tif = base.with_suffix(".tif")
    out, omask, T, k = assemble_raster(raster, tif, devs, warps, valides, facteur_max, log)
    report.update(raster_sortie=str(tif), facteur_resolution=k, largeur=out.shape[1], hauteur=out.shape[0],
                  taille_mo=round(tif.stat().st_size / 1e6, 1))
    control_image(str(base) + "_controle.jpg", out, T, principales, list(transportees.values()), devs)
    base.with_suffix(".json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=float))
    log(f"  -> {tif} ({report['taille_mo']} Mo, x{k}), {gpkg.name}, {base.name}_controle.jpg")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feuille", required=True, help="nom exact (ou motif SQL LIKE donnant une seule feuille) ou identifiant")
    ap.add_argument("--out", default="data/genefoncier_feuilles")
    ap.add_argument("--cap", type=float, default=0.0,
                    help="0 (défaut) : similitude seule, les polygones gardent exactement la forme du dessin et "
                         "seul leur bord extérieur est ajusté au contour réel. > 0 : déformation douce de "
                         "l'intérieur aussi (déplacement maximal en m), appliquée à l'identique au raster")
    ap.add_argument("--facteur-max", type=int, default=2,
                    help="agrandissement maximal du raster pour garder le détail des développements "
                         "(2 = un 1/1000 sur une feuille au 1/2000 est conservé ; 1 = résolution du scan)")
    ap.add_argument("--sans-controle", action="store_true",
                    help="quand les développements sont marqués à l'annotation, ne pas lancer en plus la "
                         "détection automatique pour la comparer aux marques")
    a = ap.parse_args()
    run(a.feuille, a.out, a.cap, a.facteur_max, not a.sans_controle)


if __name__ == "__main__":
    main()
