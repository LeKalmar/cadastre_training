"""Calage d'une commune : du système local du quadrillage vers le Lambert 93.

L'origine du système local est inconnue (clocher, point trigonométrique…). Chaque
feuille est corrélée au cadastre actuel pour ses 4 orientations possibles ; les
feuilles d'une même commune partagent origine et orientation : on retient
l'origine sur laquelle le plus de feuilles s'accordent, puis on ajuste une
similitude unique (origine + rotation, échelle = celle du Lambert 93).
"""
import itertools
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import fft as sfft
from scipy import ndimage
from scipy.spatial import ConvexHull

Image.MAX_IMAGE_PIXELS = None
RES = 2.0                                   # m/px de la recherche grossière
DELTAS = np.radians(np.arange(-1.5, 1.51, 0.5))
TOL = 25.0                                  # accord entre feuilles sur l'origine (m)


# ---------------------------------------------------------------- hypothèses
def hypotheses(points, fit):
    """Les 4 orientations non miroir : quelle famille porte les X (nord), signes."""
    hs = []
    px = np.array([[p["x"], p["y"], 1.0] for p in points])
    for xfam, sx, sy in itertools.product("ij", (1, -1), (1, -1)):
        yfam = "j" if xfam == "i" else "i"
        fx, fy = fit[xfam], fit[yfam]
        X = sx * np.array([fx["a"] + fx["step"] * p[xfam] for p in points], float)
        Y = sy * np.array([fy["a"] + fy["step"] * p[yfam] for p in points], float)
        loc = np.column_stack([Y, X])                      # (est, nord) local
        c, *_ = np.linalg.lstsq(px, loc, rcond=None)
        aff = c.T                                          # 2x3 : pixel -> local
        if np.linalg.det(aff[:, :2]) < 0:                  # y pixel vers le bas
            r = np.hypot(*(px @ c - loc).T)
            hs.append({"xfam": xfam, "sx": sx, "sy": sy, "loc": loc, "aff": aff,
                       "rms": float(np.sqrt((r ** 2).mean()))})
    return hs


# ---------------------------------------------------------------- rasters
def ink_mask(res, im):
    """Traits sombres fins à pleine résolution, dans l'enveloppe du quadrillage."""
    a = np.asarray(im).astype(np.float32)
    g = a.mean(axis=2)
    hp = ndimage.uniform_filter(g, 21) - g
    red = (a[..., 0] - a[..., 1]) > 30
    m = ndimage.binary_dilation((hp > 25) & ~red, iterations=2)
    pts = np.array([[p["x"], p["y"]] for p in res["points"]])
    hull = pts[ConvexHull(pts).vertices]
    c = hull.mean(axis=0)
    hull = c + (hull - c) * 1.04
    clip = Image.new("L", im.size, 0)
    ImageDraw.Draw(clip).polygon([tuple(p) for p in hull], fill=255)
    m &= np.asarray(clip) > 0
    return Image.fromarray((m * 255).astype(np.uint8))


def render(mask, C, res_m, bbox=None, pad=0.0):
    """Masque projeté par l'affine C (2x3, pixel -> mètres) sur une grille res_m.

    Renvoie (image float, (x0, y1)) : coin haut-gauche de la grille en mètres.
    """
    W, H = mask.size
    cs = np.array([[0, 0, 1], [W, 0, 1], [0, H, 1], [W, H, 1]], float) @ C.T
    if bbox is None:
        bbox = (cs[:, 0].min() - pad, cs[:, 1].min() - pad, cs[:, 0].max() + pad, cs[:, 1].max() + pad)
    ow, oh = int(np.ceil((bbox[2] - bbox[0]) / res_m)), int(np.ceil((bbox[3] - bbox[1]) / res_m))
    Mi = np.linalg.inv(C[:, :2])
    a = Mi @ np.array([res_m, 0.0])
    b = Mi @ np.array([0.0, -res_m])
    c = Mi @ (np.array([bbox[0], bbox[3]]) - C[:, 2])
    f = max(1, int(np.hypot(*a) // 1.5))
    small = mask.reduce(f) if f > 1 else mask
    w = small.transform((ow, oh), Image.AFFINE,
                        (a[0] / f, b[0] / f, c[0] / f, a[1] / f, b[1] / f, c[1] / f),
                        resample=Image.BILINEAR, fillcolor=0)
    return np.asarray(w).astype(np.float32) / 255.0, (bbox[0], bbox[3])


def modern(rings, bb, bbox, res_m):
    """Limites des parcelles actuelles rastérisées."""
    ow, oh = int(np.ceil((bbox[2] - bbox[0]) / res_m)), int(np.ceil((bbox[3] - bbox[1]) / res_m))
    img = Image.new("L", (ow, oh), 0)
    d = ImageDraw.Draw(img)
    sel = np.nonzero((bb[:, 2] > bbox[0]) & (bb[:, 0] < bbox[2]) &
                     (bb[:, 3] > bbox[1]) & (bb[:, 1] < bbox[3]))[0]
    for k in sel:
        r = rings[k]
        u = (r[:, 0] - bbox[0]) / res_m
        v = (bbox[3] - r[:, 1]) / res_m
        d.line(list(zip(u.tolist(), v.tolist())), fill=255, width=1)
    return (np.asarray(img) > 0).astype(np.float32)


_WORKERS = os.cpu_count() or 4


def gauss(x, sigma):
    """Identique à ndimage.gaussian_filter(x, sigma) (mode reflect, truncate 4) à 1e-7 près,
    mais par FFT sur tous les cœurs : 5x plus rapide dès sigma >= 4 (ndimage est mono-cœur
    et son coût croît avec la largeur du noyau). Petits sigma : ndimage, plus rapide."""
    r = int(4.0 * float(sigma) + 0.5)
    if sigma < 4 or min(x.shape) <= r:
        return ndimage.gaussian_filter(x, sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = np.pad(x, r, mode="symmetric").astype(np.float32)   # « symmetric » numpy = « reflect » scipy
    for ax in (0, 1):
        L = sfft.next_fast_len(out.shape[ax] + 2 * r, real=True)
        F = sfft.rfft(out, n=L, axis=ax, workers=_WORKERS)
        kk = sfft.rfft(k, n=L).reshape((-1, 1) if ax == 0 else (1, -1))
        y = sfft.irfft(F * kk, n=L, axis=ax, workers=_WORKERS)
        out = y[r:r + out.shape[0]] if ax == 0 else y[:, r:r + out.shape[1]]
    return out[r:-r, r:-r].astype(x.dtype, copy=False)


def bp(x):
    """Passe-bande + normalisation locale : une zone dense ne domine pas."""
    x = gauss(x, 1.5) - gauss(x, 8)
    # max(…, 0) : la FFT laisse des -1e-9 là où ndimage donnait 0, et sqrt(-1e-9) = NaN
    return x / (np.sqrt(np.maximum(gauss(x * x, 25), 0)) + 1e-3)


def _peaks(cc, n=3, rad=12):
    mu, sd = float(cc.mean()), float(cc.std()) + 1e-9
    cc = cc.copy()
    out = []
    for _ in range(n):
        k = np.unravel_index(int(np.argmax(cc)), cc.shape)
        out.append((int(k[0]), int(k[1]), float((cc[k] - mu) / sd)))
        cc[max(0, k[0] - rad):k[0] + rad + 1, max(0, k[1] - rad):k[1] + rad + 1] = -1e9
    return out


class Commune:
    """Cadastre actuel de la commune, prêt pour la corrélation."""

    def __init__(self, rings, bb, pad=400.0):
        self.rings, self.bb = rings, bb
        self.bbox = (bb[:, 0].min() - pad, bb[:, 1].min() - pad, bb[:, 2].max() + pad, bb[:, 3].max() + pad)
        self.B = modern(rings, bb, self.bbox, RES)
        self.b = bp(self.B)
        self._F = {}

    def FB(self, shape):
        if shape not in self._F:
            self._F = {shape: sfft.rfft2(self.b, s=shape, workers=8)}
        return self._F[shape]

    @property
    def shape(self):
        Hb, Wb = self.b.shape
        return (sfft.next_fast_len(Hb, real=True), sfft.next_fast_len(Wb, real=True))

    def cc(self, A):
        """Corrélation de A (coin haut-gauche) sur toute la commune, ou None si A déborde."""
        ha, wa = A.shape
        Hb, Wb = self.b.shape
        if ha >= Hb or wa >= Wb:
            return None
        a = bp(A) * (gauss(A, 10) > 0.01)
        a /= np.linalg.norm(a) + 1e-9
        cc = sfft.irfft2(np.conj(sfft.rfft2(a, s=self.shape, workers=8)) * self.FB(self.shape),
                         s=self.shape, workers=8)
        return cc[:Hb - ha + 1, :Wb - wa + 1]

    def search(self, A):
        """Meilleures positions du coin haut-gauche de A dans la commune."""
        cc = self.cc(A)
        return [] if cc is None else _peaks(cc)

    @property
    def territoire(self):
        """Zone couverte par les parcelles actuelles (à ~100 m près)."""
        if not hasattr(self, "_T"):
            self._T = (gauss(self.B, 25) > 0.02).astype(np.float32)
        return self._T

    def correle(self, a, key, X):
        """Corrélation de a avec le raster X de la commune pour tous les décalages, y compris
        quand a déborde de l'emprise : r[v + ha - 1, u + wa - 1] vaut pour le coin de a en (v, u).
        Une feuille de bord, avec ses marges blanches, garde ainsi sa vraie position."""
        ha, wa = a.shape
        Hb, Wb = X.shape
        shape = (sfft.next_fast_len(Hb + ha - 1, real=True), sfft.next_fast_len(Wb + wa - 1, real=True))
        if not hasattr(self, "_S"):
            self._S = {}
        if (key, shape) not in self._S:
            if len(self._S) > 4:
                self._S.clear()
            self._S[(key, shape)] = sfft.rfft2(X, s=shape, workers=8)
        r = sfft.irfft2(np.conj(sfft.rfft2(a, s=shape, workers=8)) * self._S[(key, shape)], s=shape, workers=8)
        return np.roll(r, (ha - 1, wa - 1), axis=(0, 1))[:Hb + ha - 1, :Wb + wa - 1]


def dessin(A):
    """Parcellaire dense d'une feuille rendue : les croix du quadrillage, le cadre et les
    écritures isolées n'y entrent pas (deux feuilles voisines bien placées : 4-9 % communs).
    L'encre est d'abord seuillée par rapport à celle de la feuille : au 1/500, rendue à 2 m/px,
    elle est très diluée et un seuil absolu ne garde rien."""
    nz = A[A > 0]
    if not nz.size:
        return np.zeros_like(A, np.float32)
    Ab = (A > 0.25 * np.percentile(nz, 90)).astype(np.float32)
    return (gauss(Ab, 20) > 0.2).astype(np.float32)


class Occupation:
    """Dessin des feuilles déjà placées, sur la grille de la commune : les feuilles d'un
    même levé pavent la commune (sauf les développements à grande échelle)."""

    def __init__(self, com):
        self.com = com
        self.O = np.zeros(com.b.shape, np.float32)
        self.version = 0

    def ajoute(self, mask, C):
        A, _ = render(mask, C, RES, bbox=self.com.bbox)
        D = dessin(A)[:self.O.shape[0], :self.O.shape[1]]
        self.O[:D.shape[0], :D.shape[1]] = np.maximum(self.O[:D.shape[0], :D.shape[1]], D)
        self.version += 1


def candidates(mask, hyps, com, k93, gamma):
    """Origines possibles (L93 du point local 0,0) pour chaque orientation et rotation."""
    out = []
    for hi, h in enumerate(hyps):
        for di, d in enumerate(DELTAS):
            t = gamma + d
            M = k93 * np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            A, (x0, y1) = render(mask, M @ h["aff"], RES)
            for v, u, z in com.search(A):
                O = (com.bbox[0] + u * RES - x0, com.bbox[3] - v * RES - y1)
                out.append({"h": hi, "d": di, "O": O, "z": z})
    return out


def consensus(cands):
    """cands : {feuille: [candidats]}. Renvoie (indice de rotation, origine, élus)."""
    best = None
    for di in range(len(DELTAS)):
        flat = [(s, c) for s, cs in cands.items() for c in cs if c["d"] == di]
        if not flat:
            continue
        Os = np.array([c["O"] for _, c in flat])
        for k in range(len(flat)):
            near = np.hypot(*(Os - Os[k]).T) < TOL
            per = {}
            for (s, c), ok in zip(flat, near):
                if ok and (s not in per or c["z"] > per[s]["z"]):
                    per[s] = c
            score = sum(min(c["z"], 12.0) for c in per.values())
            if best is None or score > best[0]:
                best = (score, di, per)
    if best is None:
        return None
    _, di, per = best
    w = np.array([c["z"] for c in per.values()])
    O = (np.array([c["O"] for c in per.values()]) * w[:, None]).sum(axis=0) / w.sum()
    return di, O, per


# ---------------------------------------------------------------- similitude
class Sim:
    """L93 = O + k·R(theta)·(est, nord) local."""

    def __init__(self, O, theta, k):
        self.O, self.theta, self.k = np.asarray(O, float), float(theta), float(k)

    @property
    def M(self):
        c, s = np.cos(self.theta), np.sin(self.theta)
        return self.k * np.array([[c, -s], [s, c]])

    def __call__(self, loc):
        return self.O + np.asarray(loc, float) @ self.M.T

    def affine(self, aff):
        C = self.M @ aff
        C[:, 2] += self.O
        return C


def fit_sim(loc, l93, w, k, theta0, free_rot, free_scale=False):
    """Similitude pondérée. Rotation et échelle ne sont libérées que si les
    feuilles d'appui sont assez nombreuses et écartées pour les déterminer."""
    loc, l93, w = np.asarray(loc, float), np.asarray(l93, float), np.asarray(w, float)
    w = w / w.sum()
    cl, cm = (loc * w[:, None]).sum(0), (l93 * w[:, None]).sum(0)
    theta = theta0
    a, b = loc - cl, l93 - cm
    if free_rot:
        num = (w * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])).sum()
        den = (w * (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])).sum()
        theta = float(np.arctan2(num, den))
    if free_scale:
        c, s_ = np.cos(theta), np.sin(theta)
        ar = a @ np.array([[c, -s_], [s_, c]]).T
        k = float(np.clip((w * (ar * b).sum(1)).sum() / (w * (a * a).sum(1)).sum(), k * 0.985, k * 1.015))
    s = Sim((0, 0), theta, k)
    s.O = cm - cl @ s.M.T
    return s


def ransac(cands, centres, k93, gamma, tol=15.0):
    """Similitude commune aux feuilles, cherchée sur les positions proposées.

    Chaque candidat donne où tombe le centre de sa feuille en L93 (indépendant de
    la rotation d'essai). Deux feuilles suffisent à définir une similitude ; on
    garde celle que le plus de feuilles confirment. L'origine peut être loin de
    la commune (système d'arrondissement) : rien n'est supposé sur elle.
    Renvoie (Sim, {feuille: orientation}) ou None.
    """
    P = []
    for n, cs in cands.items():
        seen = []
        for c in sorted(cs, key=lambda c: -c["z"]):
            t = gamma + DELTAS[c["d"]]
            M = k93 * np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            loc = centres[n][c["h"]]
            m = np.array(c["O"]) + M @ loc
            if any(h == c["h"] and np.hypot(*(m - mm)) < 6 for h, mm in seen):
                continue
            seen.append((c["h"], m))
            P.append((n, c["h"], loc, m, c["z"]))
    if not P:
        return None
    sh = np.array([p[0] for p in P])
    hh = np.array([p[1] for p in P])
    L = np.array([p[2] for p in P])
    Mm = np.array([p[3] for p in P])
    Z = np.minimum(np.array([p[4] for p in P]), 12.0)
    Lc, Mc = L[:, 0] + 1j * L[:, 1], Mm[:, 0] + 1j * Mm[:, 1]
    sheets = sorted(set(sh.tolist()))
    best = (0.0, None)
    order = np.argsort(-Z)
    for ia in order:
        for ib in order:
            if ib <= ia or sh[ia] == sh[ib]:
                continue
            dl, dm = Lc[ib] - Lc[ia], Mc[ib] - Mc[ia]
            if abs(dl) < 300:
                continue
            q = dm / dl                                   # échelle·e^(i·rotation)
            dth = (np.angle(q) - gamma + np.pi) % (2 * np.pi) - np.pi
            if abs(abs(q) / k93 - 1) > 0.012 or abs(dth) > np.radians(3):
                continue
            err = np.abs(Mc[ia] + q * (Lc - Lc[ia]) - Mc)
            inl = err < tol
            score = 0.0
            for s_ in sheets:
                zz = Z[inl & (sh == s_)]
                if zz.size:
                    score += zz.max()
            if score > best[0]:
                best = (score, (ia, q, inl))
    if best[1] is None:
        return None
    ia, q, inl = best[1]
    per = {}
    for k in np.nonzero(inl)[0]:
        n = int(sh[k])
        if n not in per or Z[k] > Z[per[n]]:
            per[n] = int(k)
    idx = list(per.values())
    span = np.hypot(*(L[idx].max(0) - L[idx].min(0)))
    sim = fit_sim(L[idx], Mm[idx], Z[idx], k93, float(np.angle(q)), free_rot=True)
    return sim, {n: int(hh[k]) for n, k in per.items()}


# ---------------------------------------------------------------- grille de la commune
W_COTES, W_RECOUV, W_DEHORS = 1.5, 20.0, 10.0
TOL_RECOUV, TOL_DEHORS = 0.3, 0.25        # bien placées : recouvrement ≤ 21 %, dehors ≤ ~10 %
# 99,1 % des 1377 feuilles « ok » du Nord ont le nord à moins de 90° du haut de la page : près de
# l'origine, où des cotes sans signe admettent 4 positions symétriques, cela en écarte deux
W_RETOURNEE = 8.0


def nord_vs_haut(M, h):
    """Angle (degrés) entre le haut de la page et le nord, pour l'orientation h."""
    up = M @ h["aff"][:, :2] @ np.array([0.0, -1.0])
    return float(np.degrees(np.arctan2(up[0], up[1])))


def _cotes_1d(readings, sign, st, offs):
    """Score des cotes d'une famille de lignes pour chaque décalage (multiple du pas)."""
    from labels import _line_score
    tot = np.zeros(len(offs))
    ok = np.zeros(len(offs), int)
    for k, vals in readings.items():
        sc = np.array([_line_score(abs(int(round(sign * st * k + o))), vals) for o in offs])
        tot += sc
        ok += sc >= 0.6
    return tot, ok


def sur_grille(mask, points, ocr, sim, com, steps, occ=None, top=8, rad_m=10.0):
    """Toutes les positions possibles d'une feuille sur le quadrillage de la commune.

    Le quadrillage de la commune est connu (sim) : la feuille ne peut tomber qu'à un
    nombre entier de pas de son zéro provisoire, dans l'une de ses 4 orientations. Au
    lieu de partir des seuls pics de corrélation, chaque position est notée :
      cotes   : cotes lues (même en partie, même avec des lectures concurrentes) expliquées ;
      z       : corrélation avec le cadastre actuel (carte FFT sur toute la commune) ;
      recouv  : part du dessin qui recouvre une feuille déjà placée (un levé pave la commune) ;
      dehors  : part du dessin hors du territoire couvert par le cadastre actuel.
    Renvoie les meilleures [{score, h, cotes_ok, cotes_lues, z, recouv, dehors}].
    """
    readings = {fk: {int(k): v for k, v in ocr[fk].items() if v} for fk in "ij"}
    Mi = np.linalg.inv(sim.M)
    bx = np.array([[com.bbox[0], com.bbox[1]], [com.bbox[2], com.bbox[1]],
                   [com.bbox[0], com.bbox[3]], [com.bbox[2], com.bbox[3]]])
    lb = (bx - sim.O) @ Mi.T                                  # emprise de la commune, en local
    lo, hi = lb.min(0), lb.max(0)
    rad = int(round(rad_m / RES))
    out = []
    for st in steps:
        zero = {"i": {"a": 0, "step": st}, "j": {"a": 0, "step": st}}
        for h in hypotheses(points, zero):
            A, (x0, y1) = render(mask, sim.M @ h["aff"], RES)
            ha, wa = A.shape
            a = bp(A) * (gauss(A, 10) > 0.01)
            a /= np.linalg.norm(a) + 1e-9
            cc = com.correle(a, "b", com.b)
            Z = ndimage.maximum_filter((cc - cc.mean()) / (cc.std() + 1e-9), size=2 * rad + 1)
            D = dessin(A)
            nD = D.sum() + 1e-9
            dehors = (1.0 - com.correle(D, "T", com.territoire) / nD) if D.any() else np.zeros_like(Z)
            # rapporté au moins à la moitié de la surface quadrillée : une feuille clairsemée n'a
            # de dense que la bordure bâtie qu'elle partage avec sa voisine
            n_cadre = 0.5 * ConvexHull(h["loc"]).volume * sim.k ** 2 / RES ** 2
            recouv = (com.correle(D, ("occ", id(occ), occ.version), occ.O) / max(nD, n_cadre)
                      if occ is not None else None)
            c = h["loc"].mean(axis=0)
            offE = st * np.arange(np.ceil((lo[0] - c[0]) / st), np.floor((hi[0] - c[0]) / st) + 1)
            offN = st * np.arange(np.ceil((lo[1] - c[1]) / st), np.floor((hi[1] - c[1]) / st) + 1)
            if not len(offE) or not len(offN):
                continue
            yfam = "j" if h["xfam"] == "i" else "i"
            tN, kN = _cotes_1d(readings[h["xfam"]], h["sx"], st, offN)   # cotes X (nord)
            tE, kE = _cotes_1d(readings[yfam], h["sy"], st, offE)        # cotes Y (est)
            EE, NN = np.meshgrid(offE, offN)
            O = sim.O + np.column_stack([EE.ravel(), NN.ravel()]) @ sim.M.T
            u = np.round((O[:, 0] + x0 - com.bbox[0]) / RES).astype(int) + wa - 1
            v = np.round((com.bbox[3] - O[:, 1] - y1) / RES).astype(int) + ha - 1
            ok = (u >= 0) & (v >= 0) & (u < Z.shape[1]) & (v < Z.shape[0])
            if not ok.any():
                continue
            u, v = u[ok], v[ok]
            iN, iE = np.unravel_index(np.nonzero(ok)[0], EE.shape)
            z, dh = Z[v, u], dehors[v, u]
            rc = recouv[v, u] if recouv is not None else np.zeros_like(z)
            cot = tN[iN] + tE[iE]
            retournee = abs(nord_vs_haut(sim.M, h)) > 95.0
            s = (np.clip(z, -3, 25) - W_RETOURNEE * retournee + W_COTES * cot - W_RECOUV * np.clip(rc - TOL_RECOUV, 0, 1)
                 - W_DEHORS * np.clip(dh - TOL_DEHORS, 0, 1))
            for k in np.argsort(-s)[:top]:
                snap = np.array([offE[iE[k]], offN[iN[k]]])
                hh = dict(h)
                hh["loc"] = h["loc"] + snap
                hh["aff"] = h["aff"].copy()
                hh["aff"][:, 2] += snap
                hh["step"] = st
                out.append({"score": float(s[k]), "h": hh, "cotes_ok": int(kN[iN[k]] + kE[iE[k]]),
                            "cotes": float(cot[k]), "cotes_lues": len(readings["i"]) + len(readings["j"]),
                            "z": float(z[k]), "recouv": float(rc[k]), "dehors": float(dh[k])})
    out.sort(key=lambda c: -c["score"])
    return out[:top]


def verify(mask, h, sim, com, search=40.0, res_m=1.0):
    """Corrélation fine de la feuille placée par sim : décalage résiduel et netteté."""
    C = sim.affine(h["aff"])
    A, (x0, y1) = render(mask, C, res_m, pad=search)
    bbox = (x0, y1 - A.shape[0] * res_m, x0 + A.shape[1] * res_m, y1)
    B = modern(com.rings, com.bb, bbox, res_m)
    if B.sum() < 50:
        return {"score": 0.0, "z": 0.0, "dE": 0.0, "dN": 0.0}
    a = bp(A) * (gauss(A, 10) > 0.01)
    b = bp(B)
    cc = sfft.irfft2(np.conj(sfft.rfft2(a)) * sfft.rfft2(b), s=a.shape)
    cc /= np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
    cc = np.fft.fftshift(cc)
    cy, cx = np.array(cc.shape) // 2
    n = int(search / res_m)
    win = cc[cy - n:cy + n + 1, cx - n:cx + n + 1]
    k = np.unravel_index(int(np.argmax(win)), win.shape)
    return {"score": float(win[k]), "z": float((win[k] - win.mean()) / (win.std() + 1e-9)),
            "dE": float((k[1] - n) * res_m), "dN": float((n - k[0]) * res_m)}


def control_image(mask, h, sim, com, path, res_m=1.0):
    """Feuille (vert) sur le cadastre actuel (rouge) : le jaune montre l'accord."""
    A, (x0, y1) = render(mask, sim.affine(h["aff"]), res_m, pad=30.0)
    bbox = (x0, y1 - A.shape[0] * res_m, x0 + A.shape[1] * res_m, y1)
    B = modern(com.rings, com.bb, bbox, res_m)
    A = np.clip(A * 4, 0, 1)
    rgb = np.dstack([255 * (1 - A), 255 * (1 - B), 255 * (1 - np.maximum(A, B) * 0.6)])
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path, quality=85)
