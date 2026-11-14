"""Détection du quadrillage rouge des feuilles du cadastre rénové (après ~1870).

Généralise le prototype Douai 1908 : pas, angle et échelle quelconques, traits
rouges fins et pâles. Sortie : les deux familles de droites (angle, pas en px) et
chaque intersection affinée localement à pleine résolution (x, y, indices i, j).
"""
import numpy as np
from PIL import Image
from scipy import ndimage

Image.MAX_IMAGE_PIXELS = None
DS = 4                      # réduction pour la recherche globale
PERIOD_PX = (280, 1500)     # pas plausible du quadrillage à pleine résolution


def ridge(rgb, k=6):
    """Réponse « trait rouge fin » : plus rouge que ses deux côtés à k px.

    Un bord d'aplat rose (bâtiment, liseré de commune) n'est plus rouge que d'un
    seul côté : réponse nulle. L'encre noire et sa frange JPEG sont écartées.
    """
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    d = ndimage.uniform_filter(r - 0.5 * (g + b), 5)
    out = np.zeros_like(d)
    for dy, dx in ((0, k), (k, 0), (k, k), (k, -k)):
        a = np.roll(d, (dy, dx), axis=(0, 1))
        c = np.roll(d, (-dy, -dx), axis=(0, 1))
        np.maximum(out, d - np.maximum(a, c), out=out)
    dark = ndimage.binary_dilation(np.minimum(r, g) < 110, iterations=4)
    out[dark] = 0
    out[:k + 2] = out[-k - 2:] = 0
    out[:, :k + 2] = out[:, -k - 2:] = 0
    return out


def _pool(a):
    h, w = a.shape
    h2, w2 = h // DS * DS, w // DS * DS
    return a[:h2, :w2].reshape(h2 // DS, DS, w2 // DS, DS).max(axis=(1, 3))


def _mask(pooled):
    v = pooled[pooled > 0]
    if v.size < 1000:
        return pooled > 1e9
    med = np.median(v)
    sig = 1.4826 * np.median(np.abs(v - med)) + 1e-3
    return pooled > max(med + 3.0 * sig, 5.0)


def _profile(ys, xs, theta):
    rho = xs * np.cos(theta) + ys * np.sin(theta)
    lo = np.floor(rho.min())
    h = np.bincount((rho - lo).astype(np.int64), minlength=int(rho.max() - lo) + 2)
    return h.astype(np.float64), lo


def _sharp(h):
    hp = h - ndimage.uniform_filter1d(h, 25)
    return float((np.clip(hp, 0, None) ** 2).sum())


def _best_angle(ys, xs, angles):
    sc = [(_sharp(_profile(ys, xs, np.radians(t))[0]), t) for t in angles]
    return max(sc)[1], sc


def _clean(h):
    p = h - ndimage.uniform_filter1d(h, 31)
    p[p < 0] = 0
    return p


def _autocorr(p, n):
    ac = np.correlate(p, p, mode="full")[len(p) - 1:]
    ac = ac / (ac[0] + 1e-9)
    out = np.zeros(n)
    m = min(n, len(ac))
    out[:m] = ac[:m]
    return out


def _period(pa, pb):
    """Pas commun aux deux familles (px réduits), par autocorrélation."""
    lo, hi = PERIOD_PX[0] // DS, PERIOD_PX[1] // DS
    ac = _autocorr(pa, hi + 2) + _autocorr(pb, hi + 2)
    ac = ndimage.gaussian_filter1d(ac, 1.0)
    P = lo + int(np.argmax(ac[lo:hi]))
    # un pic à 2P ou 3P peut dépasser celui de P quand il manque des traits
    for div in (3, 2):
        q = P / div
        if q >= lo:
            c = int(round(q))
            w = ac[max(lo, c - 3):c + 4]
            if w.size and w.max() > 0.55 * ac[P]:
                P = max(lo, c - 3) + int(np.argmax(w))
                break
    a, b, c = ac[P - 1], ac[P], ac[P + 1]
    return P + 0.5 * (a - c) / (a - 2 * b + c + 1e-9), float(ac[P])


def _lines(p, P, lo):
    n = len(p)
    ks = np.arange(0, int(n / P) + 1)
    best = (0.0, -1.0)
    for phase in np.arange(0, P, 0.5):
        idx = (phase + ks * P).astype(int)
        s = p[idx[idx < n]].sum()
        if s > best[1]:
            best = (phase, s)
    phase = best[0]
    ref = np.percentile(p[p > 0], 99) if (p > 0).any() else 1.0
    out = []
    for k in ks:
        c = int(phase + k * P)
        if c >= n:
            break
        a, b = max(0, c - 5), min(n, c + 6)
        seg = p[a:b]
        if seg.max() > 0.2 * ref:
            out.append((int(k), float(lo + a + int(np.argmax(seg)) + 0.5)))
    return out


def detect(path):
    """Renvoie dict(size, families[2], points, quality) ou lève ValueError."""
    im = Image.open(path).convert("RGB")
    W, H = im.size
    rgb = np.asarray(im)
    rid = ridge(rgb)
    m = _mask(_pool(rid))
    ys, xs = np.nonzero(m)
    if len(ys) < 2000:
        raise ValueError("pas de trait rouge fin")
    rng = np.random.default_rng(0)
    ys = ys + rng.uniform(-0.5, 0.5, len(ys))
    xs = xs + rng.uniform(-0.5, 0.5, len(xs))
    # les deux familles sont perpendiculaires : chaque direction est notée avec sa
    # perpendiculaire, puis les meilleures paires sont départagées par la
    # régularité du pas (des rangées de maisons roses ne sont pas périodiques)
    _, sc = _best_angle(ys, xs, np.arange(0, 180, 0.5))
    vals = np.array([s for s, _ in sc])
    loc = ndimage.maximum_filter1d(vals, 5, mode="wrap")
    joint = np.sqrt(loc[:180] * loc[180:])
    peaks = [k for k in np.argsort(joint)[::-1]
             if joint[k] >= ndimage.maximum_filter1d(joint, 21, mode="wrap")[k]][:6]
    best = None
    for k in peaks:
        t1, _ = _best_angle(ys, xs, np.arange(k * 0.5 - 1.5, k * 0.5 + 1.5, 0.05))
        t2, _ = _best_angle(ys, xs, np.arange(t1 + 90 - 1.5, t1 + 90 + 1.5, 0.05))
        profs = []
        for t in (t1, t2):
            t, _ = _best_angle(ys, xs, np.arange(t - 0.06, t + 0.06, 0.01))
            h, lo = _profile(ys, xs, np.radians(t))
            profs.append((t, _clean(h), lo))
        P, acq = _period(profs[0][1], profs[1][1])
        if best is None or acq > best[0]:
            best = (acq, P, profs, float(joint[k] / (np.median(joint) + 1e-9)))
    acq, P, profs, contrast = best
    fam = []
    for t, p, lo in profs:
        fam.append({"theta": float(t), "period_px": float(P * DS),
                    "lines": [(k, r * DS) for k, r in _lines(p, P, lo)]})
    res = {"file": path, "size": [W, H], "families": fam,
           "quality": {"angle_contrast": contrast, "autocorr": float(acq)}}
    res["points"] = refine(res, rid)
    if len(fam[0]["lines"]) < 2 or len(fam[1]["lines"]) < 2 or len(res["points"]) < 4:
        raise ValueError("quadrillage introuvable")
    return res


def _intersections(res):
    fa, fb = res["families"]
    ta, tb = np.radians(fa["theta"]), np.radians(fb["theta"])
    A = np.array([[np.cos(ta), np.sin(ta)], [np.cos(tb), np.sin(tb)]])
    W, H = res["size"]
    pts = []
    for i, ra in fa["lines"]:
        for j, rb in fb["lines"]:
            x, y = np.linalg.solve(A, [ra, rb])
            if 0 <= x < W and 0 <= y < H:
                pts.append({"i": i, "j": j, "x": float(x), "y": float(y)})
    return pts


SKIP, SEARCH = 14, 26


def refine(res, rid):
    """Affinage local de chaque intersection ; les points sans trait sont écartés.

    Le trait est intégré sur près d'un pas de part et d'autre du croisement : un
    trait pâle haché par la compression JPEG ressort quand même du bruit.
    """
    H, W = rid.shape
    along = int(0.45 * res["families"][0]["period_px"])
    R = along + 8
    rid = np.minimum(rid, 30.0)
    normals = [np.array([np.cos(np.radians(f["theta"])), np.sin(np.radians(f["theta"]))])
               for f in res["families"]]
    A = np.array(normals)
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    bins = np.arange(-SEARCH, SEARCH + 1.0)
    out = []
    for p in _intersections(res):
        x0, y0 = int(round(p["x"])), int(round(p["y"]))
        ya, yb, xa, xb = max(0, y0 - R), min(H, y0 + R + 1), max(0, x0 - R), min(W, x0 + R + 1)
        win = rid[ya:yb, xa:xb]
        dx = xx[ya - y0 + R:yb - y0 + R, xa - x0 + R:xb - x0 + R] + (x0 - p["x"])
        dy = yy[ya - y0 + R:yb - y0 + R, xa - x0 + R:xb - x0 + R] + (y0 - p["y"])
        offs, zs = [], []
        for n in normals:
            tn = dx * n[0] + dy * n[1]
            tl = -dx * n[1] + dy * n[0]
            sel = (np.abs(tn) <= SEARCH) & (np.abs(tl) <= along) & (np.abs(tl) >= SKIP)
            hs = []
            for side in (tl > 0, tl < 0):
                s2 = sel & side
                h, _ = np.histogram(tn[s2], bins=bins, weights=win[s2])
                hs.append(ndimage.gaussian_filter1d(h, 1.0))
            h = hs[0] + hs[1]
            k = int(np.argmax(h))
            if k < 2 or k > len(h) - 3:
                offs = None
                break
            rest = np.r_[h[:k - 2], h[k + 3:]]
            base = np.median(rest)
            noise = 1.4826 * np.median(np.abs(rest - base)) + 1.0
            z = (h[k] - base) / noise
            # le trait doit exister des deux côtés du croisement (sinon : bord du
            # cadre, fin de trait) sauf s'il sort de l'image
            lo_side = min(hs[0][k - 1:k + 2].max() - np.median(hs[0]),
                          hs[1][k - 1:k + 2].max() - np.median(hs[1]))
            if z < 6 or lo_side < 0.15 * (h[k] - base):
                offs = None
                break
            sub = 0.5 * (h[k - 1] - h[k + 1]) / (h[k - 1] - 2 * h[k] + h[k + 1] + 1e-9)
            offs.append(bins[k] + 0.5 + sub)
            zs.append(float(z))
        if offs is None:
            continue
        d = np.linalg.solve(A, np.array(offs))
        out.append({"i": p["i"], "j": p["j"], "x": float(p["x"] + d[0]),
                    "y": float(p["y"] + d[1]), "z": round(min(zs), 1)})
    return _consistent(out)


def _consistent(pts):
    """Écarte les points qui ne suivent pas le modèle affine (i, j) -> (x, y)."""
    if len(pts) < 6:
        return pts
    ij = np.array([[p["i"], p["j"], 1.0] for p in pts])
    xy = np.array([[p["x"], p["y"]] for p in pts])
    keep = np.ones(len(pts), bool)
    for _ in range(4):
        c, *_ = np.linalg.lstsq(ij[keep], xy[keep], rcond=None)
        r = np.hypot(*(ij @ c - xy).T)
        lim = max(6.0, 4.0 * np.median(r[keep]))
        new = r <= lim
        if (new == keep).all():
            break
        keep = new
    return [p for p, k in zip(pts, keep) if k]


def overlay(res, path, width=1800):
    """Image de contrôle : traits détectés et intersections retenues."""
    from PIL import ImageDraw
    im = Image.open(res["file"]).convert("RGB")
    s = width / im.size[0]
    im = im.resize((width, int(im.size[1] * s)))
    d = ImageDraw.Draw(im)
    for p in res["points"]:
        x, y = p["x"] * s, p["y"] * s
        d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(0, 90, 255), width=2)
        d.text((x + 6, y + 4), f"{p['i']},{p['j']}", fill=(0, 60, 200))
    im.save(path, quality=80)


if __name__ == "__main__":
    import json
    import sys
    import time
    for f in sys.argv[2:]:
        t = time.time()
        try:
            res = detect(f)
        except ValueError as e:
            print("ECHEC", f.split("/")[-1], e)
            continue
        fa, fb = res["families"]
        name = f.split("/")[-1].rsplit(".", 1)[0]
        print(f"{name}: theta={fa['theta']:.2f}/{fb['theta']:.2f} pas={fa['period_px']:.1f}px "
              f"traits={len(fa['lines'])}x{len(fb['lines'])} points={len(res['points'])} "
              f"q={res['quality']['angle_contrast']:.1f}/{res['quality']['autocorr']:.2f} "
              f"{time.time() - t:.1f}s", flush=True)
        overlay(res, f"{sys.argv[1]}/{name}_grille.jpg")
        json.dump(res, open(f"{sys.argv[1]}/{name}.json", "w"))
