#!/usr/bin/env python3
"""Fermeture des trous du masque de traits AVANT la polygonisation.

Une limite cadastrale ne s'arrête jamais dans le vide : un bout de trait libre est
presque toujours un trait que le modèle n'a pas fini. Deux étapes :

1. hystérésis : les pixels faibles (p > low) reliés à un trait sûr sont gardés --
   un trait pâle reste une crête faible, le papier vide est à ~0 ;
2. raccord des bouts : pour chaque extrémité du squelette, chemin de moindre coût
   sur (1 - p) vers un AUTRE trait, à moins de ``reach_m``. Accepté seulement si
   - il part dans l'axe du trait (écart <= ``max_angle``),
   - il suit une crête (p moyen >= ``min_mean_p``),
   - la cible n'est pas déjà à portée le long du même trait (pas de raccourci
     qui couperait une boucle existante).
"""
import numpy as np
from scipy import ndimage

_NB = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])


def hysteresis(mask, prob, low=0.12, high=0.5):
    from skimage.filters import apply_hysteresis_threshold

    return mask | apply_hysteresis_threshold(prob, low, high)


def _endpoints(skel):
    n = ndimage.convolve(skel.astype(np.uint8), _NB, mode="constant")
    return np.argwhere(n == 11)  # pixel du squelette avec exactement 1 voisin


def _trace(skel, r, c, steps):
    """Remonte le trait depuis l'extrémité ; renvoie les pixels visités (ordre)."""
    path = [(r, c)]
    seen = {(r, c)}
    H, W = skel.shape
    for _ in range(steps):
        nxt = None
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if (dr or dc) and 0 <= rr < H and 0 <= cc < W and skel[rr, cc] and (rr, cc) not in seen:
                    nxt = (rr, cc)
                    break
            if nxt:
                break
        if nxt is None:
            break
        r, c = nxt
        seen.add(nxt)
        path.append(nxt)
    return path


def _crossings(skel, r, c, n, half):
    """Nombre de traits croisés par un segment de demi-longueur ``half`` px,
    perpendiculaire à la direction ``n`` (vecteur unitaire), centré en (r, c)."""
    H, W = skel.shape
    perp = np.array([-n[1], n[0]])
    side = []
    for sgn in (1, -1):
        k, prev = 0, False
        for t in range(2, half + 1):
            rr, cc = int(round(r + sgn * perp[0] * t)), int(round(c + sgn * perp[1] * t))
            if not (0 <= rr < H and 0 <= cc < W):
                break
            on = skel[max(0, rr - 1):rr + 2, max(0, cc - 1):cc + 2].any()
            if on and not prev:
                k += 1
            prev = on
        side.append(k)
    return side


def link_endpoints(mask, prob, res, reach_m=4.0, max_angle=35.0, min_mean_p=0.04,
                   along_factor=3.0, width_px=None, valid=None, hatch_m=4.0, hatch_min=2,
                   min_trace_m=0.0):
    from skimage.graph import MCP_Geometric
    from skimage.morphology import skeletonize
    from skimage.draw import line as draw_line

    skel = skeletonize(mask)
    R = max(4, int(round(reach_m / res)))
    eps = _endpoints(skel)
    H, W = mask.shape
    if width_px is None:
        width_px = 3
    out = mask.copy()
    stats = dict(endpoints=int(len(eps)), linked=0, rejected_angle=0, rejected_ridge=0,
                 rejected_hatch=0, no_target=0)
    Hh = max(4, int(round(hatch_m / res)))
    added = np.zeros_like(mask)
    for r, c in eps:
        if valid is not None and not valid[r, c]:
            continue
        tr = _trace(skel, r, c, max(6, R // 2))
        if len(tr) < max(4, min_trace_m / res):
            continue  # moignon (tache collée à un trait, lettre) : on ne le prolonge pas
        tail = np.array(tr[min(len(tr) - 1, max(3, R // 3))], float)
        d = np.array([r, c], float) - tail
        nd = np.hypot(*d)
        if nd < 1e-6:
            continue
        d /= nd
        # Hachures : traits parallèles serrés de part et d'autre -> pas une limite
        if hatch_min:
            mid = tr[min(len(tr) - 1, max(2, R // 4))]
            a, b = _crossings(skel, mid[0], mid[1], d, Hh)
            if a >= hatch_min and b >= hatch_min:
                stats["rejected_hatch"] += 1
                continue
        r0, r1 = max(0, r - R), min(H, r + R + 1)
        c0, c1 = max(0, c - R), min(W, c + R + 1)
        sub_s = skel[r0:r1, c0:c1].copy()
        # Exclure le trait lui-même à portée le long de son squelette (along_factor*R)
        own = _trace(skel, r, c, int(along_factor * R))
        for (pr_, pc_) in own:
            if r0 <= pr_ < r1 and c0 <= pc_ < c1:
                sub_s[pr_ - r0, pc_ - c0] = False
        # les pixels voisins immédiats du trait propre (épaisseur 1 du squelette)
        tg = np.argwhere(sub_s)
        if not len(tg):
            stats["no_target"] += 1
            continue
        lr, lc = r - r0, c - c0
        v = tg - [lr, lc]
        dist = np.hypot(v[:, 0], v[:, 1])
        ok = (dist > 1.5) & (dist <= R)
        cosang = (v[:, 0] * d[0] + v[:, 1] * d[1]) / np.maximum(dist, 1e-6)
        ok &= cosang >= np.cos(np.radians(max_angle))
        if not ok.any():
            stats["rejected_angle"] += 1 if (dist <= R).any() else 0
            stats["no_target"] += 0 if (dist <= R).any() else 1
            continue
        p_sub = prob[r0:r1, c0:c1].astype(np.float32)
        cost = 1.0 / (0.05 + p_sub)
        mcp = MCP_Geometric(cost)
        cum, _ = mcp.find_costs([(lr, lc)])
        cand = tg[ok]
        best = cand[np.argmin(cum[cand[:, 0], cand[:, 1]])]
        path = np.array(mcp.traceback(tuple(best)))
        mean_p = float(p_sub[path[:, 0], path[:, 1]].mean())
        if mean_p < min_mean_p:
            stats["rejected_ridge"] += 1
            continue
        added[path[:, 0] + r0, path[:, 1] + c0] = True
        stats["linked"] += 1
    if width_px > 1 and added.any():
        added = ndimage.binary_dilation(added, iterations=max(1, int(width_px) // 2))
    out |= added
    return out, stats


def prune(mask, prob, res, spur_m=1.2, bridge_m=2.0, bridge_p=0.6):
    """Enlève du masque les branches de squelette parasites :
    - moignons : une extrémité libre, plus courts que ``spur_m`` ;
    - barreaux : entre deux jonctions distinctes, plus courts que ``bridge_m`` et
      faibles (p moyen < ``bridge_p``) -- typiquement une tache qui relie les deux
      traits d'une bande de marais et la hache en triangles.
    Une limite raccordée dans l'axe d'un trait fait partie d'une branche longue :
    elle n'est pas touchée."""
    from skimage.morphology import skeletonize

    skel = skeletonize(mask)
    deg = ndimage.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    junc = skel & (deg >= 3)
    jlab, nj = ndimage.label(ndimage.binary_dilation(junc, structure=np.ones((3, 3))) & skel,
                             structure=np.ones((3, 3)))
    br = skel & (jlab == 0)
    blab, nb = ndimage.label(br, structure=np.ones((3, 3)))
    if nb == 0:
        return mask, dict(spurs=0, bridges=0)
    idx = np.arange(1, nb + 1)
    length = ndimage.sum(br, blab, idx) * res
    ends = ndimage.sum(br & (deg == 1), blab, idx)
    meanp = ndimage.mean(prob, blab, idx)
    big = nj + 1
    jmax = ndimage.maximum_filter(jlab, size=3)
    jmin = ndimage.minimum_filter(np.where(jlab > 0, jlab, big), size=3)
    tmax = ndimage.maximum(jmax, blab, idx)
    tmin = ndimage.minimum(jmin, blab, idx)
    touch = np.where(tmax == 0, 0, np.where(tmin == tmax, 1, 2))
    spur = (ends >= 1) & (touch == 1) & (length < spur_m)
    bridge = (ends == 0) & (touch == 2) & (tmin != tmax) & (length < bridge_m) & (meanp < bridge_p)
    kill = np.zeros(nb + 1, bool)
    kill[1:] = spur | bridge
    if not kill.any():
        return mask, dict(spurs=0, bridges=0)
    # chaque pixel du masque appartient au pixel de squelette le plus proche --
    # calculé seulement dans une petite fenêtre autour de chaque branche retirée
    # (sur la feuille entière, les indices de l'EDT coûtent > 5 Go).
    out = mask.copy()
    klab = np.where(kill[blab], blab, 0)
    pad = 8
    H, W = mask.shape
    for i, sl in enumerate(ndimage.find_objects(klab), 1):
        if sl is None:
            continue
        r0, r1 = max(0, sl[0].start - pad), min(H, sl[0].stop + pad)
        c0, c1 = max(0, sl[1].start - pad), min(W, sl[1].stop + pad)
        sk = skel[r0:r1, c0:c1]
        near = ndimage.distance_transform_edt(~sk, return_distances=False, return_indices=True)
        owner = blab[r0:r1, c0:c1][near[0], near[1]]
        out[r0:r1, c0:c1] &= ~(owner == i)
    return out, dict(spurs=int(spur.sum()), bridges=int(bridge.sum()))


def close_gaps(mask, prob, res, valid=None, low=0.12, high=0.5, do_prune=False,
               prune_kw=None, **kw):
    """Hystérésis puis raccord des bouts (puis élagage) ; renvoie (masque, stats)."""
    m = hysteresis(mask, prob, low, high) if low is not None else mask.copy()
    if valid is not None:
        m &= valid
    gained = int(m.sum() - mask.sum())
    m, st = link_endpoints(m, prob, res, valid=valid, **kw)
    st["hysteresis_px"] = gained
    if do_prune:
        m, ps = prune(m, prob, res, **(prune_kw or {}))
        st.update(ps)
    return m, st
