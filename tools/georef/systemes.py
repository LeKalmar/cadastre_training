"""Systèmes de quadrillage partagés par plusieurs communes (un par arrondissement, en gros).

Les levés rénovés d'un même arrondissement partagent l'origine et l'orientation du
quadrillage : une seule similitude explique les feuilles « ok » d'une vingtaine de
communes à 1-2 m près (médiane) sur 20 km. Le calage d'une commune faible (peu de
feuilles d'appui, cadastre actuel très changé) peut donc s'appuyer sur celui de tout
son système au lieu de ses seules feuilles.

    .venv-georef/bin/python tools/georef/systemes.py data/georef_nord data/georef_ad59 \
        --out data/georef_systemes.json

Construit les systèmes à partir des calage.json existants : regroupement des communes
par origine, puis similitude robuste (les communes dont le calage s'écarte du système
sont écartées et listées). run.py --systemes <fichier> s'en sert ensuite.
"""
import argparse
import glob
import json
import os

import numpy as np

TOL_ORIGINE = 60.0       # m : deux communes d'un même système ont la même origine à ~10 m près
ECART_MAX = 8.0          # m : résidu médian au-delà duquel une commune est écartée du système
APPUI_MIN = 3            # feuilles d'appui pour qu'un calage de commune serve à bâtir un système


def _sheets(cj, m):
    """[(local, L93)] des centres des feuilles « ok » d'un calage de commune."""
    th, k, O = np.radians(m["rotation_deg"]), m["echelle"], np.array(m["origine_l93"])
    Mi = np.linalg.inv(k * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]))
    out = []
    for f in m["feuilles"]:
        if f["statut"] != "ok" or (f.get("pas_m") or 0) >= 500:
            continue
        p = os.path.join(os.path.dirname(cj),
                         "GEOREF_" + f["feuille"].replace("RAW_", "").rsplit(".", 1)[0] + ".points")
        if not os.path.exists(p):
            continue
        c = np.loadtxt(p, delimiter=",", skiprows=2, ndmin=2)[:, :2].mean(0)
        out.append((Mi @ (c - O), c))
    return out


def _fit(loc, l93):
    """Similitude L93 = O + k·R(θ)·local, échelle libre : sur 20 km, l'échelle du levé s'écarte
    mesurablement de celle du Lambert 93 (~0,4 m/km), ce qu'une commune seule ne voit pas."""
    cl, cm = loc.mean(0), l93.mean(0)
    za, zb = (loc - cl) @ [1, 1j], (l93 - cm) @ [1, 1j]
    q = (np.conj(za) * zb).sum() / (np.abs(za) ** 2).sum()
    k, th = float(abs(q)), float(np.angle(q))
    M = k * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return cm - M @ cl, th, M, k


def construire(outs):
    cals = []
    for out in outs:
        for cj in sorted(glob.glob(os.path.join(out, "*", "*", "calage.json"))):
            m = json.load(open(cj))
            if m.get("feuilles_appui", 0) >= APPUI_MIN and not m.get("systeme"):
                cals.append((cj, m))
    groupes = []
    for cj, m in sorted(cals, key=lambda x: -x[1]["feuilles_appui"]):
        O = np.array(m["origine_l93"])
        for g in groupes:
            if np.hypot(*(g["O"] - O)) < TOL_ORIGINE:
                g["cal"].append((cj, m))
                break
        else:
            groupes.append({"O": O, "cal": [(cj, m)]})
    systemes = []
    for g in groupes:
        pts = {}
        for cj, m in g["cal"]:
            cle = (m["commune"], m.get("annee"))
            pts[cle] = (_sheets(cj, m), m)
        membres = {c for c, (s, _) in pts.items() if s}
        if len(membres) < 2:
            continue
        ecartes = {}
        for _ in range(5):
            loc = np.array([p[0] for c in membres for p in pts[c][0]])
            l93 = np.array([p[1] for c in membres for p in pts[c][0]])
            O, th, M, k = _fit(loc, l93)
            med = {c: float(np.median([np.hypot(*(O + M @ p[0] - p[1])) for p in pts[c][0]])) for c in membres}
            pire = max(med, key=med.get)
            if med[pire] <= ECART_MAX or len(membres) <= 2:
                break
            ecartes[pire] = med[pire]
            membres.discard(pire)
        if len(membres) < 2:
            continue
        loc = np.array([p[0] for c in membres for p in pts[c][0]])
        l93 = np.array([p[1] for c in membres for p in pts[c][0]])
        res = np.hypot(*(l93 - (O + loc @ M.T)).T)
        centres = {}
        for c in membres:
            centres.setdefault(c[0], np.array([p[1] for p in pts[c][0]]).mean(0).tolist())
        systemes.append({
            "origine_l93": O.tolist(), "rotation_deg": float(np.degrees(th)), "echelle": k,
            "echelle_l93": float(np.median([pts[c][1]["echelle"] for c in membres])),
            "feuilles": int(len(loc)), "residu_median_m": float(np.median(res)),
            "residu_90_m": float(np.percentile(res, 90)),
            "communes": sorted({c[0] for c in membres}), "centres": centres,
            "ecartees": {f"{c[0]} {c[1]}": round(v, 1) for c, v in ecartes.items()},
        })
    systemes.sort(key=lambda s: -s["feuilles"])
    return systemes


def proches(systemes, xy, dmax=15000.0):
    """Systèmes dont une commune membre est à moins de dmax du point xy (L93), les plus proches d'abord."""
    out = []
    for s in systemes:
        d = min(np.hypot(c[0] - xy[0], c[1] - xy[1]) for c in s["centres"].values())
        if d < dmax:
            out.append((d, s))
    return [s for d, s in sorted(out, key=lambda x: x[0])]


def charger(path):
    return json.load(open(path))["systemes"] if path and os.path.exists(path) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outs", nargs="+", help="dossiers de sortie de run.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = construire(a.outs)
    json.dump({"sources": a.outs, "systemes": s}, open(a.out, "w"), indent=1, ensure_ascii=False)
    for x in s:
        print(f"origine {np.round(x['origine_l93'])} rot {x['rotation_deg']:+.3f}° : {len(x['communes'])} communes, "
              f"{x['feuilles']} feuilles, résidu méd {x['residu_median_m']:.1f} m (90 % {x['residu_90_m']:.1f} m)"
              + (f" ; écartées {x['ecartees']}" if x["ecartees"] else ""))
    print(f"{len(s)} systèmes -> {a.out}")


if __name__ == "__main__":
    main()
