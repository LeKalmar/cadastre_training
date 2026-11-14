"""Géoréférencement automatique des feuilles à quadrillage chiffré (cadastre rénové).

    .venv-georef/bin/python tools/georef/run.py --src "<…/59 - Nord>" --out data/georef_nord \
        [--only 59211_Esquerchin …] [--min-year 1870] [--jobs 6] [--no-warp]

Par commune : détection du quadrillage et lecture des cotes (mises en cache),
calage commun sur le cadastre actuel, puis GeoTIFF EPSG:2154 + .points QGIS.
Sorties dans <out>/<commune>/ ; bilan dans <out>/rapport.csv et rapport.json.
"""
import argparse
import csv
import json
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import communes as CM  # noqa: E402
import grid as G  # noqa: E402
import labels as LB  # noqa: E402
import place as PL  # noqa: E402
import systemes as SY  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
SKIP_DIRS = {"GEOREF", "Géoréférence", "DECOUPE", "Découpe", "node_modules", "FLO.Altis", "Points"}


def sheets_of(folder, min_year):
    """[(chemin, année ou None)] des scans d'un dossier de commune."""
    out = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        rel = os.path.relpath(root, folder)
        for f in sorted(files):
            if not f.lower().endswith((".jpg", ".jpeg")) or "ASSEMBLAGE" in f.upper():
                continue
            m = re.search(r"(?<!\d)(1[789]\d\d)(?!\d)", rel + "/" + f)
            year = int(m.group(1)) if m else None
            if min_year and year is not None and year < min_year:
                continue
            out.append((os.path.join(root, f), year))
    return out


def analyse(args):
    """Quadrillage + cotes d'une feuille, avec cache JSON."""
    path, cache = args
    if os.path.exists(cache):
        return json.load(open(cache))
    out = {"file": path}
    try:
        res = G.detect(path)
        rec = LB.read(res)
        out.update(res)
        out["ocr"] = rec
    except ValueError as e:
        out["error"] = str(e)
    except Exception:
        out["error"] = traceback.format_exc(limit=2)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(out, open(cache, "w"))
    return out


class MaskRef:
    """Masque d'encre gardé sur disque (PNG) : une commune comme Lille 1881 a 150
    feuilles, soit ~7 Go de masques en mémoire. Chargé à la demande."""

    def __init__(self, path):
        self.path = path

    @classmethod
    def save(cls, img, path):
        img.save(path, optimize=False, compress_level=1)
        return cls(path)

    def get(self):
        return Image.open(self.path).convert("L")


def good_fit(ft):
    return bool(ft) and ft["i"]["lines_ok"] >= 2 and ft["j"]["lines_ok"] >= 2 \
        and ft["i"]["lines_ok"] + ft["j"]["lines_ok"] >= 5


def write_points(path, px, l93):
    with open(path, "w") as f:
        f.write("#CRS: EPSG:2154\nmapX,mapY,sourceX,sourceY,enable,dX,dY,residual\n")
        for (x, y), (E, N) in zip(px, l93):
            f.write(f"{E:.3f},{N:.3f},{x:.2f},{-y:.2f},1,0,0,0\n")


def warp(src, dst, px, l93):
    """TPS GDAL sur les croisements, GeoTIFF JPEG avec masque et aperçus."""
    from osgeo import gdal
    gdal.UseExceptions()
    gcps = [gdal.GCP(float(E), float(N), 0.0, float(x), float(y)) for (x, y), (E, N) in zip(px, l93)]
    tmp = dst + ".vrt"
    ds = gdal.Translate(tmp, src, format="VRT", GCPs=gcps, outputSRS="EPSG:2154")
    ds = None
    mem = gdal.Warp("", tmp, dstSRS="EPSG:2154", tps=True, resampleAlg="cubic", dstAlpha=True,
                    format="MEM", multithread=True, warpOptions=["NUM_THREADS=4"])
    os.remove(tmp)
    # même format que les feuilles de Douai 1908 : RVB JPEG, masque interne, aperçus
    gdal.SetConfigOption("GDAL_TIFF_INTERNAL_MASK", "YES")
    ds = gdal.Translate(dst, mem, bandList=[1, 2, 3], maskBand=4, format="GTiff",
                        creationOptions=["COMPRESS=JPEG", "JPEG_QUALITY=88", "PHOTOMETRIC=YCBCR",
                                         "TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512"])
    ds.BuildOverviews("AVERAGE", [2, 4, 8, 16, 32])
    gt = ds.GetGeoTransform()
    ds = mem = None
    return abs(gt[1])


def tps_loo(px, l93):
    from scipy.interpolate import RBFInterpolator
    px, l93 = np.asarray(px), np.asarray(l93)
    if len(px) < 8:
        return None
    r = []
    for k in range(len(px)):
        m = np.ones(len(px), bool)
        m[k] = False
        f = RBFInterpolator(px[m], l93[m], kernel="thin_plate_spline", smoothing=1.0)
        r.append(np.hypot(*(f(px[k:k + 1])[0] - l93[k])))
    return float(np.sqrt(np.mean(np.square(r))))


def commune(folder, a):
    """Un calage par millésime : deux levés d'une même commune n'ont pas forcément
    la même origine."""
    name = os.path.basename(folder.rstrip("/"))
    by_year = {}
    for p, y in sheets_of(folder, a.min_year):
        by_year.setdefault(y, []).append((p, y))
    rows = []
    for y in sorted(by_year, key=lambda v: v or 0):
        cj = os.path.join(a.out, name, str(y or "sans_date"), "calage.json")
        if a.sauter_faits and os.path.exists(cj):
            print(f"{name} {y}: déjà fait, sauté", flush=True)
            rows += json.load(open(cj))["feuilles"]
            continue
        try:
            rows += millesime(name, y, by_year[y], a)
        except Exception:
            print(f"{name} {y}: ERREUR\n{traceback.format_exc()}", flush=True)
    return rows


def millesime(name, annee, sheets, a):
    rows = []
    label = f"{name} {annee or 'sans date'}"
    cache = os.path.join(a.out, "_cache")
    jobs = [(p, os.path.join(cache, name, os.path.basename(p) + ".json")) for p, _ in sheets]
    import time
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        ana = list(ex.map(analyse, jobs))
    print(f"{label}: analyse {time.time() - t0:.0f} s", flush=True)
    for r in ana:
        if "ocr" in r:
            r["fit"] = LB.fit(r["ocr"])
    year = {p: y for p, y in sheets}
    ok, weak = [], []
    for r in ana:
        row = {"commune": name, "feuille": os.path.basename(r["file"]), "annee": year[r["file"]],
               "statut": "", "source": r["file"]}
        rows.append(row)
        if "error" in r:
            row["statut"] = "sans quadrillage" if "quadrillage" in r["error"] or "trait" in r["error"] \
                else "erreur : " + r["error"].strip().splitlines()[-1][:80]
        elif not good_fit(r.get("fit")):
            row["statut"] = "cotes illisibles"
            row["croisements"] = len(r["points"])
            weak.append((r, row))
        else:
            row["croisements"] = len(r["points"])
            row["pas_m"] = r["fit"]["step"]
            ok.append((r, row))
    print(f"{label}: {len(sheets)} feuilles, {len(ok)} avec quadrillage chiffré, "
          f"{len(weak)} à cotes illisibles", flush=True)
    if not ok and not (weak and a.systemes):
        return rows
    com = CM.resolve(name, cache)
    if not com:
        for _, row in ok:
            row["statut"] = "commune non reconnue"
        return rows
    code, nom = com
    try:
        rings, bb = CM.parcels(code, cache)
    except Exception as e:
        for _, row in ok:
            row["statut"] = f"cadastre actuel indisponible ({e})"[:90]
        return rows
    C = PL.Commune(rings, bb)
    cx, cy = (C.bbox[0] + C.bbox[2]) / 2, (C.bbox[1] + C.bbox[3]) / 2
    k93, gamma = CM.l93_local_frame(cx, cy)
    voisins = SY.proches(a.systemes, (cx, cy)) if a.systemes else []
    if not ok and not voisins:
        return rows

    work = []
    cands = {}
    t0 = time.time()
    for n, (r, row) in enumerate(ok):
        im = Image.open(r["file"]).convert("RGB")
        mask = PL.ink_mask(r, im)
        hyps = PL.hypotheses(r["points"], r["fit"])
        cpath = os.path.join(cache, name, os.path.basename(r["file"]) + ".cand.json")
        cle = [r["fit"]["i"]["a"], r["fit"]["j"]["a"], r["fit"]["step"]]
        cs = json.load(open(cpath)) if os.path.exists(cpath) else None
        if isinstance(cs, dict):
            cs = cs["cands"] if cs.get("cotes") == cle else None
        elif cs is not None and cle[2] == 1000:
            cs = None          # ancien cache, d'avant le pas de 1000 m : cotes relues autrement
        if cs is None:
            cs = PL.candidates(mask, hyps, C, k93, gamma)
            json.dump({"cotes": cle, "cands": cs}, open(cpath, "w"))
        cands[n] = cs
        work.append((r, row, MaskRef.save(mask, cpath[:-len(".cand.json")] + ".masque.png"), hyps))
        del im, mask
        print(f"  {row['feuille']}: meilleur z={max(c['z'] for c in cs):.1f}" if cs else
              f"  {row['feuille']}: hors emprise", flush=True)

    print(f"  corrélations : {time.time() - t0:.0f} s", flush=True)
    centres = {n: [h["loc"].mean(axis=0) for h in w[3]] for n, w in enumerate(work)}
    # les tableaux d'assemblage (quadrillage de 500 m et plus) ne calent pas la commune
    detail = {n: cs for n, cs in cands.items() if work[n][0]["fit"]["step"] < LB.ASSEMBLAGE_PAS_M}
    if len(detail) < 2:
        detail = cands
    con = PL.ransac(detail, centres, k93, gamma) if len(detail) >= 2 else None
    sim = chosen = None
    if con is not None and len(con[1]) >= 2:
        sim, chosen = con
    else:
        con = PL.consensus(detail) if detail else None
        if con is not None:
            di, O, per0 = con
            sim = PL.Sim(O, gamma + PL.DELTAS[di], k93)
            chosen = {n: c["h"] for n, c in per0.items()}
    # calage du système (arrondissement) : préféré au calage propre de la commune quand ses
    # feuilles l'acceptent aussi bien (il repose sur des centaines de feuilles, pas sur quelques-unes)
    systeme = choisir_systeme(voisins, detail, centres, sim, chosen, k93, gamma, label, n_ok=len(ok),
                               emprise=C.bbox)
    if systeme is not None:
        s_, chosen = systeme
        sim = PL.Sim(s_["origine_l93"], np.radians(s_["rotation_deg"]), s_["echelle"])
    if sim is None:
        for _, row in ok:
            row["statut"] = "calage impossible"
        return rows
    per = dict(chosen)
    print(f"  accord initial : {len(per)}/{len(ok)} feuilles, origine {sim.O.round(1)}, "
          f"rotation {np.degrees(sim.theta):+.2f}°, échelle {sim.k:.5f}", flush=True)

    # ajustement itéré de la similitude sur les décalages fins de chaque feuille
    ver = {}
    for it in range(3):
        loc, tgt, w = [], [], []
        for n, (r, row, mask, hyps) in enumerate(work):
            hs = [chosen[n]] if n in chosen else range(len(hyps))
            m = mask.get()
            best = None
            for hi in hs:
                v = PL.verify(m, hyps[hi], sim, C, search=60.0 if it == 0 else 30.0,
                              res_m=2.0 if it == 0 else 1.0)
                if best is None or v["z"] > best[1]["z"]:
                    best = (hi, v)
            hi, v = best
            ver[n] = (hi, v)
            if v["z"] >= 6.0 and (n in detail or n in chosen):
                if it > 0 or n in chosen:
                    chosen[n] = hi
                c = hyps[hi]["loc"].mean(axis=0)
                loc.append(c)
                tgt.append(sim(c) + [v["dE"], v["dN"]])
                w.append(min(v["z"], 15.0))
        if len(loc) == 0 or systeme is not None:      # le calage du système ne se réajuste pas
            break
        loc, tgt = np.array(loc), np.array(tgt)
        span = np.hypot(*(loc.max(0) - loc.min(0))) if len(loc) > 1 else 0.0
        # échelle = celle du Lambert 93 (les levés rénovés n'ont pas d'erreur
        # d'échelle mesurable : -0,06 m/km à Douai) ; seule la rotation est libre
        sim = PL.fit_sim(loc, tgt, w, k93, sim.theta, free_rot=len(loc) >= 2 and span > 600)
    print(f"  similitude : origine {sim.O.round(2)}, rotation {np.degrees(sim.theta):+.3f}° "
          f"(convergence {np.degrees(gamma):+.3f}°), échelle {sim.k:.5f} (L93 {k93:.5f}), "
          f"{len(loc)} feuilles d'appui" + (" (calage du système)" if systeme else ""), flush=True)

    print(f"  ajustement : {time.time() - t0:.0f} s", flush=True)
    # feuilles à placer : (analyse, ligne du rapport, masque, orientation, mode)
    final = []
    steps = [r["fit"]["step"] for n, (r, row, mask, hyps) in enumerate(work) if n in per]
    steps = sorted(set(steps), key=lambda x: -steps.count(x))
    pending = [(r, row, None) for r, row in weak]
    for n, (r, row, mask, hyps) in enumerate(work):
        hi, _ = ver[n]
        v = PL.verify(mask.get(), hyps[hi], sim, C, search=30.0, res_m=1.0)
        if n in per or (v["z"] >= 6.0 and np.hypot(v["dE"], v["dN"]) <= 12.0):
            final.append((r, row, mask, hyps[hi], "cotes"))
        else:
            pending.append((r, row, (mask, hyps[hi])))
    fiable = len(per) >= 2 or systeme is not None
    if fiable:
        # feuilles en attente : placées sur le quadrillage de la commune (toutes les positions
        # entières y sont notées), les mieux lues d'abord, chacune servant de contrainte aux suivantes
        occ = PL.Occupation(C)
        for r, row, ref, h, mode in final:
            if h.get("step", r["fit"]["step"]) < LB.ASSEMBLAGE_PAS_M:      # un assemblage couvre tout
                occ.ajoute(ref.get(), sim.affine(h["aff"]))
        pending.sort(key=lambda p: -sum(p[0]["fit"][f]["lines_ok"] for f in "ij") if p[0].get("fit") else 0)
        for r, row, fallback in pending:
            if fallback:
                ref = fallback[0]
            else:
                ref = MaskRef.save(PL.ink_mask(r, Image.open(r["file"]).convert("RGB")),
                                   os.path.join(cache, name, os.path.basename(r["file"]) + ".masque.png"))
            mask = ref.get()
            st = list(steps)
            if good_fit(r.get("fit")) and r["fit"]["step"] not in st:
                st.append(r["fit"]["step"])
            c = placer_sur_grille(r, mask, sim, C, st, occ)
            if c is not None:
                row["pas_m"] = c["h"]["step"]
                row["grille"] = (f"cotes {c['cotes_ok']}/{c['cotes_lues']} z {c['z']:.1f}/{c['zf']:.1f} "
                                 f"recouv {c['recouv']:.0%} dehors {c['dehors']:.0%} marge {c['marge']:.1f}")
                final.append((r, row, ref, c["h"], "grille" if c["sur"] else "grille ?"))
                if c["sur"] and c["h"]["step"] < LB.ASSEMBLAGE_PAS_M:
                    occ.ajoute(mask, sim.affine(c["h"]["aff"]))
            elif fallback:
                final.append((r, row, ref, fallback[1], "douteux"))
            del mask
    else:
        final += [(r, row, f[0], f[1], "douteux") for r, row, f in pending if f]

    odir = os.path.join(a.out, name, str(annee or "sans_date"))
    os.makedirs(odir, exist_ok=True)
    meta = {"commune": name, "annee": annee, "insee": code, "nom": nom, "origine_l93": sim.O.tolist(),
            "rotation_deg": float(np.degrees(sim.theta)), "convergence_deg": float(np.degrees(gamma)),
            "echelle": sim.k, "echelle_l93": k93, "feuilles_appui": len(per), "feuilles": []}
    if systeme is not None:
        meta["systeme"] = {k: systeme[0][k] for k in ("origine_l93", "rotation_deg", "echelle", "communes")}
    for r, row, ref, h, mode in final:
        mask = ref.get()
        v = PL.verify(mask, h, sim, C, search=30.0, res_m=1.0)
        px = np.array([[p["x"], p["y"]] for p in r["points"]])
        l93 = sim(h["loc"])
        base = os.path.join(odir, "GEOREF_" + re.sub(r"^RAW_", "", os.path.splitext(row["feuille"])[0]))
        resid = float(np.hypot(v["dE"], v["dN"]))
        loo = tps_loo(px, l93)
        row.update(hypothese=f"{h['xfam']}{h['sx']:+d}{h['sy']:+d}", z=round(v["z"], 1),
                   ecart_cadastre_m=round(resid, 1), rms_affine_m=round(h["rms"], 2),
                   tps_loo_m=None if loo is None else round(loo, 2),
                   appui=f"{len(per)}/{len(ok)}" + (" système" if systeme else ""))
        if not fiable:
            row["statut"] = "à vérifier (une seule feuille d'appui dans la commune)"
        elif mode == "cotes":
            row["statut"] = "ok"
        elif mode == "grille":
            row["statut"] = "ok (placée sur le quadrillage de la commune)"
        elif mode == "grille ?":
            row["statut"] = "à vérifier (placée sur le quadrillage de la commune, sans certitude)"
        elif good_fit(r.get("fit")) and min(r["fit"]["i"]["lines_ok"], r["fit"]["j"]["lines_ok"]) >= 3:
            row["statut"] = "à vérifier (placée par ses cotes, bien lues ; le cadastre actuel ne confirme pas)"
        else:
            row["statut"] = "à vérifier (cotes douteuses, placement incertain)"
        write_points(base + ".points", px, l93)
        PL.control_image(mask, h, sim, C, base + "_controle.jpg", res_m=2.0)
        if not a.no_warp:
            row["resolution_m"] = round(warp(r["file"], base + ".tif", px, l93), 4)
            row["geotiff"] = base + ".tif"
        meta["feuilles"].append({k: row.get(k) for k in row})
        print(f"  {row['feuille']}: {row['statut']} z={row['z']} écart={row['ecart_cadastre_m']} m "
              f"tps={row['tps_loo_m']}", flush=True)
    json.dump(meta, open(os.path.join(odir, "calage.json"), "w"), indent=1, ensure_ascii=False)
    print(f"  total : {time.time() - t0:.0f} s", flush=True)
    return rows


def accord(frame, cands, centres, k93, gamma, tol=20.0):
    """{feuille: (orientation, z)} des feuilles dont un candidat de corrélation tombe, à tol
    près, là où le calage frame met la feuille."""
    Ms = [k93 * np.array([[np.cos(gamma + d), -np.sin(gamma + d)], [np.sin(gamma + d), np.cos(gamma + d)]])
          for d in PL.DELTAS]
    out = {}
    for n, cs in cands.items():
        for c in cs:
            pos = np.asarray(c["O"]) + Ms[c["d"]] @ centres[n][c["h"]]
            if np.hypot(*(pos - frame(centres[n][c["h"]]))) < tol and (n not in out or c["z"] > out[n][1]):
                out[n] = (c["h"], c["z"])
    return out


def choisir_systeme(voisins, cands, centres, sim, chosen, k93, gamma, label, n_ok, emprise):
    """(système, feuilles d'appui) si le calage du système convient mieux, sinon None.

    Le score d'un calage est la somme des z (plafonnés) des feuilles qui l'acceptent. Le
    système l'emporte s'il fait au moins 90 % du score du calage propre ; s'il ne s'en écarte
    pas de plus de 3 m sur l'emprise de la commune, le calage propre est gardé (rien ne bouge)."""
    if not voisins:
        return None

    def score(acc):
        return sum(min(z, 12.0) for _, z in acc.values())
    propre = accord(sim, cands, centres, k93, gamma) if sim is not None else {}
    sp = score(propre)
    best = None
    for s in voisins:
        f = PL.Sim(s["origine_l93"], np.radians(s["rotation_deg"]), s["echelle"])
        acc = accord(f, cands, centres, k93, gamma)
        if best is None or score(acc) > best[2]:
            best = (s, f, score(acc), acc)
    if best is None:
        return None
    s, f, ss, acc = best
    if sim is not None:
        # écart mesuré aux coins de la commune : deux calages d'accord sur quelques feuilles
        # d'appui groupées peuvent diverger de 0,6° ailleurs (Saint-Pol-sur-Mer 1900)
        x0, y0, x1, y1 = emprise
        Mi = np.linalg.inv(sim.M)
        pts = [Mi @ (np.array(c) - sim.O) for c in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
        ecart = max(np.hypot(*(sim(p) - f(p))) for p in pts)
        print(f"  système {np.round(s['origine_l93'])} : score {ss:.0f} contre {sp:.0f} pour le calage "
              f"propre, écart {ecart:.1f} m", flush=True)
        if ecart < 3.0 or ss < 0.9 * sp:
            return None
    elif n_ok and ss < 6.0:
        print(f"  système {np.round(s['origine_l93'])} : aucune feuille ne l'accepte", flush=True)
        return None
    print(f"  {label} : calage du système {np.round(s['origine_l93'])} ({len(s['communes'])} communes, "
          f"rotation {s['rotation_deg']:+.3f}°), {len(acc)} feuilles d'accord", flush=True)
    return s, {n: h for n, (h, _) in acc.items()}


def placer_sur_grille(r, mask, sim, C, steps, occ):
    """Meilleure position d'une feuille sur le quadrillage de la commune, ou None.

    Les meilleures positions de PL.sur_grille sont revues par corrélation fine ; « sur »
    dit si la position est assez nette pour être « ok » : cotes confirmées, cadastre actuel
    d'accord, pas de recouvrement, et une seconde position nettement moins bonne.
    """
    cs = PL.sur_grille(mask, r["points"], r["ocr"], sim, C, steps, occ)
    if not cs:
        return None
    for c in cs[:5]:
        v = PL.verify(mask, c["h"], sim, C, search=15.0, res_m=1.0)
        c["zf"], c["ecart"] = v["z"], float(np.hypot(v["dE"], v["dN"]))
        c["final"] = c["score"] - (3.0 if c["ecart"] > 12.0 else 0.0)
    cs = sorted(cs[:5], key=lambda c: -c["final"])
    best = cs[0]
    cb = best["h"]["loc"].mean(axis=0)
    autres = [c["final"] for c in cs[1:]
              if np.hypot(*(c["h"]["loc"].mean(axis=0) - cb)) > 0.5 * best["h"]["step"]]
    best["marge"] = best["final"] - max(autres) if autres else 99.0
    # plans de ville au 1/500 : corrélation modérée (z 4-8) mais cotes bien lues ; sur un repère
    # fiable, 8 cotes confirmées et aucune position concurrente suffisent (contrôlé à Lille et Saint-Pol)
    best["sur"] = ((best["cotes_ok"] >= 3 and best["z"] >= 10.0 or best["z"] >= 20.0
                    or best["cotes_ok"] >= 8 and best["z"] >= 4.0) and best["ecart"] <= 12.0
                   and (best["recouv"] < 0.4 or best["z"] >= 20.0) and best["dehors"] < 0.35
                   and best["marge"] >= 3.0)
    return best


def rescue(r, mask, sim, C, steps, tol=15.0):
    """Feuille aux cotes illisibles ou fausses, placée grâce au calage de la commune.

    Pour chaque pas et chaque orientation, la corrélation propose des positions ;
    seules comptent celles où le quadrillage de la feuille retombe sur celui de la
    commune (à tol près). Les cotes lues, même partielles, départagent.
    """
    Mi = np.linalg.inv(sim.M)
    readings = {fk: {int(k): v for k, v in r["ocr"][fk].items() if v} for fk in "ij"}
    best = None
    for st in steps:
        zero = {"i": {"a": 0, "step": st}, "j": {"a": 0, "step": st}}
        for h in PL.hypotheses(r["points"], zero):
            A, (x0, y1) = PL.render(mask, sim.M @ h["aff"], PL.RES)
            for v, u, z in C.search(A):
                O = np.array([C.bbox[0] + u * PL.RES - x0, C.bbox[3] - v * PL.RES - y1])
                q = Mi @ (O - sim.O)                       # (est, nord) local du zéro provisoire
                snap = np.round(q / st) * st
                if np.hypot(*(q - snap)) > tol:
                    continue
                yfam = "j" if h["xfam"] == "i" else "i"
                sup = 0
                for fam, off, sg in ((h["xfam"], snap[1], h["sx"]), (yfam, snap[0], h["sy"])):
                    for k, vals in readings[fam].items():
                        pred = abs(int(round(sg * st * k + off)))
                        sup += LB._line_score(pred, vals) >= 0.6
                key = (sup, z)
                if (sup >= 2 or z >= 9.0) and (best is None or key > best[0]):
                    hh = dict(h)
                    hh["loc"] = h["loc"] + snap
                    hh["aff"] = h["aff"].copy()
                    hh["aff"][:, 2] += snap
                    hh["step"] = st
                    best = (key, hh)
    return best[1] if best else None


def commune_sure(d, a):
    try:
        return commune(os.path.join(a.src, d), a)
    except Exception:
        print(f"{d}: ERREUR\n{traceback.format_exc()}", flush=True)
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--min-year", type=int, default=1870,
                    help="ignore les sous-dossiers datés d'avant (0 = tout essayer)")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--no-warp", action="store_true")
    ap.add_argument("--communes-paralleles", type=int, default=1,
                    help="communes traitées en même temps (l'ajustement d'une commune est mono-cœur)")
    ap.add_argument("--systemes", help="fichier de systemes.py : calage commun aux communes d'un "
                    "arrondissement, pour les communes dont le calage propre est faible ou faux")
    ap.add_argument("--sauter-faits", action="store_true",
                    help="ne refait pas un millésime qui a déjà son calage.json")
    a = ap.parse_args()
    a.systemes = SY.charger(a.systemes)
    os.makedirs(a.out, exist_ok=True)
    dirs = [d for d in sorted(os.listdir(a.src))
            if (not a.only or d in a.only) and os.path.isdir(os.path.join(a.src, d))]
    cols = ["commune", "feuille", "annee", "statut", "croisements", "pas_m", "hypothese", "z",
            "ecart_cadastre_m", "appui", "grille", "rms_affine_m", "tps_loo_m", "resolution_m", "geotiff", "source"]
    suffix = "_" + "_".join(a.only)[:60] if a.only else ""
    par_commune = {}

    def ecrire_rapport():
        rows = [r for d in dirs if d in par_commune for r in par_commune[d]]
        with open(os.path.join(a.out, f"rapport{suffix}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        return rows

    if a.communes_paralleles <= 1:
        for d in dirs:
            par_commune[d] = commune_sure(d, a)
            rows = ecrire_rapport()
    else:
        # plusieurs communes à la fois : l'ajustement d'une commune n'occupe qu'un cœur ;
        # l'analyse (OCR) de chacune garde son propre pool, réduit d'autant
        a.jobs = max(1, a.jobs // a.communes_paralleles)
        with ProcessPoolExecutor(a.communes_paralleles) as ex:
            futs = {ex.submit(commune_sure, d, a): d for d in dirs}
            for fut in as_completed(futs):
                par_commune[futs[fut]] = fut.result()
                rows = ecrire_rapport()
    rows = ecrire_rapport()
    n_ok = sum(r["statut"] == "ok" for r in rows)
    print(f"TERMINE : {len(rows)} feuilles vues, {n_ok} géoréférencées « ok »", flush=True)


if __name__ == "__main__":
    main()
