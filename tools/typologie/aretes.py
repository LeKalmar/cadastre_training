"""Jeu d'entraînement du regroupement des typologies en parcelles principales.

Une parcelle principale (un seul numéro sur le plan) réunit plusieurs typologies (bâti, cour…).
Selon le géomètre, ce qui les relie se voit : trait noir fort autour de la parcelle principale
mais fin ou absent entre ses typologies, flèches ou « traits sans bout » qui relient deux
polygones. On apprend donc à classer chaque ARÊTE COMMUNE entre deux typologies voisines :
    1 = même parcelle principale (même par_main_id GénéFoncier)
    0 = limite entre deux parcelles principales
Exclus : typologie sans par_main_id (jamais rattachée), parcelle non « valide » ou au lien
douteux, développement transporté, et toute parcelle qui touche le bord de sa feuille
(--bord-m) : en bord de feuille, GénéFoncier a recalé les parcelles par compromis avec les
feuilles voisines, elles ne collent pas au dessin.

Sorties (par split, train/val par feuille comme typologie_v1) :
    images/<id>.jpg  vignette du scan redressée : l'arête est horizontale, au centre
    aux/<id>.png     R = l'arête, G = polygone du dessus, B = polygone du dessous
    aretes.csv       étiquette, feuille, types des deux côtés, longueur, résolution

    .venv/bin/python tools/typologie/aretes.py --out data/regroupement_v1
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.windows import transform as win_transform
from shapely import STRtree
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "annotation"))
import common as C  # noqa: E402
from build_dataset import slug, split_de  # noqa: E402

PH, PW = 128, 192          # vignette : hauteur (de part et d'autre de l'arête), longueur le long de l'arête


def partage(a, b, tol):
    """Partie commune des contours de a et b (tolérance tol m), en morceaux continus."""
    s = b.boundary.intersection(a.buffer(tol)).intersection(a.boundary.buffer(tol))
    if s.is_empty:
        return []
    lines = [g for g in getattr(s, "geoms", [s]) if isinstance(g, (LineString, MultiLineString))]
    if not lines:
        return []
    m = linemerge(MultiLineString([p for g in lines for p in getattr(g, "geoms", [g])]))
    return list(getattr(m, "geoms", [m]))


def vignettes(src, a, b, ligne, px, pas_px):
    """Vignettes centrées sur des points de l'arête (une tous les pas_px), redressées."""
    n = max(1, int(ligne.length / px // pas_px))
    out = []
    for k in range(n):
        t = (k + 0.5) / n
        p = ligne.interpolate(t, normalized=True)
        q0 = ligne.interpolate(max(0.0, t - 0.5 / n), normalized=True)
        q1 = ligne.interpolate(min(1.0, t + 0.5 / n), normalized=True)
        inv = ~src.transform
        c, r = inv * (p.x, p.y)
        c0, r0 = inv * (q0.x, q0.y)
        c1, r1 = inv * (q1.x, q1.y)
        dxp, dyp = c1 - c0, r1 - r0
        L = math.hypot(dxp, dyp)
        if L < 1e-6:
            continue
        d = np.array([dxp, dyp]) / L
        nrm = np.array([-d[1], d[0]])
        R = int(math.ceil(math.hypot(PW / 2, PH / 2))) + 2
        x0, y0 = int(c) - R, int(r) - R
        if x0 < 0 or y0 < 0 or x0 + 2 * R > src.width or y0 + 2 * R > src.height:
            continue
        win = Window(x0, y0, 2 * R, 2 * R)
        rgb = np.transpose(src.read([1, 2, 3], window=win), (1, 2, 0))
        if (src.read_masks(1, window=win) == 0).mean() > 0.05:
            continue
        T = win_transform(win, src.transform)
        aux = np.zeros((2 * R, 2 * R, 3), np.uint8)
        aux[..., 0] = rasterize([(ligne, 255)], out_shape=(2 * R, 2 * R), transform=T, all_touched=True, dtype="uint8")
        aux[..., 1] = rasterize([(a, 255)], out_shape=(2 * R, 2 * R), transform=T, dtype="uint8")
        aux[..., 2] = rasterize([(b, 255)], out_shape=(2 * R, 2 * R), transform=T, dtype="uint8")
        # (u, v) de la vignette -> fenêtre : centre + (u - PW/2)·d + (v - PH/2)·n
        cx, cy = c - x0, r - y0
        M = np.array([[d[0], nrm[0], cx - d[0] * PW / 2 - nrm[0] * PH / 2],
                      [d[1], nrm[1], cy - d[1] * PW / 2 - nrm[1] * PH / 2]], np.float32)
        fl = cv2.WARP_INVERSE_MAP
        img = cv2.warpAffine(rgb, M, (PW, PH), flags=cv2.INTER_LINEAR | fl, borderMode=cv2.BORDER_REPLICATE)
        ax = cv2.warpAffine(aux, M, (PW, PH), flags=cv2.INTER_NEAREST | fl)
        # convention : polygone a en haut (G). Sinon on retourne la vignette.
        haut, bas = ax[: PH // 2, :, 1].mean(), ax[PH // 2:, :, 1].mean()
        if bas > haut:
            img, ax = img[::-1].copy(), ax[::-1].copy()
        out.append((img, ax))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(C.ROOT / "data" / "regroupement_v1"))
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--bord-m", type=float, default=3.0, help="parcelle exclue si elle passe à moins de ... m du bord de la feuille")
    ap.add_argument("--tol-m", type=float, default=0.3, help="tolérance de contact entre deux contours")
    ap.add_argument("--min-px", type=float, default=24, help="longueur minimale d'arête (px du scan)")
    ap.add_argument("--pas-px", type=float, default=160, help="une vignette tous les ... px le long d'une arête")
    ap.add_argument("--max-par-arete", type=int, default=3)
    ap.add_argument("--only", nargs="*", help="noms de feuilles (test)")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} existe déjà et n'est pas vide.")
    for sp in ("train", "val"):
        for sub in ("images", "aux"):
            (out / sp / sub).mkdir(parents=True, exist_ok=True)

    cur = C.connect().cursor()
    feuilles = cur.execute(
        "SELECT f.id, f.nom, f.raster, f.split, ST_AsBinary(f.geom) FROM feuilles f WHERE f.raster IS NOT NULL "
        "AND (SELECT count(*) FROM parcelles p WHERE p.feuille_id = f.id AND p.par_main_id IS NOT NULL) >= 20 "
        "ORDER BY f.nom").fetchall()
    rows, bilan = [], Counter()
    for fid, nom, raster, split, fg in feuilles:
        if not Path(raster).exists() or (a.only and nom not in a.only):
            continue
        bord = C.geom_of(fg).boundary.buffer(a.bord_m)
        ps = []
        for pid, main, t, g in cur.execute(
                "SELECT id, par_main_id, type, ST_AsBinary(geom) FROM parcelles WHERE feuille_id = %s "
                "AND statut = 'valide' AND lien = 'ok' AND developpement IS NULL", (fid,)):
            g = C.geom_of(g)
            ps.append((str(pid), str(main) if main else None, t or "", g, g.intersects(bord)))
        geoms = [p[3] for p in ps]
        tree = STRtree(geoms)
        sp = split_de(nom, split, a.val_frac)
        name = f"{slug(nom)}_{str(fid)[:6]}"
        n_f = Counter()
        with rasterio.open(raster) as src:
            px = abs(src.transform.a)
            for i, (pid, main, t, g, au_bord) in enumerate(ps):
                if main is None or au_bord:
                    continue
                for j in tree.query(g.buffer(a.tol_m)):
                    j = int(j)
                    if j <= i:
                        continue
                    pid2, main2, t2, g2, bord2 = ps[j]
                    if main2 is None or bord2:
                        continue
                    for k, ligne in enumerate(partage(g, g2, a.tol_m)):
                        if ligne.length / px < a.min_px:
                            continue
                        lab = int(main == main2)
                        for m, (img, ax) in enumerate(vignettes(src, g, g2, ligne, px, a.pas_px)[: a.max_par_arete]):
                            eid = f"{name}_{pid[:8]}_{pid2[:8]}_{k}_{m}"
                            Image.fromarray(img).save(out / sp / "images" / f"{eid}.jpg", quality=95)
                            Image.fromarray(ax).save(out / sp / "aux" / f"{eid}.png", optimize=True)
                            rows.append({"id": eid, "label": lab, "split": sp, "feuille": nom,
                                         "feuille_id": str(fid), "type_a": t, "type_b": t2,
                                         "long_px": round(ligne.length / px, 1), "px_m": round(px, 4)})
                            n_f[lab] += 1
        bilan.update({f"{sp}:{k}": v for k, v in n_f.items()})
        print(f"  {sp:5} {nom[:48]:48} même parcelle {n_f[1]:5}  limite {n_f[0]:5}", flush=True)
    if not rows:
        sys.exit("Aucune arête.")
    with open(out / "aretes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    resume = {"vignette": [PH, PW], "bord_m": a.bord_m, "tol_m": a.tol_m, "min_px": a.min_px,
              "pas_px": a.pas_px, "bilan": dict(sorted(bilan.items())),
              "val_feuilles": sorted({r["feuille"] for r in rows if r["split"] == "val"})}
    (out / "summary.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False))
    print(json.dumps(resume, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
