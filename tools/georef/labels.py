"""Lecture des cotes écrites au bout de chaque trait du quadrillage, près du cadre,
puis ajustement robuste de la numérotation (progression arithmétique).

Pour chaque trait et chacune de ses deux extrémités : on avance du dernier
croisement jusqu'au cadre, on redresse une bande le long du trait et on la lit à
l'endroit et à l'envers. Les cotes sont non signées (|a + pas·k|).
"""
import re

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
L, HH = 560, 70           # longueur de la bande le long du trait, demi-hauteur
OUT = 170                 # prolongement au-delà du cadre pour les cotes écrites en marge
STEPS = (100, 200, 250, 50, 500, 400, 20, 25, 40, 1000)
ASSEMBLAGE_PAS_M = 500        # quadrillage des tableaux d'assemblage (1/10 000 : 1000 m)
_OCR = None


def _ocr():
    global _OCR
    if _OCR is None:
        # 2 fils par processus : sinon chaque lecteur prend tous les cœurs et les
        # feuilles traitées en parallèle se marchent dessus
        import rapidocr_onnxruntime.utils as U
        from rapidocr_onnxruntime import RapidOCR
        base = U.SessionOptions

        def opts():
            o = base()
            o.intra_op_num_threads = 2
            o.inter_op_num_threads = 1
            return o
        U.SessionOptions = opts
        # bandes lues à leur taille réelle : le réglage par défaut les agrandit ~28x
        # (13x plus lent, sans gain de lecture sur les feuilles de test)
        _OCR = RapidOCR(det_limit_side_len=960, det_limit_type="max", det_model_path="")
    return _OCR


def _frame_dist(gray, x, y, ux, uy, maxd):
    """Distance du point au cadre (trait noir épais) dans la direction (ux, uy)."""
    H, W = gray.shape
    run = 0
    for t in range(40, maxd, 2):
        px, py = int(x + ux * t), int(y + uy * t)
        if not (0 <= px < W and 0 <= py < H):
            return t, False
        if gray[py, px] < 85:
            run += 2
            if run >= 6:
                return t - run, True
        else:
            run = 0
    return maxd, False


def _strip(im, p, d, t):
    """Bande couvrant t-L..t le long de d depuis p, trait sur la ligne médiane."""
    dx, dy = d
    nx, ny = -dy, dx
    t0 = t - L - 6
    cx = p[0] + dx * t0 - nx * HH
    cy = p[1] + dy * t0 - ny * HH
    return im.transform((L, 2 * HH), Image.AFFINE, (dx, nx, cx, dy, ny, cy),
                        resample=Image.BILINEAR, fillcolor=(255, 255, 255))


def _prep(strip):
    """Deux rendus de la bande : vert contrasté, et gris doux agrandi près du cadre.

    Le vert sépare bien l'encre rouge du papier ; l'agrandissement aide pour les
    petites cotes écrites sur le trait (plans des années 1870).
    """
    a = np.asarray(strip).astype(np.float32)
    g = np.minimum(a[..., 1], a[..., 2] * 0.5 + a[..., 1] * 0.5)
    lo, hi = np.percentile(g, 1), np.percentile(g, 70)
    g = np.clip((g - lo) / max(hi - lo, 20) * 255, 0, 255).astype(np.uint8)
    m = a.mean(axis=2)
    lo, hi = np.percentile(m, 0.5), np.percentile(m, 50)
    m = np.clip((m - lo) / max(hi - lo, 40) * 255, 0, 255).astype(np.uint8)
    m = Image.fromarray(m).crop((L - 400, 0, L, 2 * HH)).resize((800, 4 * HH), Image.LANCZOS)
    m = np.asarray(m)
    return [np.dstack([g, g, g]), np.dstack([m, m, m])]


def _values(arr):
    out = []
    for rot in (0, 2):
        res, _ = _ocr()(np.ascontiguousarray(np.rot90(arr, rot)))
        for _, txt, conf in res or []:
            s = (txt.replace("O", "0").replace("o", "0").replace("l", "1")
                 .replace("I", "1").replace(" ", "").replace(".", "").replace(",", ""))
            for m in re.findall(r"\d+", s):
                v = int(m)
                if v % 5 == 0 and v <= 30000 and float(conf) > 0.45 and len(m) <= 5:
                    out.append(v)
    return out


def read(res, im=None, debug_dir=None):
    """Lectures par trait : {"i": {indice: [valeurs]}, "j": {...}}."""
    im = im or Image.open(res["file"]).convert("RGB")
    gray = np.asarray(im.convert("L"))
    pts = res["points"]
    P = res["families"][0]["period_px"]
    rec = {"i": {}, "j": {}}
    n = 0
    for fk, ok in (("i", "j"), ("j", "i")):
        for idx in sorted({p[fk] for p in pts}):
            sel = sorted((p for p in pts if p[fk] == idx), key=lambda p: p[ok])
            if len(sel) < 2:
                continue
            a, b = sel[0], sel[-1]
            d = np.array([b["x"] - a["x"], b["y"] - a["y"]], float)
            d /= np.linalg.norm(d)
            vals = []
            for p, u in ((a, -d), (b, d)):
                t, found = _frame_dist(gray, p["x"], p["y"], u[0], u[1], int(3.2 * P))
                if not found and t >= int(3.2 * P):
                    continue
                strips = _prep(_strip(im, (p["x"], p["y"]), u, t))
                if found:
                    # cotes écrites au-delà du cadre, dans la marge (plans au 1/4000
                    # des années 1880, ex. Trélon) : bande prolongée de OUT px
                    strips += _prep(_strip(im, (p["x"], p["y"]), u, t + OUT))
                for st in strips:
                    if debug_dir:
                        Image.fromarray(st).save(f"{debug_dir}/{fk}{idx}_{n}.png")
                        n += 1
                    vals += _values(st)
            rec[fk][str(idx)] = vals
    return rec


def _line_score(pred, vals):
    best = -0.3            # des lectures existent mais aucune n'explique la cote
    for v in vals:
        if v == pred:
            sc = 0.3 if v == 0 else 1.0     # un « 0 » isolé est souvent un rond mal lu
        elif pred >= 1000 and v == pred % 1000 and v != 0:
            sc = 0.6                        # premier chiffre perdu
        elif pred >= 100 and v == pred // 10 and pred % 10 == 0 and v >= 10:
            sc = 0.4                        # dernier zéro perdu
        else:
            continue
        best = max(best, sc)
    return best


def fit_family(readings, steps=STEPS):
    """readings {k: [valeurs]} -> dict(a, step, lines_ok, lines_read, score) ou None."""
    ks = [k for k, v in readings.items() if v]
    if len(ks) < 2:
        return None
    best = None
    for rank, step in enumerate(steps):
        # les cotes sont des multiples du pas
        for a in range(-30000 // step * step, 30001, step):
            total, lines = 0.0, 0
            for k in ks:
                sc = _line_score(abs(a + step * k), readings[k])
                total += sc
                lines += sc >= 0.6
            key = (round(total, 3), -rank)
            if best is None or key > best[0]:
                best = (key, a, step, lines)
    (total, _), a, step, lines = best
    return {"a": a, "step": step, "lines_ok": lines, "lines_read": len(ks), "score": total}


def fit(rec, step=None):
    """Ajuste les deux familles avec le même pas (celui qui explique le mieux)."""
    r = {fk: {int(k): v for k, v in rec[fk].items()} for fk in ("i", "j")}
    best = None
    for rank, st in enumerate([step] if step else STEPS):
        fi, fj = fit_family(r["i"], (st,)), fit_family(r["j"], (st,))
        if not fi or not fj:
            continue
        key = (fi["lines_ok"] + fj["lines_ok"], round(fi["score"] + fj["score"], 3), -rank)
        if best is None or key > best[0]:
            best = (key, {"i": fi, "j": fj, "step": st})
    return best[1] if best else None


if __name__ == "__main__":
    import json
    import os
    import sys
    import time
    for f in sys.argv[1:]:
        res = json.load(open(f))
        dbg = None
        if os.environ.get("DEBUG_STRIPS"):
            dbg = f.rsplit(".", 1)[0] + "_bandes"
            os.makedirs(dbg, exist_ok=True)
        t = time.time()
        rec = read(res, debug_dir=dbg)
        ft = fit(rec)
        print(os.path.basename(f), f"{time.time() - t:.0f}s")
        print("  i", {k: v for k, v in rec["i"].items()})
        print("  j", {k: v for k, v in rec["j"].items()})
        print("  ->", ft, flush=True)
        json.dump({"ocr": rec, "fit": ft}, open(f.rsplit(".", 1)[0] + "_cotes.json", "w"))
