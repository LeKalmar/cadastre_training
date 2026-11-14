#!/usr/bin/env python3
"""Compare des modèles × variantes de post-traitement sur les 10 feuilles de validation.

    # 1. inférence (GPU, une feuille à la fois)
    .venv/bin/python tools/eval_models/run_eval.py infer --model v3_fin=work_dirs/v3_fin/best_model.pth
    # 2. polygonisation + score (CPU, parallèle)
    .venv/bin/python tools/eval_models/run_eval.py score --models v2 v3_fin --variants base gaps --jobs 3
    # 3. tableau
    .venv/bin/python tools/eval_models/run_eval.py table

Dossiers : work_dirs/eval/<modèle>/<feuille>/{mask.tif, prob.tif, text.tif, <variante>.geojson}
Vérité terrain : work_dirs/refiner/<feuille>/gt.geojson (même protocole que le correcteur).
Le modèle « v2 » réutilise les masques/probas de work_dirs/refiner (même modèle, seuil 0.30).
Tout est reprenable : un résultat déjà calculé n'est pas recalculé.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rasterio

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "refiner"))
sys.path.insert(0, "/home/lekalmar/PyCharmMiscProject/Code/Geenfoncier/GeneFoncier/terrapartus")

EVAL = ROOT / "work_dirs/eval"
REF = ROOT / "work_dirs/refiner"


def val_sheets():
    from prepare import sheets, slug
    return [(rp, slug(m["group"])) for rp, m in sheets().items() if m["split"] == "val"]


def sheet_classes(rural_min=0.3):
    """{feuille: 'rural' | 'urbain'} : une feuille est rurale quand plus de la moitié de
    ses tuiles le sont (rural_frac >= 0.3, comme le suréchantillonnage de train.py)."""
    import csv
    from prepare import TILES, slug
    acc = {}
    for t in csv.DictReader(open(TILES)):
        acc.setdefault(slug(t["group"]), []).append(float(t["rural_frac"]) >= rural_min)
    return {s: "rural" if np.mean(v) > 0.5 else "urbain" for s, v in acc.items()}


def sheet_dir(model, s):
    return (REF / s) if model == "v2" else (EVAL / model / s)


# ----------------------------------------------------------------------------- infer
def upscaled(rp, target_res, tmpdir):
    """Copie agrandie (cubique) de la feuille si elle est plus grossière que target_res."""
    from rasterio.enums import Resampling
    with rasterio.open(rp) as src:
        res = abs(src.transform.a)
        if not target_res or res <= target_res * 1.02:
            return rp, 1.0
        f = res / target_res
        H, W = int(round(src.height * f)), int(round(src.width * f))
        data = src.read(out_shape=(src.count, H, W), resampling=Resampling.cubic)
        prof = src.profile
        prof.pop("photometric", None)          # YCbCr (JPEG) ne passe pas en deflate
        prof.update(width=W, height=H, transform=src.transform * rasterio.Affine.scale(src.width / W, src.height / H),
                    compress="deflate", tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
        msk = None
        if src.count < 4:
            msk = src.read_masks(1, out_shape=(H, W), resampling=Resampling.nearest)
    out = Path(tmpdir) / "up.tif"
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(data)
            if msk is not None and (msk == 0).any():
                dst.write_mask(msk)
    del data
    return str(out), f


def infer(name, ckpt, threshold=0.30, target_res=None):
    import tempfile
    import torch
    from georeferencing import inference

    for rp0, s in val_sheets():
        d = sheet_dir(name, s)
        d.mkdir(parents=True, exist_ok=True)
        if (d / "prob.tif").exists():
            print(f"{name} {s[:40]} déjà fait", flush=True)
            continue
        t0 = time.time()
        tdir = tempfile.mkdtemp(dir=str(d))
        rp, factor = upscaled(rp0, target_res, tdir)
        tmp = d / "prob_f32.tif"
        seg = inference.segment_geotiff(rp, str(d / "mask.tif"), str(ckpt),
                                        params=inference.InferenceParams(threshold=threshold),
                                        prob_output_path=str(tmp))
        with rasterio.open(tmp) as src:
            p = src.read(1)
            prof = src.profile
        prof.update(dtype="uint8", nodata=None, compress="deflate", predictor=2)
        with rasterio.open(d / "prob.tif", "w", **prof) as dst:
            dst.write(np.clip(np.round(p * 255), 0, 255).astype(np.uint8), 1)
        tmp.unlink()
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        if int(ck.get("config", {}).get("classes", 1)) > 1:
            text_map(rp, ckpt, d / "text.tif", prof)
        seg = dict(seg or {}, upscale=round(factor, 3))
        (d / "seg.json").write_text(json.dumps(seg, indent=1, default=str))
        import shutil
        shutil.rmtree(tdir, ignore_errors=True)
        print(f"{name} {s[:40]} {time.time() - t0:.0f}s", flush=True)
        torch.cuda.empty_cache()


def text_map(raster, ckpt, out, prof):
    """Sortie 1 (texte/numéros) du modèle multi-tâche, même tuilage que les traits."""
    from georeferencing import inference

    model, device = inference.load_model(str(ckpt))
    with rasterio.open(raster) as src:
        img = np.moveaxis(src.read([1, 2, 3]), 0, -1)
    _, p = inference.predict_with_overlap(model, img, device, inference.InferenceParams(), channel=1)
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(np.clip(np.round(p * 255), 0, 255).astype(np.uint8), 1)
    del model


# ----------------------------------------------------------------------------- score
def make_mask(d, variant, out_path):
    """Masque de la variante, avec le même profil et le même masque papier que mask.tif."""
    from gaps import close_gaps

    with rasterio.open(d / "mask.tif") as src:
        m = src.read(1) > 0
        valid = src.dataset_mask() > 0
        prof = src.profile
        res = abs(src.transform.a)
    with rasterio.open(d / "prob.tif") as src:
        p = src.read(1).astype(np.float32) / 255.0
    stats = {}
    burned = m & ~(p > 0.30)          # contour papier brûlé par segment_geotiff
    if variant == "base":
        out = m
    elif variant.startswith("t"):      # t50 = seuil 0.50 (+ contour papier)
        out = ((p > int(variant[1:]) / 100) | burned) & valid
    elif variant == "gaps":
        out, stats = close_gaps(m, p, res, valid=valid)
    elif variant == "gaps_link":       # raccord seul, sans hystérésis
        out, stats = close_gaps(m, p, res, valid=valid, low=None)
    elif variant in ("gaps_p", "gaps_link_p"):   # + moignon min 1,5 m + élagage
        out, stats = close_gaps(m, p, res, valid=valid, low=0.12 if variant == "gaps_p" else None,
                                min_trace_m=1.5, do_prune=True)
    elif variant == "gaps_nohatch":    # raccord sans le test des hachures (témoin)
        out, stats = close_gaps(m, p, res, valid=valid, hatch_min=0)
    else:
        raise ValueError(variant)
    prof.update(nodata=None)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write((out * 255).astype(np.uint8), 1)
            dst.write_mask(valid)
    return stats


def score_one(job):
    model, s, variant = job
    from features import load_polys
    from georeferencing import grass_vectorize
    from georeferencing import polygon_refiner as pr
    from train_eval import score

    d = sheet_dir(model, s)
    out = EVAL / model / s
    out.mkdir(parents=True, exist_ok=True)
    full, variant = variant, variant.split("@")[0]   # gaps@c0j6 : réglages de polygonisation
    pkw = {}
    for opt in (full.split("@")[1].split("+") if "@" in full else []):
        if opt == "c0":
            pkw["line_close_iter"] = 0
        elif opt.startswith("j"):
            pkw["junction_merge_px"] = float(opt[1:])
        elif opt.startswith("s"):
            pkw["simplify_px"] = float(opt[1:])
        elif opt.startswith("p"):                    # p20 : DP par arc à 2,0 px du scan
            pkw["simplify_method"] = "arcs"
            pkw["simplify_dp_m"] = float(opt[1:]) / 10 * abs(rasterio.open(d / "prob.tif").transform.a)
        elif opt.startswith("a"):                    # a30 : DP par arc à 0,30 m
            pkw["simplify_method"] = "arcs"
            pkw["simplify_dp_m"] = float(opt[1:]) / 100
        elif opt == "vw":                            # ancienne simplification (Visvalingam)
            pkw["simplify_method"] = "coverage"
    res_path = out / f"{full}.score.json"
    if res_path.exists():
        r = json.loads(res_path.read_text())
        if "bF1@3px" in r or not (out / f"{full}.geojson").exists():
            return r
        # ancien score sans mesures en pixels : complété depuis le geojson (pas de GRASS)
        polys, _ = load_polys(out / f"{full}.geojson")
        gt, _ = load_polys(REF / s / "gt.geojson")
        res = abs(rasterio.open(d / "prob.tif").transform.a)
        r.update(score(polys, pr.Truth.from_parcels(gt), res))
        res_path.write_text(json.dumps(r, default=float))
        return r
    if not (d / "prob.tif").exists() or not (REF / s / "gt.geojson").exists():
        return None
    t0 = time.time()
    gstats = {}
    if variant == "base" and model != "v2":
        mpath = d / "mask.tif"
    else:
        mpath = out / f"mask_{full}.tif"
        gstats = make_mask(d, variant, mpath)
    feats, rep = grass_vectorize.grass_vectorize_mask(str(mpath), target_srid=2154, **pkw)
    (out / f"{full}.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    if mpath.name.startswith("mask_"):
        mpath.unlink()
    polys, _ = load_polys(out / f"{full}.geojson")
    gt, _ = load_polys(REF / s / "gt.geojson")
    truth = pr.Truth.from_parcels(gt)
    res = abs(rasterio.open(d / "prob.tif").transform.a)
    r = dict(model=model, sheet=s, variant=full, **score(polys, truth, res), gaps=gstats,
             secs=round(time.time() - t0))
    simp = pr.simplify_coverage(polys, 10 * res)
    r["simp10"] = score(simp, truth)
    res_path.write_text(json.dumps(r, default=float))
    return r


def run_scores(models, variants, jobs):
    todo = [(m, s, v) for m in models for _, s in val_sheets() for v in variants]
    with Pool(jobs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(score_one, todo):
            if r:
                print(f"{r['model']:14} {r['variant']:10} {r['sheet'][:36]:36} matched={r['matched']:5} "
                      f"under={r['under_seg']:4} bF1.3={r['bF1@0.3']:.3f} {r['secs']}s {r.get('gaps') or ''}",
                      flush=True)


def _f1(r):
    p, q = r["matched"] / max(r["n_pred"], 1), r["matched"] / max(r["n_ref"], 1)
    return 2 * p * q / max(p + q, 1e-9)


def table(only=None):
    """Deux tableaux : les totaux (historique, dominés par les 3 grosses feuilles de
    ville) puis la moyenne PAR FEUILLE, rural et urbain à parts égales, tolérances en px."""
    rows = [json.loads(p.read_text()) for p in EVAL.glob("*/*/*.score.json")]
    if only:
        rows = [r for r in rows if any(o in r["variant"] for o in only)]
    keys = ["n_pred", "matched", "missed", "spurious", "over_seg", "under_seg", "vertices"]
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["variant"]), []).append(r)
    n_sheets = len(val_sheets())
    lines = ["TOTAUX (somme des parcelles : 76 % viennent de 3 feuilles de ville)",
             f"{'modèle':14} {'variante':18} {'n':>2} " + " ".join(f"{k:>9}" for k in keys)
             + "  bF1@.3  bF1@.6"]
    for (m, v), rs in sorted(groups.items()):
        tot = {k: sum(x[k] for x in rs) for k in keys}
        f3 = np.mean([x["bF1@0.3"] for x in rs]); f6 = np.mean([x["bF1@0.6"] for x in rs])
        flag = "" if len(rs) == n_sheets else "  (incomplet)"
        lines.append(f"{m:14} {v:18} {len(rs):2} " + " ".join(f"{tot[k]:9d}" for k in keys)
                     + f"  {f3:.4f}  {f6:.4f}{flag}")

    cls = sheet_classes()
    lines += ["", "PAR FEUILLE (moyenne des feuilles ; équilibré = moyenne rural / urbain)",
              "  F1 = parcelles retrouvées (rappel x précision) ; bF1 = limites à moins de 3 px ;",
              "  som/GT = sommets du réseau / sommets de la vérité terrain ; alignés = sommets inutiles (virage < 3°)",
              f"{'modèle':14} {'variante':18} {'F1 équil.':>9} | {'rural F1':>8} {'bF1@3px':>7} {'som/GT':>6} {'alignés':>7}"
              f" | {'urbain F1':>9} {'bF1@3px':>7} {'som/GT':>6} {'alignés':>7}"]
    out = []
    for (m, v), rs in groups.items():
        if len(rs) != n_sheets or any("bF1@3px" not in x for x in rs):
            continue
        cell = {}
        for c in ("rural", "urbain"):
            sub = [x for x in rs if cls.get(x["sheet"]) == c]
            cell[c] = (np.mean([_f1(x) for x in sub]), np.mean([x["bF1@3px"] for x in sub]),
                       np.mean([x["vertices"] / max(x["gt_vertices"], 1) for x in sub]),
                       np.mean([x["fake_v"] for x in sub])) if sub else (0, 0, 0, 0)
        out.append(((cell["rural"][0] + cell["urbain"][0]) / 2, m, v, cell))
    for bal, m, v, cell in sorted(out, key=lambda t: -t[0]):
        lines.append(f"{m:14} {v:18} {bal:9.4f} | " + " | ".join(
            f"{cell[c][0]:{w}.4f} {cell[c][1]:7.4f} {cell[c][2]:6.2f} {100 * cell[c][3]:6.1f}%"
            for c, w in (("rural", 8), ("urbain", 9))))
    n_ref = {r["sheet"]: r["n_ref"] for r in rows}
    lines.append(f"n_ref total : {sum(n_ref.values())} | feuilles rurales : "
                 f"{sum(cls.get(s) == 'rural' for s in n_ref)}, urbaines : {sum(cls.get(s) == 'urbain' for s in n_ref)}")
    txt = "\n".join(lines)
    if not only:
        (EVAL / "tableau.txt").write_text(txt + "\n")
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("infer")
    a1.add_argument("--model", required=True, help="nom=chemin.pth")
    a1.add_argument("--target-res", type=float, default=None,
                    help="agrandit les feuilles plus grossières que cette résolution (m/px)")
    a2 = sub.add_parser("score")
    a2.add_argument("--models", nargs="+", required=True)
    a2.add_argument("--variants", nargs="+", default=["base", "gaps"])
    a2.add_argument("--jobs", type=int, default=3)
    a3 = sub.add_parser("table")
    a3.add_argument("--only", nargs="*", help="sous-chaînes de variantes à afficher (n'écrit pas tableau.txt)")
    a = ap.parse_args()
    if a.cmd == "infer":
        name, ck = a.model.split("=", 1)
        infer(name, ROOT / ck if not Path(ck).is_absolute() else Path(ck), target_res=a.target_res)
    elif a.cmd == "score":
        run_scores(a.models, a.variants, a.jobs)
    else:
        table(a.only)


if __name__ == "__main__":
    main()
