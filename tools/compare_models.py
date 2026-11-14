#!/usr/bin/env python3
"""
Compare des modèles sur les tuiles de validation du dataset v2 (feuilles jamais vues).

Métriques (pixels "ignorés" du masque exclus) :
  - pixel   : IoU / précision / rappel / F1 contre le masque (cœur 3 px)
  - squelette (indépendant de l'épaisseur du trait) : on squelettise prédiction et
    vérité terrain ; précision = part du squelette prédit à <= TOL px d'une vraie ligne,
    rappel = part du squelette vrai à <= TOL px d'une ligne prédite.
Ventilation rural / urbain via tiles.csv.

    .venv/bin/python tools/compare_models.py --data data/cadastre_lines_v2 \
        --model "Bertrand=work_dirs/Bertrand_Couderc_mitb3_topology/best_model.pth" \
        --model "v2=work_dirs/v2_mitb3_test/best_model.pth" --scales 1 0.5
"""
import argparse, csv, json, sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.morphology import skeletonize
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train import build_model  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
TOL = 3


def load(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    enc = (ck.get("config") or {}).get("encoder")
    if not enc:
        cfg = Path(ckpt).parent / "config.json"
        enc = json.loads(cfg.read_text())["encoder"] if cfg.exists() else "resnet34"
    m = build_model(enc, pretrained=False).to(dev)
    m.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    return m.eval(), enc


@torch.no_grad()
def predict(model, imgs, scale, dev):
    x = torch.from_numpy(((imgs / 255.0 - MEAN) / STD).transpose(0, 3, 1, 2).astype(np.float32)).to(dev)
    h, w = x.shape[2:]
    if scale != 1:
        x = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
    with torch.amp.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        y = model(x)
    y = torch.sigmoid(y.float())
    if scale != 1:
        y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
    return y[:, 0].cpu().numpy()


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def tile_stats(prob, mask, thr):
    gt = mask == 255; valid = mask != 100
    pr = prob > thr
    s = {}
    s["tp"] = float((pr & gt & valid).sum()); s["fp"] = float((pr & ~gt & valid).sum())
    s["fn"] = float((~pr & gt & valid).sum())
    sk_p = skeletonize(pr) & valid
    sk_g = skeletonize(gt)
    near_g = cv2.dilate(gt.astype(np.uint8), disk(TOL)) > 0
    near_p = cv2.dilate(pr.astype(np.uint8), disk(TOL)) > 0
    s["sk_p"] = float(sk_p.sum()); s["sk_p_ok"] = float((sk_p & near_g).sum())
    s["sk_g"] = float(sk_g.sum()); s["sk_g_ok"] = float((sk_g & near_p).sum())
    # parcelles fermées retrouvées : régions entre lignes (vérité) appariées à une région prédite IoU > 0.8
    n_g, lab_g = cv2.connectedComponents((~gt).astype(np.uint8), connectivity=4)
    n_p, lab_p = cv2.connectedComponents((~pr).astype(np.uint8), connectivity=4)
    ign = ~valid & ~gt
    tot = ok = 0
    for k in range(1, n_g):
        reg = lab_g == k; area = reg.sum()
        if area < 400 or (reg & ign).sum() > 0.1 * area: continue  # trop petite ou zone non annotée
        tot += 1
        lp = lab_p[reg]; lp = lp[lp > 0]
        if lp.size == 0: continue
        best = np.bincount(lp).argmax(); inter = (lp == best).sum()
        if inter / (area + (lab_p == best).sum() - inter) > 0.8: ok += 1
    s["reg"] = float(tot); s["reg_ok"] = float(ok)
    return s


def summarize(acc):
    tp, fp, fn = acc["tp"], acc["fp"], acc["fn"]
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    sp = acc["sk_p_ok"] / max(acc["sk_p"], 1); sr = acc["sk_g_ok"] / max(acc["sk_g"], 1)
    return dict(iou=tp / max(tp + fp + fn, 1), precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-9),
                skel_precision=sp, skel_recall=sr, skel_f1=2 * sp * sr / max(sp + sr, 1e-9),
                parcelles=acc["reg_ok"] / max(acc["reg"], 1), n_parcelles=int(acc["reg"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cadastre_lines_v2")
    ap.add_argument("--model", action="append", required=True, help="nom=chemin/checkpoint.pth")
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6])
    ap.add_argument("--split", default="val"); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None, help="JSON des résultats")
    ap.add_argument("--viz", default=None, help="PNG de comparaison visuelle")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    if dev.type == "cpu": torch.set_num_threads(8)
    D = Path(a.data)
    rural = {r["tile"]: float(r["rural_frac"]) >= 0.3 for r in csv.DictReader(open(D / "tiles.csv"))}
    files = sorted((D / "images" / a.split).glob("*.jpg"))
    if a.limit: files = files[:: max(1, len(files) // a.limit)]
    print(f"{len(files)} tuiles ({a.split}) dont {sum(rural.get(f.stem, False) for f in files)} rurales")

    results, viz_cols = {}, {}
    for spec in a.model:
        name, ckpt = spec.split("=", 1)
        model, enc = load(ckpt, dev)
        for sc in a.scales:
            key = f"{name} (x{sc:g})"
            acc = {t: {c: {k: 0.0 for k in ("tp", "fp", "fn", "sk_p", "sk_p_ok", "sk_g", "sk_g_ok", "reg", "reg_ok")}
                       for c in ("tout", "rural", "urbain")} for t in a.thresholds}
            probs_viz = []
            for i in tqdm(range(0, len(files), a.batch), desc=key, leave=False):
                fb = files[i:i + a.batch]
                imgs = np.stack([cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in fb]).astype(np.float32)
                probs = predict(model, imgs, sc, dev)
                for f, pr in zip(fb, probs):
                    mask = cv2.imread(str(D / "labels" / a.split / f"{f.stem}.png"), cv2.IMREAD_GRAYSCALE)
                    cls = "rural" if rural.get(f.stem, False) else "urbain"
                    for t in a.thresholds:
                        st = tile_stats(pr, mask, t)
                        for c in ("tout", cls):
                            for k, v in st.items(): acc[t][c][k] += v
                    if len(probs_viz) < 6 and (i // a.batch) % max(1, len(files) // a.batch // 6) == 0:
                        probs_viz.append((f, pr))
            res = {t: {c: summarize(acc[t][c]) for c in acc[t]} for t in a.thresholds}
            best_t = max(a.thresholds, key=lambda t: res[t]["tout"]["parcelles"])
            results[key] = dict(encoder=enc, checkpoint=ckpt, best_threshold=best_t, by_threshold=res)
            viz_cols[key] = (best_t, probs_viz)
            b = res[best_t]
            print(f"\n== {key} [{enc}] seuil {best_t}")
            for c in ("tout", "rural", "urbain"):
                m = b[c]
                print(f"  {c:7} IoU {m['iou']:.3f}  P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f}"
                      f" | squelette P {m['skel_precision']:.3f} R {m['skel_recall']:.3f} F1 {m['skel_f1']:.3f}"
                      f" | parcelles fermées {m['parcelles']:.3f} (n={m['n_parcelles']})")
        del model; torch.cuda.empty_cache()

    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=2))
    if a.viz and viz_cols:
        keys = list(viz_cols); n = min(len(v[1]) for v in viz_cols.values())
        rows = []
        for j in range(n):
            f = viz_cols[keys[0]][1][j][0]
            img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(D / "labels" / a.split / f"{f.stem}.png"), cv2.IMREAD_GRAYSCALE)
            gt = np.full_like(img, 0); gt[mask == 255] = 255; gt[mask == 100] = (40, 60, 110)
            cells = [img, gt]
            for k in keys:
                t, pv = viz_cols[k]; pr = pv[j][1] > t
                c = np.zeros_like(img); c[pr] = 255; cells.append(c)
            rows.append(np.concatenate([np.pad(c, ((4, 4), (4, 4), (0, 0)), constant_values=255) for c in cells], 1))
        grid = np.concatenate(rows)
        head = np.full((40, grid.shape[1], 3), 255, np.uint8)
        for i, lab in enumerate(["image", "verite terrain"] + keys):
            cv2.putText(head, lab[:28], (8 + i * 520, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imwrite(a.viz, cv2.cvtColor(np.concatenate([head, grid]), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
