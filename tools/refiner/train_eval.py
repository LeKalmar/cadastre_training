#!/usr/bin/env python3
"""Entraîne le correcteur (arêtes + faces) et l'évalue contre la vérité terrain.

Protocole honnête : apprentissage sur les feuilles « train » du U-Net, évaluation
sur les feuilles « val » (jamais vues par le U-Net ni par le correcteur).
Les prédictions du U-Net sur ses propres feuilles d'entraînement sont un peu trop
propres ; c'est le prix d'un test non contaminé.

    .venv/bin/python tools/refiner/train_eval.py work_dirs/refiner
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/lekalmar/PyCharmMiscProject/Code/Geenfoncier/GeneFoncier/terrapartus")
from georeferencing import evaluation as ev  # noqa: E402
from georeferencing import polygon_refiner as pr  # noqa: E402
from prepare import sheets, slug  # noqa: E402
from vertex_audit import network_stats  # noqa: E402

SIMPLIFY_PX = (0, 4, 6, 8, 10, 12)


def load(root):
    split = {slug(m["group"]): m["split"] for m in sheets().values()}
    recs = []
    for p in sorted(Path(root).glob("*/features.pkl")):
        r = pickle.load(open(p, "rb"))
        r["split"] = split.get(p.parent.name, "?")
        recs.append(r)
    return recs


def matrix(recs, X, y, cols):
    xs, ys = [], []
    for r in recs:
        lab = r[y]
        keep = lab >= 0
        if not keep.any():
            continue
        xs.append(np.array([[row[c] for c in cols] for row in r[X]], dtype=np.float32)[keep])
        ys.append(lab[keep])
    return np.vstack(xs), np.concatenate(ys)


def fit(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier

    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                         l2_regularization=1.0, class_weight="balanced", random_state=0)
    return clf.fit(X, y)


def proba(clf, rows, cols):
    if not rows:
        return np.zeros(0)
    X = np.array([[row[c] for c in cols] for row in rows], dtype=np.float32)
    return clf.predict_proba(X)[:, 1]


def in_zone(polys, truth):
    shapely.prepare(truth.zone)
    return [p for p in polys if truth.zone.contains(p.representative_point())]


PX_TOLS = (1.5, 3.0)   # tolérances en pixels du scan : le trait mesure ~4 px à toute échelle


def score(polys, truth, res=None):
    """``res`` (m/px du scan) ajoute les mesures en pixels, comparables d'une échelle
    à l'autre : 0,3 m vaut 5 px sur un plan de ville et moins d'un pixel en campagne."""
    pred = in_zone(polys, truth)
    m = ev.match_and_score(pred, truth.faces)
    out = {k: m[k] for k in ("n_pred", "matched", "missed", "spurious", "over_seg", "under_seg", "mean_iou")}
    out["n_ref"] = m["n_ref"]
    for tol in (0.3, 0.6):
        out[f"bF1@{tol}"] = round(pr.boundary_f1(pred, truth, tol)["b_f1"], 4)
    if res:
        for k in PX_TOLS:
            out[f"bF1@{k:g}px"] = round(pr.boundary_f1(pred, truth, k * res, step=max(0.5, res))["b_f1"], 4)
        out["res"] = round(float(res), 4)
    ns = network_stats(pred) if pred else {"nodes": 0, "deg2_turn_lt3": 0}
    out["vertices"] = ns["nodes"]
    out["fake_v"] = ns["deg2_turn_lt3"]
    out["gt_vertices"] = network_stats(truth.faces)["nodes"] if truth.faces else 0
    return out


def edge_report(clf, recs):
    from sklearn.metrics import average_precision_score, roc_auc_score

    X, y = matrix(recs, "Xe", "ye", pr.EDGE_FEATURES)
    p = clf.predict_proba(X)[:, 1]
    false = (y == 0)
    return {"n": int(len(y)), "false_edges": int(false.sum()),
            "AUC": round(float(roc_auc_score(y, p)), 4) if 0 < false.sum() < len(y) else None,
            "AP_false": round(float(average_precision_score(false, 1 - p)), 4) if false.any() else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--edge-thr", type=float, nargs="+", default=[0.5])
    ap.add_argument("--face-thr", type=float, nargs="+", default=[0.5])
    a = ap.parse_args()
    recs = load(a.root)
    train = [r for r in recs if r["split"] == "train"]
    val = [r for r in recs if r["split"] == "val"]
    print(f"{len(train)} feuilles train, {len(val)} feuilles val")

    Xe, ye = matrix(train, "Xe", "ye", pr.EDGE_FEATURES)
    Xf, yf = matrix(train, "Xf", "yf", pr.FACE_FEATURES)
    print(f"arêtes train : {len(ye)} dont {int((ye == 0).sum())} fausses | "
          f"faces train : {len(yf)} dont {int((yf == 0).sum())} non-parcelles")
    clf_e, clf_f = fit(Xe, ye), fit(Xf, yf)
    print("arêtes val :", edge_report(clf_e, val))

    results = []
    for r in val:
        truth = r["truth"]
        cov = pr.Coverage(r["faces"], r["pairs"], r["shared"], r["neighbours"], r["shared_len"])
        pe, pf = proba(clf_e, r["Xe"], pr.EDGE_FEATURES), proba(clf_f, r["Xf"], pr.FACE_FEATURES)
        base = score(r["faces"], truth)
        results.append(dict(sheet=r["sheet"], variant="grass", **base))
        for k in SIMPLIFY_PX[1:]:
            s = pr.simplify_coverage(r["faces"], k * r["res"])
            results.append(dict(sheet=r["sheet"], variant=f"grass+simp{k}", **score(s, truth)))
        for te in a.edge_thr:
            for tf in a.face_thr:
                ref = pr.refine(cov, pe, pf, te, tf)
                results.append(dict(sheet=r["sheet"], variant=f"refine e{te} f{tf}", **score(ref, truth)))
                for k in SIMPLIFY_PX[1:]:
                    s = pr.simplify_coverage(ref, k * r["res"])
                    results.append(dict(sheet=r["sheet"], variant=f"refine e{te} f{tf}+simp{k}",
                                        **score(s, truth)))
        print(f"  {r['sheet'][:40]:40} fait", flush=True)

    out = Path(a.root) / "results.json"
    out.write_text(json.dumps(results, indent=1, default=float))
    with open(Path(a.root) / "refiner_models.pkl", "wb") as fh:
        pickle.dump({"edge": clf_e, "face": clf_f, "edge_features": pr.EDGE_FEATURES,
                     "face_features": pr.FACE_FEATURES}, fh)

    # Tableau agrégé (somme sur les feuilles val)
    variants = list(dict.fromkeys(x["variant"] for x in results))
    keys = ["n_pred", "matched", "missed", "spurious", "over_seg", "under_seg", "vertices"]
    base_v = sum(x["vertices"] for x in results if x["variant"] == "grass")
    print(f"\n{'variante':32} " + " ".join(f"{k:>9}" for k in keys) + "   bF1@.3  bF1@.6  sommets  faux")
    for v in variants:
        rows = [x for x in results if x["variant"] == v]
        tot = {k: sum(x[k] for x in rows) for k in keys}
        f3 = np.mean([x["bF1@0.3"] for x in rows])
        f6 = np.mean([x["bF1@0.6"] for x in rows])
        fake = np.average([x["fake_v"] for x in rows], weights=[x["vertices"] or 1 for x in rows])
        print(f"{v:32} " + " ".join(f"{tot[k]:9d}" for k in keys)
              + f"   {f3:.4f}  {f6:.4f}  {100 * tot['vertices'] / base_v:5.1f}%  {100 * fake:4.1f}%")
    print(f"n_ref total : {sum(x['n_ref'] for x in results if x['variant'] == 'grass')}")


if __name__ == "__main__":
    main()
