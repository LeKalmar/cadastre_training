"""Regroupe les typologies d'une feuille en parcelles principales (main / typology GénéFoncier).

Pour chaque paire de typologies voisines, le classifieur d'arêtes (train_aretes.py) lit les
vignettes redressées le long de leur contour commun (mêmes vignettes qu'à l'entraînement) ;
probabilité de la paire = moyenne de ses vignettes. Les paires >= --seuil sont réunies
(union-find, les plus sûres d'abord). À venir : contrainte « un seul numéro par parcelle
principale » une fois les numéros lus par OCR.

Sortie GPKG : « typologies » (id, type, groupe) et « principales » (union de chaque groupe).
Avec --verite (typologies validées ayant un par_main_id GénéFoncier) : part des parcelles
principales exactement reconstituées, hors parcelles de bord de feuille.

    .venv/bin/python tools/typologie/regrouper.py --feuille "Arras C 0 1838" --verite \
        --out data/regroupement_pred/arras_c.gpkg
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import timm
import torch
from shapely import STRtree
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "annotation"))
import common as C  # noqa: E402
from aretes import partage, vignettes  # noqa: E402
from train_aretes import MEAN, STD  # noqa: E402


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i, j):
        self.p[self.find(i)] = self.find(j)


@torch.no_grad()
def proba_paires(model, src, polys, tol, min_px, pas_px, max_v, dev, batch=256):
    px = abs(src.transform.a)
    tree = STRtree(polys)
    paires, lots = [], []
    for i, g in enumerate(polys):
        for j in tree.query(g.buffer(tol)):
            j = int(j)
            if j <= i:
                continue
            vs = []
            for ligne in partage(g, polys[j], tol):
                if ligne.length / px >= min_px:
                    vs += vignettes(src, g, polys[j], ligne, px, pas_px)[:max_v]
            if vs:
                paires.append((i, j))
                lots.append(vs)
    flat = [(k, np.concatenate([im, ax], -1)) for k, vs in enumerate(lots) for im, ax in vs]
    probs = defaultdict(list)
    model.eval()
    for s in range(0, len(flat), batch):
        part = flat[s:s + batch]
        x = np.stack([((v.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1) for _, v in part])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(model(torch.from_numpy(x).to(dev)).float().squeeze(1)).cpu().numpy()
        for (k, _), pp in zip(part, p):
            probs[k].append(float(pp))
    return [(paires[k], float(np.mean(v))) for k, v in probs.items()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feuille", required=True)
    ap.add_argument("--modele", default="work_dirs/regroupement_v1/best_model.pth")
    ap.add_argument("--entree", help="GPKG de typologies prédites (predire.py) ; défaut : typologies validées en base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seuil", type=float, default=0.5)
    ap.add_argument("--verite", action="store_true")
    ap.add_argument("--bord-m", type=float, default=3.0)
    ap.add_argument("--tol-m", type=float, default=0.3)
    ap.add_argument("--min-px", type=float, default=24)
    ap.add_argument("--pas-px", type=float, default=160)
    ap.add_argument("--max-par-arete", type=int, default=3)
    ap.add_argument("--numeros", choices=["aucun", "simules"], default="aucun",
                    help="contrainte « un numéro par parcelle principale » ; « simules » (évaluation) : le "
                         "numéro est posé dans la plus grande typologie de chaque parcelle GénéFoncier "
                         "(en attendant l'OCR des numéros)")
    ap.add_argument("--seuil-bas", type=float, default=0.05,
                    help="un groupe sans numéro rejoint son voisin numéroté le plus probable si p >= seuil-bas")
    a = ap.parse_args()

    import geopandas as gpd

    cur = C.connect().cursor()
    fid, nom, raster, fg = cur.execute(
        "SELECT id, nom, raster, ST_AsBinary(geom) FROM feuilles WHERE id::text = %s OR nom = %s "
        "ORDER BY (SELECT count(*) FROM parcelles p WHERE p.feuille_id = feuilles.id) DESC LIMIT 1",
        (a.feuille, a.feuille)).fetchone()
    if a.entree:
        t = gpd.read_file(a.entree, layer="typologies")
        ids, polys, types, mains = list(map(str, t.index)), list(t.geometry), list(t["type"]), [None] * len(t)
    else:
        rows = cur.execute("SELECT id, ST_AsBinary(geom), type, par_main_id FROM parcelles WHERE feuille_id = %s "
                           "AND statut = 'valide' AND developpement IS NULL", (fid,)).fetchall()
        ids = [str(r[0]) for r in rows]
        polys = [C.geom_of(r[1]) for r in rows]
        types = [r[2] for r in rows]
        mains = [str(r[3]) if r[3] else None for r in rows]
    dev = "cuda"
    e = torch.load(a.modele, map_location=dev)
    model = timm.create_model(e["config"]["arch"], pretrained=False, in_chans=6, num_classes=1).to(dev)
    model.load_state_dict(e["model"])
    with rasterio.open(raster) as src:
        pp = proba_paires(model, src, polys, a.tol_m, a.min_px, a.pas_px, a.max_par_arete, dev)
    uf = UF(len(polys))
    numero = [None] * len(polys)
    if a.numeros == "simules":
        plus_grande = {}
        for i, m in enumerate(mains):
            if m is not None and (m not in plus_grande or polys[i].area > polys[plus_grande[m]].area):
                plus_grande[m] = i
        for m, i in plus_grande.items():
            numero[i] = m
    num_grp = {i: numero[i] for i in range(len(polys))}      # racine -> numéro du groupe

    def reunir(i, j):
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            return
        if num_grp.get(ri) is not None and num_grp.get(rj) is not None:
            return                                          # deux numéros : deux parcelles
        uf.union(ri, rj)
        num_grp[uf.find(rj)] = num_grp.get(ri) or num_grp.get(rj)

    for (i, j), p in sorted(pp, key=lambda t: -t[1]):
        if p >= a.seuil:
            reunir(i, j)
    if a.numeros != "aucun":                               # groupes orphelins -> voisin numéroté
        for (i, j), p in sorted(pp, key=lambda t: -t[1]):
            if p < a.seuil_bas:
                break
            ri, rj = uf.find(i), uf.find(j)
            if ri != rj and (num_grp.get(ri) is None) != (num_grp.get(rj) is None):
                reunir(i, j)
    grp = [uf.find(i) for i in range(len(polys))]
    typ = gpd.GeoDataFrame({"id": ids, "type": types, "groupe": grp, "par_main_verite": mains},
                           geometry=polys, crs=f"EPSG:{C.SRID}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    typ.to_file(a.out, layer="typologies", driver="GPKG")
    princ = typ.dissolve("groupe", aggfunc={"id": "count"}).rename(columns={"id": "n_typologies"})
    princ.to_file(a.out, layer="principales", driver="GPKG")
    paires = gpd.GeoDataFrame(
        {"proba": [p for _, p in pp]},
        geometry=[polys[i].boundary.intersection(polys[j].buffer(a.tol_m)) for (i, j), _ in pp], crs=f"EPSG:{C.SRID}")
    paires.to_file(a.out, layer="aretes", driver="GPKG")
    print(f"{nom} : {len(polys)} typologies, {len(pp)} paires voisines -> {len(princ)} parcelles principales -> {a.out}")

    if a.verite and not a.entree:
        bord = C.geom_of(fg).boundary.buffer(a.bord_m)
        au_bord = [g.intersects(bord) for g in polys]
        vrai, pred = defaultdict(set), defaultdict(set)
        for i, (m, g) in enumerate(zip(mains, grp)):
            if m is not None:
                vrai[m].add(i)
            pred[g].add(i)
        ok = n = 0
        for m, membres in vrai.items():
            if any(au_bord[i] for i in membres):
                continue
            n += 1
            ok += pred[grp[next(iter(membres))]] == membres
        # exactitude par paire voisine (hors bord, typologies rattachées des deux côtés)
        pv = [(mains[i] == mains[j], p >= a.seuil) for (i, j), p in pp
              if mains[i] and mains[j] and not au_bord[i] and not au_bord[j]]
        acc = np.mean([x == y for x, y in pv]) if pv else float("nan")
        print(f"Parcelles principales exactement reconstituées (hors bord) : {ok}/{n} = {ok / max(n, 1):.1%}")
        print(f"Paires voisines bien classées (hors bord) : {acc:.1%} sur {len(pv)}")
    _ = unary_union


if __name__ == "__main__":
    main()
