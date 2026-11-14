"""Relit les cotes des feuilles restées « cotes illisibles » dans le cache de run.py,
avec la lecture actuelle de labels.py (ex. bande prolongée dans la marge, Trélon).

Le quadrillage déjà détecté est réutilisé ; seule la lecture OCR est refaite et le
cache mis à jour. Affiche les dossiers de commune concernés, à passer ensuite à
run.py --only pour refaire leur calage.

    .venv-georef/bin/python tools/georef/relire_cotes.py --out data/georef_nord [--jobs 6] \
        [--liste communes.txt]
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labels as LB  # noqa: E402
from run import good_fit  # noqa: E402


def relire(path):
    r = json.load(open(path))
    try:
        rec = LB.read(r)
    except Exception as e:                     # image illisible : on garde l'ancienne lecture
        return path, False, str(e)
    r["ocr_avant"] = r["ocr"]
    r["ocr"] = rec
    tmp = path + ".tmp"
    json.dump(r, open(tmp, "w"))
    os.replace(tmp, path)
    return path, good_fit(LB.fit(rec)), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--liste", help="écrit ici les dossiers de commune à recaler")
    a = ap.parse_args()
    todo = []
    for p in glob.glob(os.path.join(a.out, "_cache", "*", "*.json")):
        if p.endswith(".cand.json"):
            continue
        r = json.load(open(p))
        if "ocr" in r and "ocr_avant" not in r and not good_fit(LB.fit(r["ocr"])):
            todo.append(p)
    print(f"{len(todo)} feuilles à relire", flush=True)
    gagnees, communes = 0, set()
    with ProcessPoolExecutor(a.jobs) as ex:
        for k, (p, ok, err) in enumerate(ex.map(relire, todo), 1):
            com = os.path.basename(os.path.dirname(p))
            communes.add(com)
            gagnees += ok
            print(f"[{k}/{len(todo)}] {'LISIBLE ' if ok else 'toujours'} {com} "
                  f"{os.path.basename(p)[:-5]} {err}", flush=True)
    print(f"TERMINE : {gagnees}/{len(todo)} feuilles désormais lisibles, "
          f"{len(communes)} communes à recaler", flush=True)
    if a.liste:
        open(a.liste, "w").write("\n".join(sorted(communes)) + "\n")


if __name__ == "__main__":
    main()
