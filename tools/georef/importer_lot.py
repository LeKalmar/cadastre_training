"""Passe les feuilles géoréférencées par tools/georef/run.py dans l'extracteur
(tools/annotation/import_ia.py) : une feuille par scan, propositions « pending ».

    .venv/bin/python tools/georef/importer_lot.py --out data/georef_nord            # liste seulement
    .venv/bin/python tools/georef/importer_lot.py --out data/georef_nord --lancer   # exécute
        [--statuts ok] [--only 59172_denain …] [--agrandir 0.13]
        [--workers 2] [--attendre-pid <pid de run.py>]

Par défaut, seules les feuilles « ok » sont prises, hors tableaux d'assemblage
(quadrillage de 400 m ou plus ; --avec-assemblages pour les garder). Les feuilles
« à vérifier » jugées « calage bon » dans QGIS (table georef_verif, voir verif.py) sont
prises aussi ; celles jugées « calage faux » jamais. Plusieurs --out possibles. Une feuille déjà en base qui a
des propositions est sautée ; sans propositions (extraction interrompue), elle est
relancée par son identifiant. Avec --attendre-pid, le lot suit le géoréférencement :
il reprend les communes au fur et à mesure qu'elles sont terminées.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "annotation"))
PY = os.path.join(ROOT, ".venv", "bin", "python")
SCRIPT = os.path.join(ROOT, "tools", "annotation", "import_ia.py")
_print = threading.Lock()
# quadrillage de 400 m ou plus : tableau d'assemblage (1/10 000 et moins), pas une feuille de détail
ASSEMBLAGE_PAS_M = 400
AUTRES = set()      # rasters rangés comme autres exemplaires d'une feuille (rempli par state())
DECISIONS = {}      # calages « à vérifier » jugés dans QGIS : geotiff -> 'bon' | 'faux' (verif.py)


def log(msg):
    with _print:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


def candidates(a):
    todo = []
    for cj in sorted(c for out in a.out for c in glob.glob(os.path.join(out, "*", "*", "calage.json"))):
        meta = json.load(open(cj))
        if a.only and meta["commune"] not in a.only:
            continue
        for f in meta["feuilles"]:
            tif = f.get("geotiff")
            if not tif or not os.path.exists(tif):
                continue
            if DECISIONS.get(tif) == "faux":
                continue                    # calage jugé faux dans QGIS
            if not (any(f["statut"].startswith(s) for s in a.statuts) or DECISIONS.get(tif) == "bon"):
                continue
            if (f.get("pas_m") or 0) >= ASSEMBLAGE_PAS_M and not a.avec_assemblages:
                continue                    # tableau d'assemblage : rien à extraire
            m = re.search(r"_(\d+)$", os.path.splitext(f["feuille"])[0])
            num = m.group(1) if m else os.path.splitext(f["feuille"])[0]
            nom = f"{meta['nom']} {meta.get('annee') or 's.d.'} – feuille {num}"
            todo.append((tif, nom, meta["nom"], meta.get("annee")))
    return todo


def state():
    """raster -> (id de feuille, nombre de propositions) pour les feuilles en base."""
    import common as C
    with C.connect() as db:
        rows = db.execute(
            "select f.id, f.raster, (select count(*) from parcelles_pending p where p.feuille_id = f.id) "
            "from feuilles f where f.raster is not null").fetchall()
        autres = db.execute(
            "select f.id, trim(r) from feuilles f, unnest(string_to_array(f.autres_rasters, ';')) r "
            "where f.autres_rasters is not null").fetchall()
    out = {}
    for i, r, n in rows:                    # doublon éventuel : garder celle qui a des propositions
        if r not in out or n > out[r][1]:
            out[r] = (str(i), n)
    # autre exemplaire d'une feuille déjà en base (champ autres_rasters) : jamais réimporté
    AUTRES.clear()
    AUTRES.update(r for _, r in autres if r not in out)
    return out


def emprise(tif):
    """Emprise L93 (xmin, ymin, xmax, ymax) lue dans les balises GeoTIFF (sans GDAL :
    le .venv de l'extracteur n'en a pas). GeoTIFF nord en haut, écrits par run.py."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(tif) as im:
            sx, sy = im.tag_v2[33550][:2]
            _, _, _, x0, y0, _ = im.tag_v2[33922][:6]
            W, H = im.size
    except Exception:
        return None
    return [x0, y0 - sy * H, x0 + sx * W, y0]


def photo(a):
    """Emprise de tous les GeoTIFF déjà produits, avant un recalage."""
    out = {}
    for cj in (c for out in a.out for c in glob.glob(os.path.join(out, "*", "*", "calage.json"))):
        for f in json.load(open(cj))["feuilles"]:
            tif = f.get("geotiff")
            if tif and os.path.exists(tif):
                out[tif] = emprise(tif)
    json.dump(out, open(a.photo_calage, "w"))
    print(f"{len(out)} emprises enregistrées dans {a.photo_calage}")


def deplacees(a, existing, tol=0.5):
    """Feuilles déjà extraites dont le GeoTIFF a bougé de plus de tol m depuis la photo."""
    ref = json.load(open(a.controle_calage))
    out = set()
    for tif, (_, n) in existing.items():
        if n and ref.get(tif) and os.path.exists(tif):
            e = emprise(tif)
            if e and max(abs(u - v) for u, v in zip(e, ref[tif])) > tol:
                out.add(tif)
    return out


def avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return 99.0


def run_one(item, a, existing):
    tif, nom, commune, annee = item
    if tif in existing:
        cmd = [PY, SCRIPT, "--feuille-id", existing[tif][0], "--remplacer"]
    else:
        cmd = [PY, SCRIPT, "--raster", tif, "--nom", nom, "--commune", commune]
        if annee:
            cmd += ["--annee", str(annee)]
    if a.agrandir:
        cmd += ["--agrandir", str(a.agrandir)]
    if a.memoire_max:
        # plafond mémoire : un pic à ~15 Go (scan agrandi x2) ne tue que cette extraction
        cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={a.memoire_max}",
               "-p", "MemorySwapMax=2G"] + cmd
    while avail_gb() < a.memoire_libre:     # attendre que la RAM se libère avant de démarrer
        time.sleep(20)
    t = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = [l for l in (r.stdout + r.stderr).splitlines() if "warn" not in l.lower()]
    res = next((l for l in out if "propositions à valider" in l), None)
    if r.returncode == 0 and res:
        log(f"OK    {nom} ({time.time() - t:.0f} s) : {res.split(':', 1)[1].split('(')[0].strip()}")
    else:
        log(f"ECHEC {nom} (code {r.returncode}) : {' | '.join(out[-3:])[:300]}")
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, nargs="+", help="dossier(s) de sortie de run.py")
    ap.add_argument("--statuts", nargs="*", default=["ok"],
                    help="débuts de statut acceptés (ex. ok « à vérifier (placée par ses cotes »)")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--avec-assemblages", action="store_true",
                    help=f"prend aussi les feuilles à quadrillage >= {ASSEMBLAGE_PAS_M} m (tableaux d'assemblage)")
    ap.add_argument("--agrandir", type=float, default=0.13,
                    help="résolution cible avant l'IA (feuilles au 1/2000 ≈ 0,25 m/px)")
    ap.add_argument("--workers", type=int, default=1,
                    help="extractions simultanées ; 1 sur 32 Go (pic mesuré : 15 Go par extraction)")
    ap.add_argument("--memoire-max", default="18G", help="plafond mémoire par extraction ('' = aucun)")
    ap.add_argument("--memoire-libre", type=float, default=17.0,
                    help="RAM disponible (Go) exigée avant de démarrer une extraction")
    ap.add_argument("--attendre-pid", type=int, help="suit ce géoréférencement jusqu'à sa fin")
    ap.add_argument("--lancer", action="store_true", help="sans cette option : liste seulement")
    ap.add_argument("--photo-calage", help="enregistre l'emprise des GeoTIFF existants puis s'arrête")
    ap.add_argument("--controle-calage",
                    help="photo prise avant un recalage : une feuille déjà extraite dont le GeoTIFF "
                         "a bougé est réextraite (--remplacer), pour que parcelles et scan concordent")
    a = ap.parse_args()
    if a.photo_calage:
        return photo(a)

    if not a.lancer:
        try:
            import verif
            DECISIONS.update(verif.decisions())
        except Exception:
            pass
        todo = candidates(a)
        print(f"{len(todo)} feuilles à extraire")
        for _, nom, _, _ in todo:
            print("  ", nom)
        return

    done, fails, signale = set(), {}, set()
    with ThreadPoolExecutor(a.workers) as ex:
        while True:
            georef_alive = bool(a.attendre_pid) and os.path.exists(f"/proc/{a.attendre_pid}")
            existing = state()
            try:                            # calages « à vérifier » : table georef_verif, jugés dans QGIS
                import verif
                verif.sync(a.out)
                DECISIONS.clear()
                DECISIONS.update(verif.decisions())
            except Exception as e:          # la table manque ou la base est indisponible : on continue sans
                log(f"calages à vérifier non lus : {e}")
            moved = deplacees(a, existing) if a.controle_calage else set()
            todo = [it for it in candidates(a) if it[0] not in done and fails.get(it[0], 0) < 2
                    and it[0] not in AUTRES
                    and (not (it[0] in existing and existing[it[0]][1] > 0) or it[0] in moved)]
            for tif in moved - {it[0] for it in candidates(a)} - signale:
                signale.add(tif)            # déplacée mais plus « ok » : à regarder à la main
                log(f"ATTENTION {tif} : recalée hors « ok », parcelles extraites avec l'ancien calage")
            if moved & {it[0] for it in todo}:
                log(f"{len(moved & {it[0] for it in todo})} feuilles déjà extraites recalées : réextraction")
            if todo:
                log(f"{len(todo)} feuilles à extraire (déjà faites : {len(done)})")
                futs = {ex.submit(run_one, it, a, existing): it for it in todo}
                for f, it in futs.items():
                    if f.result() == 0:
                        done.add(it[0])
                    else:
                        fails[it[0]] = fails.get(it[0], 0) + 1
            elif not georef_alive:
                break
            else:
                time.sleep(60)
    log(f"TERMINE : {len(done)} feuilles extraites, {sum(1 for v in fails.values() if v >= 2)} en échec")


if __name__ == "__main__":
    main()
