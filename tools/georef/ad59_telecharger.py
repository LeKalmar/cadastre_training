"""Télécharge les plans du fonds P 31 (« Cadastre napoléonien ») des Archives du Nord
qui manquent à la base locale : communes absentes et campagnes de 1870 et après.

Le portail (Mnesys) n'a ni Cloudflare ni manifeste IIIF : liste de résultats en GET,
puis l'API de la visionneuse, qui ne rend qu'une fenêtre de 15 vues ; on la fait
glisser jusqu'à avoir toutes les vues de la notice. Requêtes espacées (--pause).

    .venv-georef/bin/python tools/georef/ad59_telecharger.py --liste                 # ce qui manque
    .venv-georef/bin/python tools/georef/ad59_telecharger.py --commune ANICHE TRELON  # essai
    .venv-georef/bin/python tools/georef/ad59_telecharger.py --manquants             # tout

Rangement (comme la base) : <sortie>/<insee>_<slug>/<année>/RAW/RAW_<insee>_<slug>_<année>_UNKNOWN_0_<n>.jpg
Une notice qui mêle deux campagnes (« TRELON - 1822, - 1882 ») est rangée sous la plus
récente ; ses feuilles napoléoniennes y seront classées « sans quadrillage ».
"""
import argparse
import collections
import html
import json
import os
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = "https://archivesdepartementales.lenord.fr"
FORM = "b57520fe-2768-4335-b138-119f274f0a73"
NORD = ("/home/lekalmar/BACKUP WINDOWS FILES/Travail/Cartographie/INDEXATION CADASTRE/"
        "Cadastre XIX/Cartes/Communes/59 - Nord")
UA = "Mozilla/5.0 (X11; Linux x86_64) cadastre-recherche"
PAUSE = 1.5


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def get(url, binary=False, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            time.sleep(PAUSE)
            return data if binary else data.decode("utf-8", "replace")
        except Exception as e:                      # réseau : on réessaie en ralentissant
            if k == tries - 1:
                raise
            log(f"  réessai ({e})")
            time.sleep(10 * (k + 1))


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\(.*?\)|sans[- ]date|\d+", "", s)
    w = [x for x in re.split(r"[^a-z]+", s) if x]
    if w and w[-1] in ("le", "la", "les", "l"):
        w = [w[-1]] + w[:-1]
    return "".join(w).replace("sainte", "ste").replace("saint", "st").replace("lez", "les")


def slug(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", re.sub(r"\(.*?\)", "", s)).strip("-")


# ----------------------------------------------------------------------------- catalogue
def catalogue(out):
    path = os.path.join(out, "_catalogue.json")
    if os.path.exists(path):
        return json.load(open(path))
    recs, page, pages = {}, 1, 1
    while page <= pages:
        s = get(f"{BASE}/search/results?formUuid={FORM}&sort=referencecode_asc&mode=&page={page}")
        m = re.search(r"Page suivante : \d+/(\d+)", s)
        pages = int(m.group(1)) if m else pages
        for li in s.split('<li class="element-list">')[1:]:
            a = re.search(r'href="/ark:/33518/([a-z0-9]+)/([0-9a-f-]{36})"', li)
            t = re.search(r"<h2>\s*<span>(.*?)</span>", li, re.S)
            c = re.search(r'class="referenceCodes">(.*?)<', li)
            n = re.search(r"(\d+)\s+medias?", li)
            if a and t:
                recs[a.group(1)] = {"ark": a.group(1), "media0": a.group(2),
                                    "titre": html.unescape(t.group(1)).strip(),
                                    "cote": html.unescape(c.group(1)).strip() if c else "",
                                    "medias": int(n.group(1)) if n else 0}
        log(f"catalogue page {page}/{pages} : {len(recs)} notices")
        page += 1
    out_l = sorted(recs.values(), key=lambda r: r["cote"])
    for r in out_l:
        r["commune"] = re.split(r" - |,| \d{4}", r["titre"])[0].strip()
        r["annees"] = sorted({int(y) for y in re.findall(r"(?<!\d)(1[89]\d\d)(?!\d)", r["titre"])})
    os.makedirs(out, exist_ok=True)
    json.dump(out_l, open(path, "w"), ensure_ascii=False, indent=0)
    return out_l


def base_locale():
    """norm(commune) -> années présentes dans la base locale."""
    have = collections.defaultdict(set)
    for d in os.listdir(NORD):
        p = os.path.join(NORD, d)
        if not os.path.isdir(p):
            continue
        name = re.sub(r"^\d+\s*[_-]?\s*(-\s*)?", "", d)
        ys = have[norm(name)]
        for root, _, files in os.walk(p):
            if any(f.lower().endswith((".jpg", ".jpeg")) for f in files):
                ys.update(int(y) for y in re.findall(r"(?<!\d)(1[89]\d\d)(?!\d)", root))
    return have


def manquants(cat, an_min):
    have = base_locale()
    todo = []
    for r in cat:
        k = norm(r["commune"])
        if k not in have:
            todo.append((r, "commune absente"))
        elif any(y >= an_min and not any(abs(h - y) <= 1 for h in have[k]) for y in r["annees"]):
            todo.append((r, "campagne absente"))
    return todo


# ----------------------------------------------------------------------------- insee
_INSEE = {}


def insee(commune, cache):
    if not _INSEE and os.path.exists(cache):
        _INSEE.update(json.load(open(cache)))
    if commune not in _INSEE:
        q = urllib.parse.urlencode({"nom": re.sub(r"\(.*?\)", "", commune).strip(),
                                    "codeDepartement": "59", "fields": "code,nom", "limit": 5})
        try:
            res = json.loads(urllib.request.urlopen(f"https://geo.api.gouv.fr/communes?{q}", timeout=30).read())
        except Exception:
            res = []
        hit = next((c for c in res if norm(c["nom"]) == norm(commune)), None)
        _INSEE[commune] = hit["code"] if hit else "00000"     # commune disparue ou fusionnée
        json.dump(_INSEE, open(cache, "w"), ensure_ascii=False, indent=0)
    return _INSEE[commune]


# ----------------------------------------------------------------------------- vues
def vues(rec):
    """Toutes les vues d'une notice, en faisant glisser la fenêtre de la visionneuse."""
    seen, asked, todo = {}, set(), [rec["media0"]]
    total = rec["medias"]
    while todo and len(seen) < max(total, 1):
        u = todo.pop(0)
        if u in asked:
            continue
        asked.add(u)
        d = json.loads(get(f"{BASE}/visualizer/api?arkName={rec['ark']}&uuid={u}"))
        total = (d.get("counts") or {}).get("media", total)
        media = d.get("media") or []
        if isinstance(media, dict):                 # {position: vue} quand la fenêtre a glissé
            items = sorted((int(k), m) for k, m in media.items())
        else:                                       # liste : la vue demandée est à positions.media
            idx = next((i for i, m in enumerate(media) if m.get("uuid") == u), 0)
            off = (d.get("positions") or {}).get("media", idx) - idx
            items = [(off + i, m) for i, m in enumerate(media)]
        items = [(p, m) for p, m in items if isinstance(m, dict) and m.get("uuid")]
        new = [m for _, m in items if m["uuid"] not in seen]
        for p, m in items:
            seen.setdefault(m["uuid"], (p, m["location"]["original"]))
        if new:                                     # repartir des deux bouts de la fenêtre
            todo += [new[-1]["uuid"], new[0]["uuid"]]
    return [(u, url) for u, (_, url) in sorted(seen.items(), key=lambda kv: kv[1][0])], total


def telecharger(rec, out, cache_insee, partage=False):
    """partage : plusieurs notices tombent dans le même dossier (ex. Lille 1881, une
    notice par section) ; leurs vues sont alors distinguées par l'ark de la notice."""
    com = rec["commune"]
    code = insee(com, cache_insee)
    an = str(max(rec["annees"])) if rec["annees"] else "sans-date"
    d = os.path.join(out, f"{code}_{slug(com)}", an, "RAW")
    os.makedirs(d, exist_ok=True)
    vs, total = vues(rec)
    if len(vs) < total:
        log(f"  attention : {len(vs)} vues trouvées sur {total} annoncées")
    ok = 0
    for i, (u, url) in enumerate(vs):
        lot = rec["ark"] if partage else "0"
        f = os.path.join(d, f"RAW_{code}_{slug(com)}_{an}_UNKNOWN_{lot}_{i}.jpg")
        if os.path.exists(f):
            ok += 1
            continue
        data = get(url, binary=True)
        tmp = f + ".part"
        open(tmp, "wb").write(data)
        try:
            Image.open(tmp).verify()
        except Exception as e:
            os.remove(tmp)
            log(f"  image invalide {url} ({e})")
            continue
        os.replace(tmp, f)
        ok += 1
    json.dump({"notice": rec, "vues": vs}, open(os.path.join(d, f"_notice_{rec['ark']}.json"), "w"),
              ensure_ascii=False, indent=0)
    return ok, total, d


def main():
    global PAUSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "ad59_complement"))
    ap.add_argument("--commune", nargs="*", help="noms tels que le portail les écrit (ex. ANICHE)")
    ap.add_argument("--manquants", action="store_true", help="tout ce qui manque à la base locale")
    ap.add_argument("--annee-min", type=int, default=1870)
    ap.add_argument("--liste", action="store_true", help="liste seulement")
    ap.add_argument("--pause", type=float, default=1.5, help="secondes entre deux requêtes")
    a = ap.parse_args()
    PAUSE = a.pause

    cat = catalogue(a.out)
    if a.commune:
        want = {norm(c) for c in a.commune}
        todo = [(r, "demandée") for r in cat if norm(r["commune"]) in want]
    else:
        todo = manquants(cat, a.annee_min)
    log(f"{len(todo)} notices, {sum(r['medias'] for r, _ in todo)} vues annoncées")
    if a.liste or not (a.commune or a.manquants):
        for r, why in todo:
            print(f"  {r['cote']:<28} {r['titre']:<45} {r['medias']:>4} vues  ({why})")
        return
    cache_insee = os.path.join(a.out, "_insee.json")
    fait = os.path.join(a.out, "_faites.txt")
    done = set(open(fait).read().split()) if os.path.exists(fait) else set()
    dossier = collections.Counter((norm(r["commune"]), max(r["annees"], default=0)) for r, _ in todo)
    for k, (r, why) in enumerate(todo, 1):
        if r["ark"] in done:
            continue
        try:
            partage = dossier[(norm(r["commune"]), max(r["annees"], default=0))] > 1
            ok, total, d = telecharger(r, a.out, cache_insee, partage)
            log(f"[{k}/{len(todo)}] {r['titre']} ({r['cote']}) : {ok}/{total} vues -> {os.path.relpath(d, a.out)}")
            if ok >= total:
                open(fait, "a").write(r["ark"] + "\n")
        except Exception as e:
            log(f"[{k}/{len(todo)}] ECHEC {r['titre']} : {e}")
    log("TERMINE")


if __name__ == "__main__":
    sys.exit(main())
