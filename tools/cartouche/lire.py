"""Lit le cartouche (titre) d'une feuille cadastrale avec un modèle de vision local.

Deux passes avec Qwen3-VL-8B-Instruct (4 bits, ~6 Go de VRAM ; ne pas lancer pendant un
entraînement) :
  1. la feuille entière réduite : où est le cartouche (boîte en millièmes de l'image) et
     première lecture ;
  2. le cartouche recadré à pleine résolution : lecture précise, en JSON.
Recoupement avec ce que la base sait déjà (commune, année) : un champ en désaccord est
signalé « a_verifier ». Sortie : un JSON par feuille dans --out (rien n'est écrit en base).

    .venv/bin/python tools/cartouche/lire.py --feuille "Arras C 0 1838" --out data/cartouche
    .venv/bin/python tools/cartouche/lire.py --toutes --source export --out data/cartouche
"""
import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "annotation"))
import common as C  # noqa: E402

MODELE = "Qwen/Qwen3-VL-8B-Instruct"
CHAMPS = ("commune", "section_lettre", "section_nom", "feuille_numero", "feuille_nom", "echelle",
          "annee", "geometres", "autres_mentions")

INVITE_1 = (
    "Ceci est une feuille du cadastre français ancien (XIXe siècle). Repère le cartouche ou le titre "
    "de la feuille : l'inscription qui donne la commune, la section, le numéro de feuille, l'échelle, "
    "l'année et les noms des géomètres. Réponds UNIQUEMENT par un objet JSON : "
    '{"bbox_2d": [x0, y0, x1, y1], "commune": ..., "section_lettre": ..., "section_nom": ..., '
    '"feuille_numero": ..., "feuille_nom": ..., "echelle": ..., "annee": ..., "geometres": [...], '
    '"autres_mentions": ...}. bbox_2d entoure tout le cartouche, en coordonnées 0-1000 de l\'image. '
    "Mets null pour ce qui n'est pas écrit. Recopie l'orthographe exacte du document.")
REGLES = (
    " Règles : ne devine rien ; ce qui n'est pas écrit lisiblement vaut null (une feuille déchirée "
    "ou un titre effacé donne des null, pas une invention). Un nombre seul entre tirets en tête "
    "(« -62- ») est une cote d'archive, pas le numéro de feuille. L'écriture est une calligraphie "
    "du XIXe siècle : « dite de … » introduit le nom de la section ; « Terminée / Levée en 18.. par "
    "N…, géomètre » donne l'année et un géomètre. Recopie l'orthographe exacte.")
INVITE_2 = (
    "Voici le cartouche d'une feuille du cadastre français ancien. Transcris-le fidèlement, puis "
    "réponds UNIQUEMENT par un objet JSON : "
    '{"transcription": "...", "commune": ..., "section_lettre": ..., "section_nom": ..., '
    '"feuille_numero": ..., "feuille_nom": ..., "echelle": "1/2500"..., "annee": ..., '
    '"geometres": [{"nom": ..., "role": ...}], "autres_mentions": ..., '
    '"confiance": {"commune": 0-1, "section": 0-1, "feuille": 0-1, "annee": 0-1, "geometres": 0-1}}. '
    "section_lettre = la lettre seule (A, B…) ; feuille_numero = le numéro de la feuille dans la "
    "section, en chiffres arabes (« troisième feuille » = 3, « en une feuille » = 1) ; "
    "annee = l'année de levé ou de terminaison, en chiffres, seulement si elle est écrite." + REGLES)
INVITE_1 = INVITE_1 + REGLES


def charger(dev="cuda"):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(MODELE, quantization_config=q, device_map=dev,
                                                        dtype=torch.bfloat16)
    return model.eval(), AutoProcessor.from_pretrained(MODELE)


def demander(model, proc, img, invite, max_tokens=700):
    import torch
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": invite}]}]
    x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                 return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**x, max_new_tokens=max_tokens, do_sample=False)
    return proc.batch_decode(out[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def json_de(texte):
    m = re.search(r"\{.*\}", texte, re.S)
    if not m:
        return None
    s = m.group(0)
    for essai in (s, re.sub(r",\s*([}\]])", r"\1", s)):
        try:
            return json.loads(essai)
        except json.JSONDecodeError:
            pass
    return None


def apercu(src, cote):
    k = max(src.width, src.height) / cote
    h, w = round(src.height / k), round(src.width / k)
    a = src.read([1, 2, 3], out_shape=(3, h, w), resampling=Resampling.average)
    return Image.fromarray(np.transpose(a, (1, 2, 0))), k


def recadrer(src, bbox, k_apercu, w_ap, h_ap, cote, marge=0.08):
    x0, y0, x1, y1 = [float(v) for v in bbox]
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    dx, dy = (x1 - x0) * marge, (y1 - y0) * marge
    X0 = max(0, int((x0 - dx) / 1000 * w_ap * k_apercu))
    Y0 = max(0, int((y0 - dy) / 1000 * h_ap * k_apercu))
    X1 = min(src.width, int((x1 + dx) / 1000 * w_ap * k_apercu))
    Y1 = min(src.height, int((y1 + dy) / 1000 * h_ap * k_apercu))
    if X1 - X0 < 50 or Y1 - Y0 < 50:
        return None, None
    k = max(1.0, max(X1 - X0, Y1 - Y0) / cote)
    a = src.read([1, 2, 3], window=Window(X0, Y0, X1 - X0, Y1 - Y0),
                 out_shape=(3, round((Y1 - Y0) / k), round((X1 - X0) / k)), resampling=Resampling.average)
    return Image.fromarray(np.transpose(a, (1, 2, 0))), [X0, Y0, X1, Y1]


def norm(s):
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def recouper(lu, connu):
    """Champs en désaccord avec la base : liste de messages."""
    pb = []
    if connu.get("commune") and lu.get("commune"):
        a, b = norm(connu["commune"]), norm(lu["commune"])
        if a and b and a not in b and b not in a:
            pb.append(f"commune lue « {lu['commune']} » ≠ base « {connu['commune']} »")
    if connu.get("annee") and lu.get("annee"):
        m = re.search(r"1[78]\d\d", str(lu["annee"]))
        if m and abs(int(m.group(0)) - int(connu["annee"])) > 3:
            pb.append(f"année lue {m.group(0)} ≠ base {connu['annee']}")
    return pb


CONF_MIN = 0.5
CONF_CHAMPS = {"commune": "commune", "section_lettre": "section", "section_nom": "section",
               "feuille_numero": "feuille", "feuille_nom": "feuille", "annee": "annee", "geometres": "geometres"}


def lire(model, proc, raster, cote1, cote2, indice=None):
    ind = (f" Indice : d'après nos fichiers, la commune est probablement « {indice} » ; "
           "lis ce qui est écrit, et signale-le si c'est autre chose.") if indice else ""
    with rasterio.open(raster) as src:
        ap, k = apercu(src, cote1)
        r1 = demander(model, proc, ap, INVITE_1 + ind)
        j1 = json_de(r1) or {}
        res = {"passe1": j1, "brut1": r1 if not j1 else None}
        bbox = j1.get("bbox_2d")
        if isinstance(bbox, list) and len(bbox) == 4:
            crop, px = recadrer(src, bbox, k, ap.width, ap.height, cote2)
            if crop is not None:
                r2 = demander(model, proc, crop, INVITE_2 + ind, 1200)
                j2 = json_de(r2) or {}
                res.update(passe2=j2, brut2=r2 if not j2 else None, bbox_px=px)
    src_final = res.get("passe2") or res["passe1"]      # la passe 1 n'est qu'un repli
    final = {c: src_final.get(c) for c in CHAMPS}
    final["transcription"] = src_final.get("transcription")
    conf = src_final.get("confiance") or {}
    final["confiance"] = conf
    for c, kc in CONF_CHAMPS.items():                    # champ peu sûr -> null (gardé dans passe2)
        try:
            if kc in conf and float(conf[kc]) < CONF_MIN:
                final[c] = None
        except (TypeError, ValueError):
            pass
    res["lecture"] = final
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feuille", nargs="*", default=[], help="noms ou ids")
    ap.add_argument("--toutes", action="store_true")
    ap.add_argument("--source", help="filtre feuilles.source (export, scan, ia)")
    ap.add_argument("--limite", type=int, default=0)
    ap.add_argument("--out", default=str(C.ROOT / "data" / "cartouche"))
    ap.add_argument("--cote1", type=int, default=1600, help="grand côté de l'aperçu (passe 1)")
    ap.add_argument("--cote2", type=int, default=2400, help="grand côté du recadrage (passe 2)")
    ap.add_argument("--refaire", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cur = C.connect().cursor()
    q = "SELECT id, nom, raster, commune, annee FROM feuilles WHERE raster IS NOT NULL"
    args = []
    if a.feuille:
        q += " AND (id::text = ANY(%s) OR nom = ANY(%s))"
        args += [a.feuille, a.feuille]
    elif not a.toutes:
        sys.exit("Donner --feuille ou --toutes.")
    if a.source:
        q += " AND source = %s"
        args.append(a.source)
    feuilles = cur.execute(q + " ORDER BY nom", args).fetchall()
    if a.limite:
        feuilles = feuilles[: a.limite]
    model, proc = charger()
    for fid, nom, raster, commune, annee in feuilles:
        dest = out / f"{fid}.json"
        if dest.exists() and not a.refaire:
            continue
        if not Path(raster).exists():
            print(f"  scan introuvable : {nom}")
            continue
        t0 = time.time()
        res = lire(model, proc, raster, a.cote1, a.cote2, indice=commune)
        res.update(feuille_id=str(fid), feuille=nom, raster=raster,
                   connu={"commune": commune, "annee": annee})
        res["a_verifier"] = recouper(res["lecture"], res["connu"])
        dest.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        L = res["lecture"]
        print(f"  {nom[:40]:40} {time.time() - t0:5.0f} s | {L.get('commune')} | sect. {L.get('section_lettre')} "
              f"{L.get('section_nom') or ''} | f. {L.get('feuille_numero')} {L.get('feuille_nom') or ''} | "
              f"{L.get('echelle')} | {L.get('annee')}" + (f"  ⚠ {'; '.join(res['a_verifier'])}" if res["a_verifier"] else ""),
              flush=True)


if __name__ == "__main__":
    main()
