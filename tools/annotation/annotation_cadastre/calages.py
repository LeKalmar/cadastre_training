"""Juger dans QGIS les calages « à vérifier » du géoréférencement automatique.

tools/georef/run.py classe « à vérifier » une feuille qu'il a placée sans pouvoir
confirmer le calage (cadastre actuel trop différent, une seule feuille d'appui, cotes
douteuses). tools/georef/verif.py les recopie dans la table georef_verif ; ce module
les montre une à une et enregistre la décision. tools/georef/importer_lot.py extrait
ensuite les feuilles jugées « calage bon » et ne propose plus jamais les « calage faux ».

Touches (carte active, comme le reste de la barre) :
    C   calage suivant à vérifier (ou celui sélectionné dans la couche « Calages à vérifier ») :
        son GeoTIFF s'ouvre en haut de la carte, les scans déjà extraits de la commune s'allument
    B   calage bon   -> décision enregistrée, puis calage suivant
    F   calage faux  -> décision enregistrée, puis calage suivant
Avec une sélection dans la couche « Calages à vérifier », B et F s'appliquent à toute la sélection.
Pour revenir sur une décision : sélectionner l'emprise, puis B ou F (ou éditer le champ
« decision » : a_verifier, bon, faux).

Bon test : le scan se raccorde-t-il aux feuilles voisines de la même commune le long de
leurs limites communes, et ses chemins tombent-ils sur ceux du parcellaire actuel (fond
OpenStreetMap) là où ils n'ont pas changé ?
"""
import os
import re

from qgis.core import (NULL, QgsDataSourceUri, QgsFeatureRequest, QgsFillSymbol, QgsPalLayerSettings,
                       QgsProject, QgsRasterLayer, QgsRuleBasedRenderer, QgsTextFormat,
                       QgsVectorLayer, QgsVectorLayerSimpleLabeling)
from qgis.PyQt.QtGui import QColor
from qgis.utils import iface

SERVICE = "cadastre_annotation"
COUCHE = "Calages à vérifier"
GROUPE = "Calages à vérifier"
APERCU = "Calage en cours"
_etat = {"geotiff": None}


def _repo():
    """Racine du dépôt : dossier parent du projet (annotation_parcelles/..), sinon celui de l'extension."""
    home = QgsProject.instance().homePath()
    if home:
        return os.path.dirname(os.path.abspath(home))
    return os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", ".."))


def _chemin(p):
    if os.path.isabs(p) or os.path.exists(p):
        return p
    return os.path.join(_repo(), p)


def _couche(creer=True):
    P = QgsProject.instance()
    lyrs = P.mapLayersByName(COUCHE)
    if lyrs:
        return lyrs[0]
    if not creer:
        return None
    uri = QgsDataSourceUri()
    uri.setParam("service", SERVICE)
    uri.setDataSource("public", "georef_verif", "geom", "present", "id")
    uri.setSrid("2154")
    lyr = QgsVectorLayer(uri.uri(False), COUCHE, "postgres")
    if not lyr.isValid():
        iface.messageBar().pushWarning("Calages", "Table georef_verif inaccessible : la créer avec "
                                       "annotation_parcelles/schema.sql, puis lancer tools/georef/verif.py.")
        return None
    _styler(lyr)
    P.addMapLayer(lyr, False)
    root = P.layerTreeRoot()
    grp = root.findGroup(GROUPE) or root.insertGroup(0, GROUPE)
    grp.insertLayer(0, lyr)
    return lyr


def _styler(lyr):
    def sym(color, width, style="solid", fill=None):
        return QgsFillSymbol.createSimple({
            "style": "solid" if fill else "no", "color": fill or "0,0,0,0", "outline_color": color,
            "outline_width": str(width), "outline_width_unit": "MM", "outline_style": style})
    root = QgsRuleBasedRenderer.Rule(None)
    for label, expr, s in (
            ("À vérifier", "\"decision\" = 'a_verifier'", sym("#e8590c", 0.8, fill="232,89,12,25")),
            ("Calage bon", "\"decision\" = 'bon'", sym("#2b8a3e", 0.6)),
            ("Calage faux", "\"decision\" = 'faux'", sym("#c92a2a", 0.6, "dash"))):
        r = QgsRuleBasedRenderer.Rule(s, 0, 0, expr, label)
        root.appendChild(r)
    lyr.setRenderer(QgsRuleBasedRenderer(root))
    pal = QgsPalLayerSettings()
    pal.fieldName = ("coalesce(\"nom\", \"commune\") || ' ' || coalesce(to_string(\"annee\"), '') || "
                     "' · vue ' || regexp_substr(\"feuille\", '_(\\\\d+)\\\\.[a-z]+$')")
    pal.isExpression = True
    pal.scaleVisibility = True
    pal.maximumScale = 1
    pal.minimumScale = 60000
    fmt = QgsTextFormat()
    fmt.setColor(QColor("#e8590c"))
    fmt.setSize(9)
    pal.setFormat(fmt)
    lyr.setLabeling(QgsVectorLayerSimpleLabeling(pal))
    lyr.setLabelsEnabled(True)


def _q(t):
    return "'" + str(t).replace("'", "''") + "'"


def _cle(f):
    m = re.search(r"_(\d+)\.\w+$", str(f["feuille"]))
    return (str(f["lot"]), str(f["commune"]), f["annee"] or 0, int(m.group(1)) if m else 0)


def _a_verifier(lyr):
    req = QgsFeatureRequest().setFilterExpression("\"decision\" = 'a_verifier'")
    return sorted(lyr.getFeatures(req), key=_cle)


def _ouvrir(f):
    """Affiche le GeoTIFF de la feuille, zoome dessus, allume les scans de la commune."""
    P = QgsProject.instance()
    root = P.layerTreeRoot()
    grp = root.findGroup(GROUPE) or root.insertGroup(0, GROUPE)
    for node in list(grp.findLayers()):
        if node.layer() is not None and node.layer().name().startswith(APERCU):
            P.removeMapLayer(node.layer().id())
    tif = str(f["geotiff"])
    nom = f"{f['nom'] or f['commune']} {f['annee'] or ''}"
    vue = re.search(r"_(\d+)\.\w+$", str(f["feuille"]))
    r = QgsRasterLayer(_chemin(tif), f"{APERCU} – {nom} vue {vue.group(1) if vue else '?'}")
    if r.isValid():
        r.renderer().setOpacity(0.85)
        P.addMapLayer(r, False)
        grp.addLayer(r)
    else:
        iface.messageBar().pushWarning("Calages", f"GeoTIFF introuvable : {_chemin(tif)}")
    scans = root.findGroup("Scans")
    if scans is not None and f["nom"] not in (None, NULL):
        com = scans.findGroup(str(f["nom"]))
        if com is not None:
            com.setItemVisibilityChecked(True)
            for node in com.findLayers():
                node.setItemVisibilityChecked(True)
    iface.mapCanvas().setExtent(f.geometry().boundingBox().buffered(150))
    iface.mapCanvas().refresh()
    _etat["geotiff"] = tif
    lyr = _couche()
    lyr.removeSelection()      # pas de surbrillance : elle masquerait le scan ; B/F suivent _etat
    reste = len(_a_verifier(lyr))
    mesures = ", ".join(x for x in (
        f"écart au cadastre actuel {f['ecart_m']:.1f} m" if f["ecart_m"] not in (None, NULL) else "",
        f"corrélation z = {f['z']:.1f} (ok vers 6)" if f["z"] not in (None, NULL) else "",
        f"quadrillage {f['pas_m']} m" if f["pas_m"] not in (None, NULL) else "",
        f"appui {f['appui']}" if f["appui"] not in (None, NULL) else "") if x)
    iface.messageBar().pushInfo(
        "Calage", f"{nom} vue {vue.group(1) if vue else '?'} — {f['statut_calage']} — {mesures}. "
                  f"B bon · F faux · C suivant ({reste} à vérifier).")


def suivant():
    lyr = _couche()
    if lyr is None:
        return
    sel = list(lyr.getSelectedFeatures())
    if len(sel) == 1 and str(sel[0]["geotiff"]) != _etat["geotiff"]:
        return _ouvrir(sel[0])               # emprise choisie à la main
    todo = _a_verifier(lyr)
    if not todo:
        iface.messageBar().pushSuccess("Calages", "Plus aucun calage à vérifier.")
        return
    cur = next((f for f in lyr.getFeatures(QgsFeatureRequest().setFilterExpression(
        f"\"geotiff\" = {_q(_etat['geotiff'])}"))), None) if _etat["geotiff"] else None
    apres = [f for f in todo if cur is None or _cle(f) > _cle(cur)]
    _ouvrir((apres or todo)[0])


def decider(decision):
    lyr = _couche()
    if lyr is None:
        return
    sel = list(lyr.getSelectedFeatures())
    if len(sel) > 1 or (sel and str(sel[0]["geotiff"]) != _etat["geotiff"]):
        cibles = sel
    else:
        cibles = [f for f in lyr.getFeatures(QgsFeatureRequest().setFilterExpression(
            f"\"geotiff\" = {_q(_etat['geotiff'])}"))] if _etat["geotiff"] else []
    if not cibles:
        iface.messageBar().pushInfo("Calages", "Aucun calage ouvert : touche C pour commencer.")
        return
    idx = lyr.fields().indexOf("decision")
    if lyr.isEditable():
        lyr.beginEditCommand(f"Calage : {decision}")
        for f in cibles:
            lyr.changeAttributeValue(f.id(), idx, decision)
        lyr.endEditCommand()
    elif not lyr.dataProvider().changeAttributeValues({f.id(): {idx: decision} for f in cibles}):
        iface.messageBar().pushWarning("Calages", "Décision refusée par la base.")
        return
    lyr.reload()
    lyr.triggerRepaint()
    lyr.removeSelection()
    libelle = {"bon": "calage bon : sera extraite au prochain passage de l'extracteur",
               "faux": "calage faux : ne sera pas extraite"}[decision]
    iface.messageBar().pushSuccess("Calages", f"{len(cibles)} feuille(s) — {libelle}.")
    if len(cibles) == 1:
        suivant()


def commandes():
    """(nom, texte, touche, fonction) ajoutées à la barre « Validation annotation »."""
    return [
        ("calage_suivant", "C Calage à vérifier", "C", suivant),
        ("calage_bon", "B Calage bon", "B", lambda: decider("bon")),
        ("calage_faux", "F Calage faux", "F", lambda: decider("faux")),
    ]
