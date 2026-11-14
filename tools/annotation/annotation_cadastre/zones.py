"""Zone utile des feuilles : vérifier et recaler dans QGIS les zones proposées.

La zone utile est la partie de la feuille réellement couverte par ses parcelles et ses
informations utiles (pas le cadre). Elle est tracée fidèlement au dessin du scan, feuille
par feuille, dans la table feuille_zones ; tools/zone/mosaique.py arbitre ensuite les
recouvrements entre feuilles voisines. Les propositions viennent des découpes « DCP »
faites pour GénéFoncier (tools/zone/amorcer_dcp.py), puis du modèle : ce sont des compromis
à recaler sur le dessin. Seules les zones « valide » nourrissent un entraînement.

Touches (carte active, comme le reste de la barre) :
    Z   zone suivante à vérifier (ou celle sélectionnée dans « Zones – à vérifier ») : le plan
        non découpé s'affiche, la couche passe en édition, outil de sommets prêt
    V   valider la zone en cours : enregistre les retouches, statut « valide », puis suivante
    X   rejeter la zone en cours (découpe inutilisable : à redessiner), puis suivante
Pour dessiner une zone manquante : feuille active (liste « Feuille » ou touche 8), couche
« Zones – à vérifier » en édition, outil « Ajouter un polygone » : la feuille et l'origine
« manuel » sont remplies d'office.
En pointillés orange, la découpe d'origine (« Zones – découpe d'origine ») : l'écart avec la
zone validée mesure le compromis que l'on a corrigé.
"""
from qgis.core import (NULL, QgsDataSourceUri, QgsDefaultValue, QgsFeatureRequest, QgsFillSymbol,
                       QgsProject, QgsRasterLayer, QgsRuleBasedRenderer, QgsVectorLayer)
from qgis.utils import iface

from . import calages

SERVICE = "cadastre_annotation"
COUCHE = "Zones – à vérifier"
COUCHE_ORIGINE = "Zones – découpe d'origine"
GROUPE = "Zones utiles"
APERCU = "Plan de la zone"
FILTREES = (COUCHE, COUCHE_ORIGINE)
_etat = {"zone": None}


def _sym(color, width, style="solid", fill=None):
    return QgsFillSymbol.createSimple({
        "style": "solid" if fill else "no", "color": fill or "0,0,0,0", "outline_color": color,
        "outline_width": str(width), "outline_width_unit": "MM", "outline_style": style})


def _regles(lyr, items):
    root = QgsRuleBasedRenderer.Rule(None)
    for label, expr, s in items:
        root.appendChild(QgsRuleBasedRenderer.Rule(s, 0, 0, expr, label))
    lyr.setRenderer(QgsRuleBasedRenderer(root))


def _pg(nom, geom):
    uri = QgsDataSourceUri()
    uri.setParam("service", SERVICE)
    uri.setDataSource("public", "feuille_zones", geom, "", "id")
    uri.setSrid("2154")
    return QgsVectorLayer(uri.uri(False), nom, "postgres")


def _groupe():
    root = QgsProject.instance().layerTreeRoot()
    return root.findGroup(GROUPE) or root.insertGroup(0, GROUPE)


def _couche(creer=True):
    P = QgsProject.instance()
    lyrs = P.mapLayersByName(COUCHE)
    if lyrs:
        return lyrs[0]
    if not creer:
        return None
    lyr = _pg(COUCHE, "geom")
    if not lyr.isValid():
        iface.messageBar().pushWarning("Zones", "Table feuille_zones inaccessible : la créer avec "
                                       "tools/zone/amorcer_dcp.py --lancer (ou annotation_parcelles/schema.sql).")
        return None
    _regles(lyr, [
        ("À vérifier", "\"statut\" = 'a_verifier'", _sym("#f08c00", 0.9, fill="240,140,0,20")),
        ("Valide", "\"statut\" = 'valide'", _sym("#2b8a3e", 0.8, fill="43,138,62,15")),
        ("Rejetée", "\"statut\" = 'rejete'", _sym("#868e96", 0.5, "dash")),
    ])
    f = lyr.fields()
    lyr.setDefaultValueDefinition(f.indexOf("feuille_id"), QgsDefaultValue("@feuille_active"))
    lyr.setDefaultValueDefinition(f.indexOf("origine"), QgsDefaultValue("'manuel'"))
    lyr.setDefaultValueDefinition(f.indexOf("statut"), QgsDefaultValue("'a_verifier'"))
    go = lyr.geometryOptions()
    go.setRemoveDuplicateNodes(True)
    go.setGeometryPrecision(0.01)
    orig = _pg(COUCHE_ORIGINE, "geom_origine")
    if orig.isValid():
        orig.setReadOnly(True)
        _regles(orig, [("Découpe d'origine", "true", _sym("#e8590c", 0.5, "dot"))])
        P.addMapLayer(orig, False)
    P.addMapLayer(lyr, False)
    grp = _groupe()
    grp.insertLayer(0, lyr)
    if orig.isValid():
        grp.insertLayer(1, orig)
    return lyr


def assurer():
    """Crée les couches si la table existe (appelé à l'ouverture du projet), sans message."""
    if QgsProject.instance().mapLayersByName(COUCHE):
        return
    lyr = _pg(COUCHE, "geom")
    if lyr.isValid():
        _couche()


def _cle(f):
    return (str(f["feuille_id"]), str(f["source"] or ""))


def _a_verifier(lyr):
    req = QgsFeatureRequest().setFilterExpression("\"statut\" = 'a_verifier'")
    noms = {fid: nom for fid, nom, _ in _feuilles()}
    return sorted(lyr.getFeatures(req), key=lambda f: (noms.get(str(f["feuille_id"]), ""), _cle(f)))


def _feuilles():
    from . import validation
    return validation._feuilles()


def _raster(f):
    """Plan non découpé de la zone, à défaut le scan de la feuille."""
    from . import validation
    if f["raster"] not in (None, NULL) and str(f["raster"]).strip():
        return validation._raster(f["raster"])
    info = {fid: r for fid, _, r in _feuilles()}
    return info.get(str(f["feuille_id"]), "")


def _ouvrir(f):
    from . import validation
    P = QgsProject.instance()
    lyr = _couche()
    fid = str(f["feuille_id"])
    # la feuille seule (parcelles, propositions, zones) ; on garde le filtre de la couche des zones
    # vide pendant la vérification pour pouvoir passer à la feuille suivante
    validation._filtrer(fid)
    grp = _groupe()
    for node in list(grp.findLayers()):
        if node.layer() is not None and node.layer().name().startswith(APERCU):
            P.removeMapLayer(node.layer().id())
    chemin = _raster(f)
    nom = dict((a, b) for a, b, _ in _feuilles()).get(fid, fid)
    r = QgsRasterLayer(chemin, f"{APERCU} – {nom}") if chemin else None
    if r is not None and r.isValid():
        P.addMapLayer(r, False)
        grp.addLayer(r)                        # sous les zones, au-dessus des autres scans
    else:
        iface.messageBar().pushWarning("Zones", f"Plan introuvable : {chemin or '(aucun)'}")
    iface.mapCanvas().setExtent(f.geometry().boundingBox().buffered(60))
    _etat["zone"] = f["id"]
    lyr.removeSelection()
    iface.setActiveLayer(lyr)
    if not lyr.isEditable():
        lyr.startEditing()
    iface.actionVertexToolActiveLayer().trigger()
    iface.mapCanvas().refresh()
    reste = len(_a_verifier(lyr))
    origine = {"dcp": "découpe GénéFoncier", "ia": "modèle", "parcelles": "union des parcelles",
               "manuel": "tracée à la main"}.get(str(f["origine"]), str(f["origine"]))
    iface.messageBar().pushInfo(
        "Zone", f"{nom} — {origine}. Recaler les sommets sur le dessin, puis V valider · X rejeter · "
                f"Z suivante ({reste} à vérifier).")


def suivant():
    lyr = _couche()
    if lyr is None:
        return
    sel = list(lyr.getSelectedFeatures())
    if len(sel) == 1 and sel[0]["id"] != _etat["zone"]:
        return _ouvrir(sel[0])
    todo = _a_verifier(lyr)
    if not todo:
        iface.messageBar().pushSuccess("Zones", "Plus aucune zone à vérifier.")
        return
    ids = [f["id"] for f in todo]
    i = ids.index(_etat["zone"]) + 1 if _etat["zone"] in ids else 0
    _ouvrir(todo[i % len(todo)])


def decider(statut):
    lyr = _couche()
    if lyr is None:
        return
    sel = list(lyr.getSelectedFeatures())
    if sel:
        cibles = [f.id() for f in sel]
    elif _etat["zone"] is not None:
        cibles = [f.id() for f in lyr.getFeatures(QgsFeatureRequest().setFilterExpression(
            f"\"id\" = '{_etat['zone']}'"))]
    else:
        cibles = []
    if not cibles:
        iface.messageBar().pushInfo("Zones", "Aucune zone ouverte : touche Z pour commencer.")
        return
    if not lyr.isEditable():
        lyr.startEditing()
    idx = lyr.fields().indexOf("statut")
    lyr.beginEditCommand(f"Zone : {statut}")
    for i in cibles:
        lyr.changeAttributeValue(i, idx, statut)
    lyr.endEditCommand()
    if not lyr.commitChanges(False):           # retouches de sommets + décision, d'un coup
        iface.messageBar().pushCritical(
            "Zones", "Enregistrement refusé : " + " ; ".join(lyr.commitErrors())
            + " (une seule zone « valide » par feuille).")
        return
    lyr.removeSelection()
    lyr.reload()
    for n in FILTREES:
        for l in QgsProject.instance().mapLayersByName(n):
            l.triggerRepaint()
    libelle = {"valide": "validée : vérité terrain du modèle de zone et de la mosaïque",
               "rejete": "rejetée : à redessiner à la main"}[statut]
    iface.messageBar().pushSuccess("Zones", f"{len(cibles)} zone(s) {libelle}.")
    if len(cibles) == 1:
        suivant()


def commandes():
    """(nom, texte, touche, fonction) ajoutées à la barre « Validation annotation »."""
    return [
        ("zone_suivante", "Z Zone à vérifier", "Z", suivant),
        ("zone_valider", "V Zone valide", "V", lambda: decider("valide")),
        ("zone_rejeter", "X Zone rejetée", "X", lambda: decider("rejete")),
    ]
