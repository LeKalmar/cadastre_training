"""Validation rapide des propositions IA dans le projet d'annotation (QGIS).

Barre d'outils « Validation annotation ». Deux façons d'aller vite :

1. Touches seules -- actives quand la CARTE a le focus (cliquer une fois dessus),
   jamais pendant la saisie dans un formulaire. Elles agissent sur la sélection
   de la couche « IA – à valider » (quelle que soit la couche active) :

       1   Valider   (« telle quelle » ou « corrigée » est déduit par la base en
                      comparant la géométrie à celle de l'IA : rien à choisir)
       2   Rejeter…  -> menu des motifs, touche 1 à 0 (A = autre) pour choisir
       3   Remettre « à valider »
       4   Aller à la prochaine proposition à valider

2. Outil « Validation au clic » (bouton ✔ de la barre, ou touche 5) :
       clic gauche          valider la parcelle cliquée
       clic droit           rejeter (menu des motifs, touche 1 à 0, A = autre)
       Maj + clic gauche    remettre « à valider »
   Échap ou un autre outil pour en sortir.

3. Touche 6 : afficher / masquer les sommets (sans rien sélectionner), à partir du
   1:2500 environ, sur « IA – à valider » et « Parcelles validées » :
       petit point bleu     sommet
       gros point orange    sommet superflu : l'enlever déplace le contour de moins
                            de @tol_sommets mètres (0,2 par défaut ; à régler dans
                            Projet > Propriétés > Variables, variable « tol_sommets »)
   Suit les modifications en cours d'édition (pas besoin d'enregistrer).

4. Touche 7 : « Transporter un développement » (agrandissement dessiné en marge).
   a. Sélectionner les polygones du développement (couche active, ou « IA – à
      valider » si la couche active n'a pas de sélection), puis touche 7.
   b. Cliquer des paires de points homologues : un point DANS le développement,
      puis le même point à son EMPLACEMENT RÉEL ; recommencer (2 paires minimum,
      3 ou plus = moyenne et contrôle d'erreur). L'accrochage QGIS est utilisé
      (activer l'accrochage aux sommets pour viser juste).
   c. Aperçu en rouge des polygones transportés. Entrée ou clic droit = appliquer,
      Retour arrière = retirer le dernier point, Échap = annuler.
   Transformation de similitude (échelle + rotation + translation), écrite dans
   la session d'édition (Ctrl+Z possible) ; échelle, rotation et erreur affichées.
   La transformation est enregistrée dans le champ « developpement » : la forme
   transportée sert à la prod, mais l'entraînement utilise la position sur le scan
   et ignore l'emplacement réel (export_dataset.py).

   Touche 0 : « Marquer un développement », SANS rien déplacer (l'annotation reste
   là où le dessin se trouve). Sélectionner des polygones (couche active, sinon
   « IA – à valider »), touche 0, puis choisir :
       1  Parcelles du développement n° …      les parcelles dessinées dans l'agrandissement
       2  Emplacement réel du développement n° …  la « fausse parcelle » du plan principal
                                                qui contient le développement ; elle est
                                                validée (son contour est un vrai trait)
       3  Parcelle de contexte du développement n° …  dessinée dans l'agrandissement pour
                                                situer, déjà présente sur le plan principal
       0  Retirer le marquage
   Le numéro relie les parcelles d'un développement à leur emplacement réel ; il est
   propre à la feuille. Violet = parcelles d'un développement, violet épais en tirets =
   emplacement réel, gris-violet = contexte. Ces marques servent de vérité terrain à la
   détection automatique et à tools/developpements/assembler_feuille.py (GénéFoncier).

5. N'afficher qu'une feuille (parcelles, propositions, zones ; son scan seul visible) :
       liste « Feuille » de la barre   choisir une feuille (taper pour chercher), zoom dessus
       8   feuille de la parcelle sous la souris (ou de la sélection) ; 8 à nouveau = tout
       9   tout afficher
   La couche ne doit pas avoir de modifications non enregistrées (QGIS refuse alors le filtre).

6. Calages « à vérifier » du géoréférencement automatique (module calages.py) :
       C   ouvrir le calage suivant (son GeoTIFF en haut de la carte)
       B   calage bon  -> la feuille sera extraite ;  F   calage faux -> jamais extraite

7. Zone utile des feuilles (module zones.py) :
       Z   zone suivante à vérifier (plan non découpé affiché, outil de sommets prêt)
       V   valider la zone (retouches enregistrées) ;  X   rejeter (à redessiner)

8. Numéros de parcelles (module numeros.py), vérité terrain de l'OCR :
       N   outil « Pointer les numéros » : clic gauche sur le numéro écrit (valeur GénéFoncier
           recopiée), clic droit efface ; couche « Numéros à pointer » = ce qui reste

9. Parcelles principales proposées (module principales.py, après tools/typologie/proposer.py) :
       P   suivante · M   valider la parcelle principale et ses typologies
       G   regrouper la sélection · U   détacher la sélection · T   changer le type

Les touches se changent dans Préférences > Raccourcis clavier (« Validation annotation »).
Si la couche est en mode édition, les décisions vont dans la session (Ctrl+Z
possible, à enregistrer avec le reste) ; sinon elles sont écrites tout de suite.

Fait partie de l'extension QGIS « Annotation cadastrale » (plugin.py appelle
installer() / desinstaller()). Après une modification du code : Extensions >
Gérer les extensions, décocher puis recocher (ou l'extension « Plugin Reloader »).
"""
import json
import os

from qgis.core import (NULL, Qgis, QgsExpressionContextUtils, QgsFeatureRequest, QgsFillSymbol, QgsRenderContext, QgsGeometry,
                       QgsGeometryGeneratorSymbolLayer, QgsMarkerSymbol, QgsProject, QgsRasterLayer, QgsRectangle, QgsRuleBasedRenderer,
                       QgsWkbTypes)
from qgis.gui import QgsMapTool, QgsRubberBand, QgsVertexMarker
from qgis.PyQt.QtCore import QSize, Qt
from qgis.PyQt.QtGui import QColor, QCursor, QIcon, QKeySequence, QTransform
from qgis.PyQt.QtWidgets import QAction, QComboBox, QCompleter, QMenu, QToolBar
from qgis.utils import iface

from . import calages, numeros, principales, zones

COUCHE = "IA – à valider"
RAFRAICHIR = ("Parcelles validées", "IA – à valider", "Avancement")
MOTIFS = [("Bruit / faux trait", "bruit"), ("Texte / numéro", "texte"), ("Hachures", "hachures"),
          ("Double trait (fossé, chemin)", "double_trait"), ("Marge / cadre", "marge_cadre"),
          ("Hors feuille", "hors_feuille"), ("Doublon", "doublon"),
          ("Parcelle coupée en trop (sur-segmentation)", "sur_segmentation"),
          ("Parcelles fusionnées (sous-segmentation)", "sous_segmentation"),
          ("Route, chemin, rivière, canal…", "voirie_eau"), ("Autre", "autre")]
TOUCHES_REJET = "1234567890a"   # « Autre » passe sur la touche A
TOOLBAR = "AnnotationValidationToolbar"
VALIDER = ({"review_status": "approved"}, "validée")
ATTENTE = ({"review_status": "pending"}, "à valider")
COUCHES_SOMMETS = ("IA – à valider", "Parcelles validées")
ECHELLE_SOMMETS = 2500
_ZOOM = f"@map_scale <= {ECHELLE_SOMMETS}"
_TOL = "coalesce(to_real(@tol_sommets), 0.2)"
SOMMETS = [  # (expression, style du point) -- l'ordre = ordre de dessin
    (f"CASE WHEN {_ZOOM} THEN nodes_to_points($geometry) END",
     {"name": "circle", "size": "1.2", "color": "#1f4e9e", "outline_color": "#ffffff",
      "outline_width": "0.2"}),
    (f"CASE WHEN {_ZOOM} THEN difference(nodes_to_points($geometry), "
     f"nodes_to_points(simplify($geometry, {_TOL}))) END",
     {"name": "circle", "size": "2.4", "color": "#ff7f00", "outline_color": "#000000",
      "outline_width": "0.3"}),
]


def _couche():
    lyrs = QgsProject.instance().mapLayersByName(COUCHE)
    if not lyrs:
        iface.messageBar().pushWarning("Validation", f"Couche « {COUCHE} » absente du projet.")
        return None
    return lyrs[0]


def _appliquer(values, libelle, ids=None):
    lyr = _couche()
    if lyr is None:
        return
    selection = ids is None
    ids = list(lyr.selectedFeatureIds()) if selection else list(ids)
    if not ids:
        iface.messageBar().pushInfo("Validation", f"Aucune parcelle sélectionnée dans « {COUCHE} ».")
        return
    if values.get("review_status") != "rejected":
        values = dict(values, motif_rejet=None)
    attrs = {lyr.fields().indexOf(k): v for k, v in values.items()}
    if lyr.isEditable():
        lyr.beginEditCommand(f"Validation : {libelle}")
        for i in ids:
            for k, v in attrs.items():
                lyr.changeAttributeValue(i, k, v)
        lyr.endEditCommand()
        ou = "session d'édition, à enregistrer"
    else:
        if not lyr.dataProvider().changeAttributeValues({i: attrs for i in ids}):
            iface.messageBar().pushCritical("Validation", "Écriture refusée par la base (voir le journal).")
            return
        lyr.reload()
        ou = "enregistré"
    if selection:
        lyr.removeSelection()
    for n in RAFRAICHIR:
        for l in QgsProject.instance().mapLayersByName(n):
            l.triggerRepaint()
    iface.messageBar().pushSuccess("Validation", f"{len(ids)} parcelle(s) → {libelle} ({ou}).", 2)


# ----------------------------------------------------------------------------- développements (marquage)
ROLES_DEV = [("parcelle", "Parcelles du développement"),
             ("emplacement", "Emplacement réel (fausse parcelle) du développement"),
             ("contexte", "Parcelle de contexte du développement")]
STYLES_DEV = [  # règles ajoutées au style des couches de parcelles (dessinées par-dessus)
    ("Développement – parcelle", "\"dev_role\" = 'parcelle'",
     {"color": "150,60,220,35", "outline_color": "#7a1fd1", "outline_width": "0.6"}),
    ("Développement – emplacement réel", "\"dev_role\" = 'emplacement'",
     {"color": "150,60,220,15", "outline_color": "#7a1fd1", "outline_width": "1.2", "outline_style": "dash"}),
    ("Développement – contexte", "\"dev_role\" = 'contexte'",
     {"color": "120,110,140,35", "outline_color": "#6f6880", "outline_width": "0.6", "outline_style": "dot"}),
]


def _champs_dev(lyr):
    """Vrai si la couche connaît dev_role / dev_groupe (rechargée au besoin : les
    colonnes ont pu être ajoutées à la base après l'ouverture du projet)."""
    if lyr.fields().indexOf("dev_role") < 0:
        lyr.dataProvider().reloadData()
        lyr.updateFields()
    return lyr.fields().indexOf("dev_role") >= 0 and lyr.fields().indexOf("dev_groupe") >= 0


def _couche_dev():
    """Couche active si elle a une sélection et les champs de développement, sinon « IA – à valider »."""
    lyr = iface.activeLayer()
    if (lyr is not None and hasattr(lyr, "selectedFeatureCount") and lyr.selectedFeatureCount()
            and lyr.fields().indexOf("feuille_id") >= 0 and _champs_dev(lyr)):
        return lyr
    lyr = _couche()
    if lyr is not None and not _champs_dev(lyr):
        iface.messageBar().pushWarning(
            "Développement", "Champs dev_role / dev_groupe absents : fermer puis rouvrir le projet.")
        return None
    return lyr


def _groupes_dev(feuille_ids):
    """Numéros de développement déjà utilisés sur ces feuilles (toutes couches de parcelles)."""
    if not feuille_ids:
        return []
    liste = ", ".join("'%s'" % f for f in feuille_ids)
    out = set()
    for nom in COUCHES_SOMMETS:
        for l in QgsProject.instance().mapLayersByName(nom):
            if l.fields().indexOf("dev_groupe") < 0:
                continue
            req = QgsFeatureRequest().setFilterExpression(
                f"\"dev_groupe\" IS NOT NULL AND \"feuille_id\" IN ({liste})")
            req.setFlags(QgsFeatureRequest.Flag.NoGeometry)
            req.setSubsetOfAttributes(["dev_groupe"], l.fields())
            out |= {int(f["dev_groupe"]) for f in l.getFeatures(req) if f["dev_groupe"] not in (None, NULL)}
    return sorted(out)


def _marquer_dev(lyr, ids, role, groupe):
    values = {"dev_role": role, "dev_groupe": groupe if role else None}
    # l'emplacement réel est un contour réellement tracé : une proposition encore
    # « à valider » est validée (elle n'entre jamais en prod comme parcelle)
    valider = role == "emplacement" and lyr.fields().indexOf("review_status") >= 0
    attrs = {lyr.fields().indexOf(k): v for k, v in values.items()}
    i_rs = lyr.fields().indexOf("review_status")
    par_id = {}
    for f in lyr.getFeatures(QgsFeatureRequest().setFilterFids(ids).setFlags(QgsFeatureRequest.Flag.NoGeometry)):
        a = dict(attrs)
        if valider and f["review_status"] in ("pending", "rejected"):
            a[i_rs] = "approved"
            a[lyr.fields().indexOf("motif_rejet")] = None
        par_id[f.id()] = a
    libelle = "marquage retiré" if not role else f"{dict(ROLES_DEV)[role].lower()} n° {groupe}"
    if lyr.isEditable():
        lyr.beginEditCommand(f"Développement : {libelle}")
        for i, a in par_id.items():
            for k, v in a.items():
                lyr.changeAttributeValue(i, k, v)
        lyr.endEditCommand()
        ou = "session d'édition, à enregistrer"
    else:
        if not lyr.dataProvider().changeAttributeValues(par_id):
            iface.messageBar().pushCritical("Développement", "Écriture refusée par la base (voir le journal).")
            return
        lyr.reload()
        ou = "enregistré"
    lyr.removeSelection()
    for n in RAFRAICHIR:
        for l in QgsProject.instance().mapLayersByName(n):
            l.triggerRepaint()
    iface.messageBar().pushSuccess("Développement", f"{len(par_id)} polygone(s) → {libelle} ({ou}).", 3)


def _menu_dev():
    lyr = _couche_dev()
    if lyr is None:
        return
    ids = list(lyr.selectedFeatureIds())
    if not ids:
        iface.messageBar().pushInfo(
            "Développement", f"Sélectionner d'abord des polygones dans « {lyr.name()} », puis touche 0.")
        return
    req = QgsFeatureRequest().setFilterFids(ids).setFlags(QgsFeatureRequest.Flag.NoGeometry)
    feuilles = {str(f["feuille_id"]) for f in lyr.getFeatures(req)}
    if len(feuilles) != 1:
        iface.messageBar().pushWarning(
            "Développement", "La sélection couvre plusieurs feuilles : un développement appartient à une seule.")
        return
    groupes = _groupes_dev(feuilles)
    nouveau = (max(groupes) + 1) if groupes else 1
    m = QMenu(iface.mainWindow())
    m.addAction(f"{len(ids)} polygone(s) de « {lyr.name()} »").setEnabled(False)
    for k, (role, label) in enumerate(ROLES_DEV, 1):
        sm = m.addMenu(f"&{k}  {label}…")
        for g in groupes:
            a = sm.addAction(f"n° &{g}" if g < 10 else f"n° {g}")
            a.triggered.connect(lambda _=False, r=role, gg=g: _marquer_dev(lyr, ids, r, gg))
        a = sm.addAction(f"&nouveau développement (n° {nouveau})")
        a.triggered.connect(lambda _=False, r=role: _marquer_dev(lyr, ids, r, nouveau))
    m.addSeparator()
    a = m.addAction("&0  Retirer le marquage")
    a.triggered.connect(lambda _=False: _marquer_dev(lyr, ids, None, None))
    m.exec(QCursor.pos())


def _style_dev():
    """Ajoute au style des couches de parcelles les règles qui montrent les développements."""
    for nom in COUCHES_SOMMETS:
        for lyr in QgsProject.instance().mapLayersByName(nom):
            r = lyr.renderer()
            if r is None or not hasattr(r, "rootRule") or not _champs_dev(lyr):
                continue
            root = r.rootRule()
            if any(ch.label().startswith("Développement –") for ch in root.children()):
                continue
            for label, expr, props in STYLES_DEV:
                root.appendChild(QgsRuleBasedRenderer.Rule(QgsFillSymbol.createSimple(props), 0, 0, expr, label))
            lyr.triggerRepaint()
            iface.layerTreeView().refreshLayerSymbology(lyr.id())


def _suivante():
    lyr = _couche()
    if lyr is None:
        return
    req = QgsFeatureRequest().setFilterExpression("review_status = 'pending'")
    req.addOrderBy("aire_m2", False)
    req.setLimit(1)
    f = next(lyr.getFeatures(req), None)
    if f is None:
        iface.messageBar().pushSuccess("Validation", "Plus aucune proposition à valider ici.")
        return
    lyr.selectByIds([f.id()])
    bb = f.geometry().boundingBox()
    iface.mapCanvas().setExtent(bb.buffered(max(15.0, bb.width())))
    iface.mapCanvas().refresh()


def _symboles(lyr):
    r = lyr.renderer()
    if r is None:
        return []
    if hasattr(r, "rootRule"):
        return [ru.symbol() for ru in r.rootRule().descendants() if ru.symbol() is not None]
    return [s for s in r.symbols(QgsRenderContext()) if s is not None]


def _est_sommet(sl):
    return (isinstance(sl, QgsGeometryGeneratorSymbolLayer)
            and "nodes_to_points" in sl.geometryExpression())


def _sommets():
    """Bascule l'affichage des sommets sur les couches de COUCHES_SOMMETS."""
    lyrs = [l for n in COUCHES_SOMMETS for l in QgsProject.instance().mapLayersByName(n)]
    if not lyrs:
        iface.messageBar().pushWarning("Sommets", "Aucune couche de parcelles trouvée.")
        return
    actif = any(_est_sommet(sl) for l in lyrs for sym in _symboles(l) for sl in sym.symbolLayers())
    for l in lyrs:
        for sym in _symboles(l):
            for i in reversed(range(sym.symbolLayerCount())):
                if _est_sommet(sym.symbolLayer(i)):
                    sym.deleteSymbolLayer(i)
            if not actif:
                for expr, props in SOMMETS:
                    gg = QgsGeometryGeneratorSymbolLayer.create(
                        {"geometryModifier": expr, "SymbolType": "Marker"})
                    gg.setSubSymbol(QgsMarkerSymbol.createSimple(props))
                    sym.appendSymbolLayer(gg)
        l.triggerRepaint()
        iface.layerTreeView().refreshLayerSymbology(l.id())
    if actif:
        iface.messageBar().pushInfo("Sommets", "Sommets masqués.")
    else:
        iface.messageBar().pushInfo(
            "Sommets", f"Sommets affichés (zoom ≥ 1:{ECHELLE_SOMMETS}) : orange = superflu "
                       "(variable de projet tol_sommets, 0,2 m par défaut).")


def _menu_rejet(ids=None):
    m = QMenu(iface.mainWindow())
    for k, (label, motif) in enumerate(MOTIFS):
        touche = TOUCHES_REJET[k]
        a = m.addAction(f"&{touche}  {label}")
        a.triggered.connect(lambda _=False, mo=motif, la=label:
                            _appliquer({"review_status": "rejected", "motif_rejet": mo},
                                       f"rejetée ({la})", ids))
    m.exec_(QCursor.pos())


class OutilValidation(QgsMapTool):
    """Clic gauche = valider, clic droit = rejeter, Maj+clic = remettre à valider."""

    def __init__(self, canvas):
        super().__init__(canvas)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def _parcelle(self, e):
        lyr = _couche()
        if lyr is None:
            return None
        pt = self.toLayerCoordinates(lyr, e.mapPoint())
        tol = self.searchRadiusMU(self.canvas())
        req = QgsFeatureRequest().setFilterRect(
            QgsRectangle(pt.x() - tol, pt.y() - tol, pt.x() + tol, pt.y() + tol))
        geom_pt = QgsGeometry.fromPointXY(pt)
        best = None
        for f in lyr.getFeatures(req):
            g = f.geometry()
            if g.contains(geom_pt):            # la plus petite qui contient le clic
                if best is None or g.area() < best[1]:
                    best = (f.id(), g.area())
        return best[0] if best else None

    def canvasReleaseEvent(self, e):
        fid = self._parcelle(e)
        if fid is None:
            return
        # pas de sélection (le jaune masque tout) ; la sélection en cours est conservée
        if e.button() == Qt.MouseButton.RightButton:
            # simple contour fin le temps du menu, pour voir quelle parcelle est visée
            lyr = _couche()
            rb = QgsRubberBand(self.canvas(), QgsWkbTypes.GeometryType.PolygonGeometry)
            rb.setColor(QColor(0, 0, 0, 0))
            rb.setStrokeColor(QColor("#e31a1c"))
            rb.setWidth(2)
            rb.setToGeometry(next(lyr.getFeatures(QgsFeatureRequest(fid))).geometry(), lyr)
            try:
                _menu_rejet([fid])
            finally:
                rb.reset(QgsWkbTypes.GeometryType.PolygonGeometry)
                self.canvas().scene().removeItem(rb)
        elif e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            _appliquer(*ATTENTE, ids=[fid])
        else:
            _appliquer(*VALIDER, ids=[fid])

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            iface.actionPan().trigger()


FEUILLES = "Feuilles"
FILTREES = ("Parcelles validées", "IA – à valider", "IA – géométrie d'origine",
            "Zones validées", "Zones ignorées") + numeros.FILTREES + principales.FILTREES
_etat_feuille = {"fid": "", "combo": None}


def _raster(path):
    """Chemin absolu normalisé du scan. import_ia.py enregistre les chemins relatifs à la
    racine du dépôt (data/georef_…) : le dossier courant de QGIS n'y est pas."""
    if path in (None, NULL) or not str(path).strip():
        return ""
    return os.path.normpath(os.path.abspath(calages._chemin(str(path))))


def _sources(grp):
    return {os.path.normpath(n.layer().source()): n for n in grp.findLayers() if n.layer() is not None}


def _feuilles():
    """[(id, nom, raster)] triées par nom, lues dans la couche « Feuilles »."""
    lyrs = QgsProject.instance().mapLayersByName(FEUILLES)
    if not lyrs:
        return []
    lyr = lyrs[0]
    return sorted(((str(f["id"]), str(f["nom"]), _raster(f["raster"])) for f in lyr.getFeatures()),
                  key=lambda t: t[1])


def _sync_scans(message=True):
    """Ajoute au groupe « Scans » les rasters des feuilles qui n'y sont pas encore
    (feuilles importées après la génération du projet). Rangés par commune, éteints."""
    P = QgsProject.instance()
    lyrs = P.mapLayersByName(FEUILLES)
    if not lyrs:
        return 0
    fl = lyrs[0]
    root = P.layerTreeRoot()
    grp = root.findGroup("Scans")
    if grp is None:
        grp = root.addGroup("Scans")
    deja = set(_sources(grp))
    a_commune = fl.fields().indexOf("commune") >= 0
    ajout = 0
    for f in fl.getFeatures():
        path = _raster(f["raster"])
        if not path or path in deja or not os.path.exists(path):
            continue
        deja.add(path)
        r = QgsRasterLayer(path, str(f["nom"]))
        if not r.isValid():
            continue
        P.addMapLayer(r, False)
        dest = grp
        com = f["commune"] if a_commune else None
        if com not in (None, NULL) and str(com).strip():
            dest = grp.findGroup(str(com))
            if dest is None:
                dest = grp.addGroup(str(com))
                dest.setExpanded(False)
        dest.addLayer(r).setItemVisibilityChecked(False)
        ajout += 1
    if ajout and message:
        iface.messageBar().pushInfo("Scans", f"{ajout} scan(s) ajouté(s) au groupe « Scans » "
                                             "(enregistrer le projet pour les garder).")
    return ajout


def _actualiser_feuilles():
    n = _sync_scans(message=False)
    _remplir_combo()
    iface.messageBar().pushInfo("Feuilles", f"Liste des feuilles actualisée ; {n} nouveau(x) scan(s) "
                                            "ajouté(s) au groupe « Scans ».")


def _filtrer(fid, zoom=False):
    """N'affiche que la feuille ``fid`` ('' = tout) sur les couches de FILTREES."""
    P = QgsProject.instance()
    refus = []
    for name in FILTREES:
        for lyr in P.mapLayersByName(name):
            if not lyr.setSubsetString(f"feuille_id = '{fid}'" if fid else ""):
                refus.append(name)
    if refus:
        iface.messageBar().pushWarning(
            "Feuille", "Filtre refusé sur « " + "», « ".join(refus) + " » : enregistrer d'abord ses modifications.")
        return
    QgsExpressionContextUtils.setProjectVariable(P, "feuille_active", fid)
    _etat_feuille["fid"] = fid
    info = {f[0]: f for f in _feuilles()}
    nom = info.get(fid, (fid, "", ""))[1] if fid else ""
    if fid:
        raster = info.get(fid, ("", "", ""))[2]
        grp = P.layerTreeRoot().findGroup("Scans")
        if raster and (grp is None or raster not in _sources(grp)):
            _sync_scans()                      # feuille importée depuis l'ouverture du projet
            grp = P.layerTreeRoot().findGroup("Scans")
        if grp and raster:
            for src, node in _sources(grp).items():
                node.setItemVisibilityChecked(src == raster)
            node = _sources(grp).get(raster)
            if node is None:
                iface.messageBar().pushWarning("Feuille", f"Scan introuvable : {raster}")
            else:                              # allumer aussi les groupes parents (Scans, commune)
                parent = node.parent()
                while parent is not None:
                    parent.setItemVisibilityChecked(True)
                    parent = parent.parent()
        if zoom:
            fl = P.mapLayersByName(FEUILLES)[0]
            f = next(fl.getFeatures(QgsFeatureRequest().setFilterExpression(f"id = '{fid}'")), None)
            if f is not None:
                iface.mapCanvas().setExtent(f.geometry().boundingBox().buffered(20))
    combo = _etat_feuille["combo"]
    if combo is not None:
        combo.blockSignals(True)
        i = combo.findData(fid)
        combo.setCurrentIndex(max(0, i))
        combo.blockSignals(False)
    iface.mapCanvas().refresh()
    iface.messageBar().pushInfo("Feuille", f"Feuille affichée : {nom}" if fid else "Toutes les feuilles affichées.")


def _feuille_sous_souris():
    if _etat_feuille["fid"]:                   # 2e appui : tout réafficher
        _filtrer("")
        return
    canvas = iface.mapCanvas()
    pt = canvas.getCoordinateTransform().toMapCoordinates(canvas.mouseLastXY())
    fid = None
    for name in ("IA – à valider", "Parcelles validées"):
        for lyr in QgsProject.instance().mapLayersByName(name):
            if lyr.selectedFeatureCount():
                fid = str(next(lyr.getSelectedFeatures())["feuille_id"])
                break
            p = canvas.mapSettings().mapToLayerCoordinates(lyr, pt)
            gp = QgsGeometry.fromPointXY(p)
            best = None
            for f in lyr.getFeatures(QgsFeatureRequest().setFilterRect(gp.boundingBox().buffered(0.01))):
                g = f.geometry()
                if g.contains(gp) and (best is None or g.area() < best[1]):
                    best = (str(f["feuille_id"]), g.area())
            if best:
                fid = best[0]
                break
        if fid:
            break
    if not fid:
        iface.messageBar().pushInfo("Feuille", "Placer la souris sur une parcelle (ou en sélectionner une), puis 8.")
        return
    _filtrer(fid)


def _remplir_combo():
    combo = _etat_feuille["combo"]
    if combo is None:
        return
    _etat_feuille["fid"] = QgsExpressionContextUtils.projectScope(QgsProject.instance()).variable(
        "feuille_active") or ""
    combo.blockSignals(True)
    combo.clear()
    combo.addItem("— Toutes les feuilles —", "")
    for fid, nom, _ in _feuilles():
        combo.addItem(nom, fid)
    combo.setCurrentIndex(max(0, combo.findData(_etat_feuille["fid"])))
    combo.blockSignals(False)


def _combo_feuilles(tb):
    combo = QComboBox(tb)
    combo.setObjectName("annotation_combo_feuille")
    combo.setEditable(True)
    combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    combo.setMinimumContentsLength(22)
    combo.setToolTip("N'afficher qu'une feuille (taper pour chercher)")
    comp = combo.completer()
    comp.setFilterMode(Qt.MatchFlag.MatchContains)
    comp.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
    combo.activated.connect(lambda i: _filtrer(combo.itemData(i) or "", zoom=True))
    _etat_feuille["combo"] = combo
    _remplir_combo()
    return combo


def _similitude(src, dst):
    """Similitude aux moindres carrés dst ≈ z·src + t (nombres complexes).
    Renvoie (QTransform, échelle, rotation °, erreur max m)."""
    import cmath

    a = [complex(p.x(), p.y()) for p in src]
    b = [complex(p.x(), p.y()) for p in dst]
    am, bm = sum(a) / len(a), sum(b) / len(b)
    den = sum(abs(x - am) ** 2 for x in a)
    z = sum((y - bm) * (x - am).conjugate() for x, y in zip(a, b)) / den
    t = bm - z * am
    err = max(abs(z * x + t - y) for x, y in zip(a, b))
    # x' = zr·x − zi·y + tx ; y' = zi·x + zr·y + ty
    qt = QTransform(z.real, z.imag, -z.imag, z.real, t.real, t.imag)
    return qt, abs(z), cmath.phase(z) * 180 / cmath.pi, err


class OutilDeveloppement(QgsMapTool):
    """Paires de clics (développement -> emplacement réel), puis similitude."""

    def __init__(self, canvas):
        super().__init__(canvas)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.pts = []       # coordonnées couche (calcul)
        self.vus = []       # coordonnées carte (affichage)
        self.items = []
        self.apercu = None
        self.lyr = None

    def activate(self):
        super().activate()
        lyr = iface.activeLayer()
        if lyr is None or not hasattr(lyr, "selectedFeatureCount") or not lyr.selectedFeatureCount():
            lyr = _couche()
        if lyr is None or not lyr.selectedFeatureCount():
            iface.messageBar().pushWarning(
                "Développement", "Sélectionner d'abord les polygones du développement, puis touche 7.")
            self.lyr = None
        elif lyr.fields().indexOf("developpement") < 0:
            iface.messageBar().pushCritical(
                "Développement", f"La couche « {lyr.name()} » n'a pas le champ « developpement » : "
                "fermer et rouvrir le projet (colonne ajoutée en base), sinon le transport ne serait "
                "pas exclu de l'entraînement.")
            self.lyr = None
        else:
            self.lyr = lyr
            iface.messageBar().pushInfo(
                "Développement", f"{lyr.selectedFeatureCount()} polygone(s) de « {lyr.name()} ». Cliquer un point "
                "du développement puis son emplacement réel (2 paires min.). Entrée = appliquer.")
        self._reset()

    def deactivate(self):
        self._reset()
        super().deactivate()

    def _reset(self):
        scene = self.canvas().scene()
        for it in self.items + ([self.apercu] if self.apercu else []):
            if isinstance(it, QgsRubberBand):
                it.reset()
            scene.removeItem(it)
        self.items, self.apercu, self.pts, self.vus = [], None, [], []

    def _redessiner(self):
        pts = self.pts
        self._reset_visuel()
        for k, p in enumerate(self.vus):
            m = QgsVertexMarker(self.canvas())
            m.setCenter(p)
            m.setColor(QColor("#1f78b4" if k % 2 == 0 else "#e31a1c"))
            m.setIconType(QgsVertexMarker.IconType.ICON_CROSS if k % 2 == 0 else QgsVertexMarker.IconType.ICON_BOX)
            m.setIconSize(12)
            m.setPenWidth(3)
            self.items.append(m)
            if k % 2 == 1:
                rb = QgsRubberBand(self.canvas(), QgsWkbTypes.GeometryType.LineGeometry)
                rb.setColor(QColor(80, 80, 80, 200))
                rb.setWidth(1)
                rb.setLineStyle(Qt.PenStyle.DashLine)
                rb.setToGeometry(QgsGeometry.fromPolylineXY([self.vus[k - 1], p]))
                self.items.append(rb)
        n = len(pts) // 2
        if n >= 2 and self.lyr is not None:
            qt, ech, rot, err = self._transfo()
            rb = QgsRubberBand(self.canvas(), QgsWkbTypes.GeometryType.PolygonGeometry)
            rb.setColor(QColor(0, 0, 0, 0))
            rb.setFillColor(QColor(227, 26, 28, 40))
            rb.setStrokeColor(QColor("#e31a1c"))
            rb.setWidth(2)
            for f in self.lyr.getSelectedFeatures():
                g = QgsGeometry(f.geometry())
                g.transform(qt)
                rb.addGeometry(g, self.lyr)
            self.apercu = rb
            iface.messageBar().pushInfo(
                "Développement", f"{n} paires : échelle ×{ech:.4f} (1/{1 / ech:.2f}), rotation {rot:+.2f}°"
                + (f", erreur max {err:.2f} m" if n >= 3 else "") + ". Entrée = appliquer.")

    def _reset_visuel(self):
        scene = self.canvas().scene()
        for it in self.items + ([self.apercu] if self.apercu else []):
            if isinstance(it, QgsRubberBand):
                it.reset()
            scene.removeItem(it)
        self.items, self.apercu = [], None

    def _transfo(self):
        n = len(self.pts) // 2
        src = [self.pts[2 * k] for k in range(n)]
        dst = [self.pts[2 * k + 1] for k in range(n)]
        return _similitude(src, dst)

    def canvasReleaseEvent(self, e):
        if self.lyr is None:
            return
        if e.button() == Qt.MouseButton.RightButton:
            self._appliquer()
            return
        p = e.snapPoint()                       # accrochage du projet s'il est actif
        self.vus.append(p)
        self.pts.append(self.toLayerCoordinates(self.lyr, p))
        self._redessiner()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._appliquer()
        elif e.key() == Qt.Key.Key_Backspace and self.pts:
            self.pts.pop()
            self.vus.pop()
            self._redessiner()
        elif e.key() == Qt.Key.Key_Escape:
            self._reset()
            iface.actionPan().trigger()

    def _appliquer(self):
        if len(self.pts) // 2 < 2:
            iface.messageBar().pushWarning("Développement", "Il faut au moins 2 paires de points.")
            return
        lyr = self.lyr
        qt, ech, rot, err = self._transfo()
        if not lyr.isEditable():
            lyr.startEditing()
        ids = list(lyr.selectedFeatureIds())
        idev = lyr.fields().indexOf("developpement")
        lyr.beginEditCommand("Transport d'un développement")
        ok = 0
        for f in lyr.getSelectedFeatures():
            g = QgsGeometry(f.geometry())
            if (g.transform(qt) == Qgis.GeometryOperationResult.Success
                    and lyr.changeGeometry(f.id(), g)):
                total = qt
                old = f.attribute(idev)
                if old not in (None, NULL, ""):         # déjà transporté : on compose
                    try:
                        m = json.loads(old)
                        total = QTransform(*m) * qt     # ancien d'abord, puis le nouveau
                    except (ValueError, TypeError):
                        pass
                lyr.changeAttributeValue(f.id(), idev, json.dumps(
                    [total.m11(), total.m12(), total.m21(), total.m22(), total.dx(), total.dy()]))
                ok += 1
        lyr.endEditCommand()
        lyr.selectByIds(ids)
        lyr.triggerRepaint()
        self._reset()
        iface.messageBar().pushSuccess(
            "Développement", f"{ok} polygone(s) transporté(s) : échelle 1/{1 / ech:.2f}, rotation {rot:+.2f}°. "
            "Vérifier puis enregistrer la couche (Ctrl+Z pour annuler).", 8)
        iface.actionPan().trigger()


_etat = {}   # outils, barre et connexions installés (pour desinstaller)


def _retirer_actions():
    """Retire nos actions (objectName « annotation_* ») : sinon leurs raccourcis
    restent actifs et deviennent « ambigus » au rechargement."""
    mw = iface.mainWindow()
    canvas = iface.mapCanvas()
    actions = {a for w in (mw, canvas) for a in w.findChildren(QAction) + w.actions()
               if a.objectName().startswith("annotation_")}
    for a in actions:
        iface.unregisterMainWindowAction(a)
        a.setShortcut(QKeySequence())
        canvas.removeAction(a)                   # c'est la carte qui porte le raccourci
        mw.removeAction(a)
        a.setParent(None)
        a.deleteLater()
    old = mw.findChild(QToolBar, TOOLBAR)
    if old is not None:
        mw.removeToolBar(old)
        old.setParent(None)
        old.deleteLater()


def desinstaller():
    canvas = iface.mapCanvas()
    for cle in ("outil", "outil_dev", "outil_num"):
        if _etat.get(cle) is not None and canvas.mapTool() is _etat[cle]:
            canvas.unsetMapTool(_etat[cle])
    if _etat.get("projet_lu") is not None:
        try:
            iface.projectRead.disconnect(_etat["projet_lu"])
        except (TypeError, RuntimeError):
            pass
    _etat.clear()
    _etat_feuille.update(fid="", combo=None)
    _retirer_actions()


def installer():
    desinstaller()                               # idempotent : rechargement à chaud
    mw = iface.mainWindow()
    canvas = iface.mapCanvas()

    tb = QToolBar("Validation annotation", mw)
    tb.setObjectName(TOOLBAR)
    mw.addToolBar(tb)

    outil = OutilValidation(canvas)
    _etat["outil"] = outil                       # garder une référence (sinon détruit)
    outil_dev = OutilDeveloppement(canvas)
    _etat["outil_dev"] = outil_dev
    a_dev = QAction("⤢ Développement", tb)
    a_dev.setCheckable(True)
    outil_dev.setAction(a_dev)

    def basculer_dev(on):
        if on:
            canvas.setMapTool(outil_dev)
        elif canvas.mapTool() is outil_dev:
            iface.actionPan().trigger()
    a_dev.toggled.connect(basculer_dev)
    a_outil = QAction("✔ Validation au clic", tb)
    a_outil.setCheckable(True)
    outil.setAction(a_outil)

    def basculer(on):
        if on:
            canvas.setMapTool(outil)
        elif canvas.mapTool() is outil:
            iface.actionPan().trigger()
    a_outil.toggled.connect(basculer)

    cmds = [
        ("valider", "1 Valider", "1", lambda: _appliquer(*VALIDER), None),
        ("rejeter", "2 Rejeter…", "2", _menu_rejet, None),
        ("attente", "3 À valider", "3", lambda: _appliquer(*ATTENTE), None),
        ("suivante", "4 Suivante", "4", _suivante, None),
        ("outil", None, "5", None, a_outil),
        ("sommets", "6 Sommets", "6", _sommets, None),
        ("developpement", None, "7", None, a_dev),
        ("marquer_dev", "0 Marquer développement…", "0", _menu_dev, None),
        ("feuille", "8 Feuille sous la souris", "8", _feuille_sous_souris, None),
        ("toutes", "9 Toutes les feuilles", "9", lambda: _filtrer(""), None),
        ("actualiser", "↻ Nouvelles feuilles", "", _actualiser_feuilles, None),
        ("numeros", None, "N", None, numeros.action(tb, canvas, _etat)),
    ] + [(nom, texte, seq, fn, None) for nom, texte, seq, fn in calages.commandes() + zones.commandes() + principales.commandes()]
    # icônes seules (le nom complet et la touche restent dans l'infobulle) ; séparateur entre familles
    icones = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icones")
    tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
    tb.setIconSize(QSize(24, 24))
    separer = {"developpement", "feuille", "numeros", "calage_suivant", "zone_suivante", "main_suivante"}
    for nom, texte, seq, fn, action in cmds:
        if nom in separer:
            tb.addSeparator()
        a = action or QAction(texte, tb)
        svg = os.path.join(icones, f"{nom}.svg")
        if os.path.exists(svg):
            a.setIcon(QIcon(svg))
        a.setObjectName("annotation_" + nom)
        # touche seule : uniquement quand la carte a le focus (pas dans les formulaires)
        a.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
        if fn is not None:
            a.triggered.connect(lambda _=False, f=fn: f())
        a.setToolTip(f"{a.text()} — touche {seq} (carte active)" if seq else a.text())
        iface.registerMainWindowAction(a, seq)
        canvas.addAction(a)
        tb.addAction(a)
    tb.addSeparator()
    tb.addWidget(_combo_feuilles(tb))
    # l'extension est chargée avant l'ouverture du projet : remplir la liste à chaque projet
    def projet_lu():
        _sync_scans()
        _remplir_combo()
        _style_dev()
        zones.assurer()
        numeros.assurer()
        principales.assurer()
    _etat["projet_lu"] = projet_lu
    iface.projectRead.connect(projet_lu)
    _style_dev()                                 # projet déjà ouvert (rechargement à chaud)
    zones.assurer()
    numeros.assurer()
    principales.assurer()
    iface.messageBar().pushInfo(
        "Validation", "Carte active : 1 valider · 2 rejeter · 3 à valider · 4 suivante · "
                      "5 outil clic (gauche valider, droit rejeter) · 6 sommets · 7 transporter un développement · "
                      "8 feuille sous la souris · 9 toutes · 0 marquer un développement.")
