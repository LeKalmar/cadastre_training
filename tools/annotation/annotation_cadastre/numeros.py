"""Pointer les numéros de parcelles écrits sur le plan : vérité terrain de l'OCR des numéros.

La valeur du numéro est déjà connue (GénéFoncier : parcelles_no + parcelles_bis) ; il ne manque
que l'endroit où il est écrit. Outil « Pointer les numéros » (touche N) :
    clic gauche   sur le numéro écrit d'une parcelle -> point enregistré (valeur recopiée)
                  (re-cliquer une parcelle déjà pointée déplace son point)
    clic droit    efface le point le plus proche
    Échap         quitte l'outil
Le clic se fait DANS la parcelle (n'importe laquelle de ses typologies : un seul point par
parcelle principale). Couche « Numéros à pointer » : parcelles numérotées sans point, en jaune,
avec le numéro attendu en étiquette ; elle se vide à mesure. La liste « Feuille » de la barre
(ou la touche 8) restreint aux parcelles d'une feuille.
"""
import getpass
import uuid

from qgis.core import (NULL, QgsDataSourceUri, QgsFeature, QgsFeatureRequest, QgsFillSymbol, QgsGeometry,
                       QgsMarkerSymbol, QgsPalLayerSettings, QgsPointXY, QgsProject, QgsRectangle,
                       QgsSingleSymbolRenderer, QgsTextFormat, QgsVectorLayer, QgsVectorLayerSimpleLabeling)
from qgis.gui import QgsMapTool
from qgis.PyQt.QtCore import QDateTime, Qt
from qgis.PyQt.QtGui import QColor
from qgis.utils import iface

SERVICE = "cadastre_annotation"
POINTS = "Numéros pointés"
A_POINTER = "Numéros à pointer"
PARCELLES = "Parcelles validées"
GROUPE = "Numéros"
FILTREES = (POINTS, A_POINTER)


def _pg(table, nom, key, geom="geom"):
    uri = QgsDataSourceUri()
    uri.setParam("service", SERVICE)
    uri.setDataSource("public", table, geom, "", key)
    uri.setSrid("2154")
    return QgsVectorLayer(uri.uri(False), nom, "postgres")


def _etiquettes(lyr, couleur, taille=9, min_echelle=4000):
    pal = QgsPalLayerSettings()
    pal.fieldName = "numero"
    pal.scaleVisibility = True
    pal.maximumScale = 1
    pal.minimumScale = min_echelle
    fmt = QgsTextFormat()
    fmt.setColor(QColor(couleur))
    fmt.setSize(taille)
    buf = fmt.buffer()
    buf.setEnabled(True)
    buf.setColor(QColor("white"))
    fmt.setBuffer(buf)
    pal.setFormat(fmt)
    lyr.setLabeling(QgsVectorLayerSimpleLabeling(pal))
    lyr.setLabelsEnabled(True)


def couches(creer=True):
    P = QgsProject.instance()
    pts, a_p = P.mapLayersByName(POINTS), P.mapLayersByName(A_POINTER)
    if pts and a_p:
        return pts[0], a_p[0]
    if not creer:
        return None, None
    pts = _pg("numeros_positions", POINTS, "id")
    a_p = _pg("v_numeros_a_pointer", A_POINTER, "id")
    if not pts.isValid() or not a_p.isValid():
        iface.messageBar().pushWarning("Numéros", "Table numeros_positions inaccessible (annotation_parcelles/schema.sql).")
        return None, None
    pts.setRenderer(QgsSingleSymbolRenderer(QgsMarkerSymbol.createSimple(
        {"name": "cross2", "size": "3.2", "color": "#d6336c", "outline_color": "#d6336c", "outline_width": "0.6"})))
    _etiquettes(pts, "#d6336c")
    a_p.setReadOnly(True)
    a_p.setRenderer(QgsSingleSymbolRenderer(QgsFillSymbol.createSimple(
        {"style": "solid", "color": "255,212,59,45", "outline_color": "#f59f00", "outline_width": "0.5",
         "outline_width_unit": "MM"})))
    _etiquettes(a_p, "#e67700", 8, 2500)
    P.addMapLayer(a_p, False)
    P.addMapLayer(pts, False)
    root = P.layerTreeRoot()
    grp = root.findGroup(GROUPE) or root.insertGroup(0, GROUPE)
    grp.insertLayer(0, pts)
    grp.insertLayer(1, a_p)
    return pts, a_p


def assurer():
    if QgsProject.instance().mapLayersByName(POINTS):
        return
    if _pg("numeros_positions", POINTS, "id").isValid():
        couches()


def _parcelle_sous(pt, rayon):
    lyrs = QgsProject.instance().mapLayersByName(PARCELLES)
    if not lyrs:
        return None
    lyr = lyrs[0]
    g = QgsGeometry.fromPointXY(pt)
    rect = QgsRectangle(pt.x() - rayon, pt.y() - rayon, pt.x() + rayon, pt.y() + rayon)
    best = None
    for f in lyr.getFeatures(QgsFeatureRequest().setFilterRect(rect)):
        fg = f.geometry()
        if fg.contains(g) and (best is None or fg.area() < best[1]):
            best = (f, fg.area())
    return best[0] if best else None


class OutilNumeros(QgsMapTool):
    def __init__(self, canvas):
        super().__init__(canvas)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def activate(self):
        super().activate()
        couches()
        iface.messageBar().pushInfo("Numéros", "Cliquer sur le numéro écrit de chaque parcelle jaune. "
                                               "Clic droit : effacer un point. Échap : quitter.")

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            iface.actionPan().trigger()

    def canvasReleaseEvent(self, e):
        pts, a_p = couches()
        if pts is None:
            return
        pt = self.toMapCoordinates(e.pos())
        mupp = self.canvas().mapUnitsPerPixel()
        if e.button() == Qt.MouseButton.RightButton:
            return self._effacer(pts, a_p, pt, 8 * mupp)
        f = _parcelle_sous(pt, 2 * mupp)
        if f is None:
            iface.messageBar().pushWarning("Numéros", "Aucune parcelle validée sous le clic.", 2)
            return
        no = f["parcelles_no"]
        if no in (None, NULL) or not str(no).strip():
            iface.messageBar().pushWarning("Numéros", "Cette parcelle n'a pas de numéro dans GénéFoncier.", 2)
            return
        numero = str(no).strip() + (str(f["parcelles_bis"]).strip() if f["parcelles_bis"] not in (None, NULL) else "")
        main = f["par_main_id"] if f["par_main_id"] not in (None, NULL) else f["id"]
        fid = f["feuille_id"]
        deja = [g.id() for g in pts.getFeatures(QgsFeatureRequest().setFilterExpression(
            f"\"feuille_id\" = '{fid}' AND \"main_key\" = '{main}'"))]
        prov = pts.dataProvider()
        if deja:
            prov.deleteFeatures(deja)
        nf = QgsFeature(pts.fields())
        for k, v in (("id", str(uuid.uuid4())), ("feuille_id", str(fid)), ("parcelle_id", str(f["id"])),
                     ("main_key", str(main)), ("numero", numero), ("cree_le", QDateTime.currentDateTime()),
                     ("cree_par", getpass.getuser())):
            nf.setAttribute(k, v)
        nf.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(pt)))
        ok, _ = prov.addFeatures([nf])
        if not ok:
            iface.messageBar().pushCritical("Numéros", "Point refusé par la base : " + "; ".join(prov.errors()))
            return
        self._rafraichir(pts, a_p)
        reste = a_p.featureCount()
        iface.messageBar().pushSuccess("Numéros", f"n° {numero} {'déplacé' if deja else 'pointé'} — "
                                                  f"{reste} typologie(s) encore à pointer dans l'affichage.", 1)

    def _effacer(self, pts, a_p, pt, rayon):
        g = QgsGeometry.fromPointXY(pt)
        rect = QgsRectangle(pt.x() - rayon, pt.y() - rayon, pt.x() + rayon, pt.y() + rayon)
        cand = sorted(pts.getFeatures(QgsFeatureRequest().setFilterRect(rect)),
                      key=lambda f: f.geometry().distance(g))
        if not cand:
            return
        pts.dataProvider().deleteFeatures([cand[0].id()])
        self._rafraichir(pts, a_p)
        iface.messageBar().pushInfo("Numéros", f"Point n° {cand[0]['numero']} effacé.", 1)

    @staticmethod
    def _rafraichir(pts, a_p):
        for l in (pts, a_p):
            l.dataProvider().reloadData()
            l.triggerRepaint()


def action(tb, canvas, etat):
    """QAction cochable de l'outil (touche N), à ajouter à la barre."""
    from qgis.PyQt.QtWidgets import QAction
    outil = OutilNumeros(canvas)
    etat["outil_num"] = outil
    a = QAction("N Pointer les numéros", tb)
    a.setCheckable(True)
    outil.setAction(a)

    def basculer(on):
        if on:
            canvas.setMapTool(outil)
        elif canvas.mapTool() is outil:
            iface.actionPan().trigger()
    a.toggled.connect(basculer)
    return a
