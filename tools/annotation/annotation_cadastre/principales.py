"""Valider les parcelles principales proposées par l'IA (main / typology GénéFoncier), d'un coup.

tools/typologie/proposer.py type chaque proposition IA en attente (bâti, cour, eau…) et les
regroupe en parcelles principales (mains_pending). Ici on valide PAR PARCELLE PRINCIPALE :
contour épais = la parcelle principale, remplissage = le type de chaque typologie.

Touches (carte active) :
    P   parcelle principale suivante à valider (zoom dessus)
    M   valider la parcelle principale en cours (ou celle sélectionnée dans « Parcelles principales –
        à valider ») : elle et toutes ses typologies sont validées (parcelles_main + parcelles)
    G   regrouper la sélection de « IA – à valider » dans une même parcelle principale
    U   détacher la sélection de « IA – à valider » : elle forme une nouvelle parcelle principale
    T   changer le type de la sélection de « IA – à valider » (menu, touche 1-6)
Une typologie fausse se rejette comme avant (touche 2) : elle sort de sa parcelle principale.
Le numéro d'une parcelle principale se saisit dans le champ « parcelles_no » de sa couche.
"""
from qgis.core import (QgsDataSourceUri, QgsFeatureRequest, QgsFillSymbol, QgsPalLayerSettings, QgsProject,
                       QgsProviderRegistry, QgsRuleBasedRenderer, QgsTextFormat, QgsVectorLayer,
                       QgsVectorLayerSimpleLabeling)
from qgis.PyQt.QtGui import QColor, QCursor
from qgis.PyQt.QtWidgets import QMenu
from qgis.utils import iface

SERVICE = "cadastre_annotation"
MAINS = "Parcelles principales – à valider"
TYPES_C = "Typologies proposées"
PENDING = "IA – à valider"
GROUPE = "Parcelles principales"
FILTREES = (MAINS, TYPES_C)
TYPES = [("Bâti", "batiment"), ("Bâti public", "bat-public"), ("Cour", "cour"), ("Eau", "eau"),
         ("Jardin / végétation", "jardin"), ("Sans type visible", None)]
COULEURS = {"batiment": "230,70,70,110", "bat-public": "150,20,60,130", "cour": "90,170,230,70",
            "eau": "30,90,220,120", "jardin": "60,170,60,100"}
_etat = {"main": None}


def _sql(requete):
    md = QgsProviderRegistry.instance().providerMetadata("postgres")
    con = md.createConnection(f"service='{SERVICE}'", {})
    return con.executeSql(requete)


def _pg(table, nom, filtre=""):
    uri = QgsDataSourceUri()
    uri.setParam("service", SERVICE)
    uri.setDataSource("public", table, "geom", filtre, "id")
    uri.setSrid("2154")
    return QgsVectorLayer(uri.uri(False), nom, "postgres")


def _sym(**kw):
    base = {"style": "no", "color": "0,0,0,0", "outline_style": "no"}
    base.update(kw)
    return QgsFillSymbol.createSimple(base)


def couches(creer=True):
    P = QgsProject.instance()
    m, t = P.mapLayersByName(MAINS), P.mapLayersByName(TYPES_C)
    if m and t:
        return m[0], t[0]
    if not creer:
        return None, None
    mains = _pg("mains_pending", MAINS, "review_status = 'pending'")
    typ = _pg("parcelles_pending", TYPES_C, "review_status = 'pending' AND main_pending IS NOT NULL")
    if not mains.isValid() or not typ.isValid():
        iface.messageBar().pushWarning("Principales", "Tables mains_pending inaccessibles (annotation_parcelles/schema.sql).")
        return None, None
    root_r = QgsRuleBasedRenderer.Rule(None)
    root_r.appendChild(QgsRuleBasedRenderer.Rule(_sym(outline_style="solid", outline_color="#212529",
                                                      outline_width="1.1", outline_width_unit="MM"),
                                                 0, 0, "true", "Parcelle principale"))
    mains.setRenderer(QgsRuleBasedRenderer(root_r))
    pal = QgsPalLayerSettings()
    pal.fieldName = "coalesce(parcelles_no, '')"
    pal.isExpression = True
    fmt = QgsTextFormat()
    fmt.setColor(QColor("#212529"))
    fmt.setSize(9)
    pal.setFormat(fmt)
    mains.setLabeling(QgsVectorLayerSimpleLabeling(pal))
    mains.setLabelsEnabled(True)
    root_t = QgsRuleBasedRenderer.Rule(None)
    for lib, t in TYPES[:-1]:
        root_t.appendChild(QgsRuleBasedRenderer.Rule(_sym(style="solid", color=COULEURS[t]), 0, 0,
                                                     f"\"typology_type\" = '{t}'", lib))
    root_t.appendChild(QgsRuleBasedRenderer.Rule(_sym(style="solid", color="200,200,200,40"), 0, 0,
                                                 "\"typology_type\" IS NULL", "Sans type visible"))
    root_t.appendChild(QgsRuleBasedRenderer.Rule(_sym(outline_style="dash", outline_color="#f08c00",
                                                      outline_width="0.6", outline_width_unit="MM"), 0, 0,
                                                 "\"review_note\" LIKE '%à découper%'", "Typologie mixte (à découper ?)"))
    typ.setRenderer(QgsRuleBasedRenderer(root_t))
    typ.setReadOnly(True)
    P.addMapLayer(typ, False)
    P.addMapLayer(mains, False)
    root = P.layerTreeRoot()
    grp = root.findGroup(GROUPE) or root.insertGroup(0, GROUPE)
    grp.insertLayer(0, mains)
    grp.insertLayer(1, typ)
    return mains, typ


def assurer():
    if QgsProject.instance().mapLayersByName(MAINS):
        return
    if _pg("mains_pending", MAINS).isValid():
        couches()


def _rafraichir():
    for n in (MAINS, TYPES_C, PENDING, "Parcelles validées", "Avancement"):
        for l in QgsProject.instance().mapLayersByName(n):
            l.dataProvider().reloadData()
            l.triggerRepaint()


def _a_valider(mains):
    return sorted(mains.getFeatures(), key=lambda f: (str(f["feuille_id"]), -f.geometry().centroid().asPoint().y(),
                                                      f.geometry().centroid().asPoint().x()))


def _ouvrir(f):
    _etat["main"] = str(f["id"])
    iface.mapCanvas().setExtent(f.geometry().boundingBox().buffered(max(8.0, f.geometry().boundingBox().width() * 0.6)))
    iface.mapCanvas().refresh()
    mains, _ = couches()
    mains.selectByIds([f.id()])
    n = len(list(mains.getFeatures()))
    conf = f["confiance"]
    conf_txt = f" — confiance du regroupement {conf:.2f}" if conf else ""
    iface.messageBar().pushInfo("Parcelle principale",
                                f"n° {f['parcelles_no'] or '?'}{conf_txt} — M valider · G regrouper · "
                                f"U détacher · T type · P suivante ({n} à valider).")


def suivante():
    mains, _ = couches()
    if mains is None:
        return
    todo = _a_valider(mains)
    if not todo:
        iface.messageBar().pushSuccess("Principales", "Plus aucune parcelle principale à valider dans l'affichage.")
        return
    ids = [str(f["id"]) for f in todo]
    i = ids.index(_etat["main"]) + 1 if _etat["main"] in ids else 0
    _ouvrir(todo[i % len(todo)])


def valider():
    mains, _ = couches()
    if mains is None:
        return
    sel = [str(f["id"]) for f in mains.getSelectedFeatures()] or ([_etat["main"]] if _etat["main"] else [])
    if not sel:
        iface.messageBar().pushInfo("Principales", "Aucune parcelle principale ouverte : touche P.")
        return
    pend = QgsProject.instance().mapLayersByName(PENDING)
    if pend and pend[0].isEditable() and pend[0].isModified():
        iface.messageBar().pushWarning("Principales", "Enregistrer d'abord les retouches de « IA – à valider ».")
        return
    for mid in sel:
        _sql(f"SELECT valider_main('{mid}'::uuid)")
    _rafraichir()
    iface.messageBar().pushSuccess("Principales", f"{len(sel)} parcelle(s) principale(s) validée(s), typologies comprises.", 2)
    if len(sel) == 1:
        suivante()


def _selection_pending():
    pend = QgsProject.instance().mapLayersByName(PENDING)
    if not pend:
        return []
    return [str(f["id"]) for f in pend[0].getSelectedFeatures()]


def _tableau(ids):
    return "ARRAY[" + ",".join(f"'{i}'::uuid" for i in ids) + "]"


def regrouper():
    ids = _selection_pending()
    if len(ids) < 2:
        iface.messageBar().pushInfo("Principales", f"Sélectionner au moins 2 polygones dans « {PENDING} », puis G.")
        return
    _sql(f"SELECT regrouper_pending({_tableau(ids)})")
    _rafraichir()
    iface.messageBar().pushSuccess("Principales", f"{len(ids)} typologies réunies dans une parcelle principale.", 2)


def detacher():
    ids = _selection_pending()
    if not ids:
        iface.messageBar().pushInfo("Principales", f"Sélectionner des polygones dans « {PENDING} », puis U.")
        return
    _sql(f"SELECT detacher_pending({_tableau(ids)})")
    _rafraichir()
    iface.messageBar().pushSuccess("Principales", f"{len(ids)} typologie(s) détachée(s) : nouvelle parcelle principale.", 2)


def changer_type():
    ids = _selection_pending()
    if not ids:
        iface.messageBar().pushInfo("Principales", f"Sélectionner des polygones dans « {PENDING} », puis T.")
        return
    menu = QMenu(iface.mapCanvas())
    for k, (lib, t) in enumerate(TYPES, 1):
        act = menu.addAction(f"&{k}  {lib}")
        act.setData(t)
    choix = menu.exec(QCursor.pos())
    if choix is None:
        return
    t = choix.data()
    val = "NULL" if t is None else f"'{t}'"
    _sql(f"UPDATE parcelles_pending SET typology_type = {val} WHERE id = ANY({_tableau(ids)}) AND review_status = 'pending'")
    _rafraichir()
    iface.messageBar().pushSuccess("Principales", f"{len(ids)} typologie(s) → {choix.text()[4:]}.", 2)


def commandes():
    return [
        ("main_suivante", "P Parcelle principale suivante", "P", suivante),
        ("main_valider", "M Valider la parcelle principale", "M", valider),
        ("main_regrouper", "G Regrouper la sélection", "G", regrouper),
        ("main_detacher", "U Détacher la sélection", "U", detacher),
        ("main_type", "T Changer le type…", "T", changer_type),
    ]
