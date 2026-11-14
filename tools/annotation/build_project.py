#!/usr/bin/python3.14
"""Génère le projet QGIS d'annotation (annotation_parcelles/annotation.qgz).

À lancer avec le Python de QGIS (PyQGIS) :
    QT_QPA_PLATFORM=offscreen /usr/bin/python3.14 tools/annotation/build_project.py

Le projet se connecte à la base par le service PostgreSQL « cadastre_annotation »
(aucun mot de passe enregistré). Il peut être régénéré à tout moment : les données
sont dans la base, pas dans le projet.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qgis.core import (  # noqa: E402
    Qgis, QgsAction, QgsApplication, QgsCoordinateReferenceSystem, QgsDataSourceUri,
    QgsDefaultValue, QgsEditorWidgetSetup, QgsExpressionContextUtils, QgsFieldConstraints,
    QgsFillSymbol, QgsLayerTreeGroup, QgsPalLayerSettings, QgsProject, QgsRasterLayer,
    QgsRelation, QgsRuleBasedRenderer, QgsSnappingConfig, QgsTextBufferSettings,
    QgsTextFormat, QgsVectorLayer, QgsVectorLayerSimpleLabeling, QgsTolerance,
)
from qgis.PyQt.QtGui import QColor  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "annotation_parcelles" / "annotation.qgz"
VENV_PY = ROOT / ".venv" / "bin" / "python"
TOOLS = ROOT / "tools" / "annotation"
LOGS = ROOT / "annotation_parcelles" / "logs"
SERVICE = "cadastre_annotation"

N_FEUILLES = "Feuilles"
N_PARCELLES = "Parcelles validées"
N_PENDING = "IA – à valider"
N_PENDING_IA = "IA – géométrie d'origine"
N_ZV = "Zones validées"
N_ZI = "Zones ignorées"
FILTERED = (N_PARCELLES, N_PENDING, N_PENDING_IA, N_ZV, N_ZI)

STATUT_REVUE = {"À valider": "pending", "Validée telle quelle": "approved",
                "Corrigée puis validée": "corrected", "Rejetée": "rejected"}
MOTIFS = {"Bruit / faux trait": "bruit", "Texte / numéro": "texte", "Hachures": "hachures",
          "Double trait (fossé, chemin)": "double_trait", "Marge / cadre": "marge_cadre",
          "Hors feuille": "hors_feuille", "Doublon": "doublon",
          "Parcelle coupée en trop (sur-segmentation)": "sur_segmentation",
          "Parcelles fusionnées (sous-segmentation)": "sous_segmentation",
          "Route, chemin, rivière, canal…": "voirie_eau", "Autre": "autre"}
FEUILLE_STATUTS = {"À vérifier": "a_verifier", "En cours": "en_cours", "Validée": "validee",
                   "Exclue": "exclue"}
PARCELLE_STATUTS = {"Valide": "valide", "À vérifier": "a_verifier", "Rejetée": "rejete"}
LIENS = {"OK": "ok", "À vérifier": "a_verifier", "Hors feuille": "hors_feuille"}
ORIGINS = {"Export GénéFoncier": "export", "IA (validée)": "ai",
           "Découpe d'une proposition IA": "decoupe", "Fusion de propositions IA": "fusion",
           "Dessinée dans QGIS": "qgis",
           "Dessinée à la main": "manual"}
SPLITS = {"Entraînement": "train", "Validation": "val"}
MOTIFS_IGN = {"Hachures": "hachures", "Texte": "texte", "Cartouche": "cartouche",
              "Illisible": "illisible", "Autre": "autre"}


# ----------------------------------------------------------------------------- couches
def pg_layer(table, name, geom="geom", key="id", geom_type=None):
    uri = QgsDataSourceUri()
    uri.setParam("service", SERVICE)
    uri.setDataSource("public", table, geom, "", key)
    uri.setSrid("2154") if geom else None
    if geom_type:
        uri.setWkbType(geom_type)
    lyr = QgsVectorLayer(uri.uri(False), name, "postgres")
    if not lyr.isValid():
        sys.exit(f"Couche invalide : {table} ({lyr.error().summary()})")
    return lyr


def outline(color, width=0.5, style="solid", fill=None):
    props = {"style": "no" if fill is None else "solid", "outline_color": color,
             "outline_width": str(width), "outline_width_unit": "MM", "outline_style": style}
    if fill is not None:
        props["color"] = fill
    return QgsFillSymbol.createSimple(props)


def rules(layer, items):
    root = QgsRuleBasedRenderer.Rule(None)
    for label, expr, sym in items:
        root.appendChild(QgsRuleBasedRenderer.Rule(sym, 0, 0, expr, label))
    layer.setRenderer(QgsRuleBasedRenderer(root))


def labels(layer, expr, size=9, max_scale=None, color="#202020"):
    s = QgsPalLayerSettings()
    s.fieldName = expr
    s.isExpression = True
    fmt = QgsTextFormat()
    fmt.setSize(size)
    fmt.setColor(QColor(color))
    buf = QgsTextBufferSettings()
    buf.setEnabled(True)
    buf.setSize(0.8)
    fmt.setBuffer(buf)
    s.setFormat(fmt)
    if max_scale:
        s.scaleVisibility = True
        s.maximumScale = 1
        s.minimumScale = max_scale
    layer.setLabeling(QgsVectorLayerSimpleLabeling(s))
    layer.setLabelsEnabled(True)


def widget(layer, field, kind, config=None):
    i = layer.fields().indexOf(field)
    if i >= 0:
        layer.setEditorWidgetSetup(i, QgsEditorWidgetSetup(kind, config or {}))


def value_map(layer, field, mapping):
    widget(layer, field, "ValueMap", {"map": [{k: v} for k, v in mapping.items()]})


def read_only(layer, *fields):
    cfg = layer.editFormConfig()
    for f in fields:
        i = layer.fields().indexOf(f)
        if i >= 0:
            cfg.setReadOnly(i, True)
    layer.setEditFormConfig(cfg)


def hidden(layer, *fields):
    for f in fields:
        widget(layer, f, "Hidden")


def default(layer, field, expr):
    i = layer.fields().indexOf(field)
    if i >= 0:
        layer.setDefaultValueDefinition(i, QgsDefaultValue(expr, False))


def feuille_widget(layer, feuilles):
    widget(layer, "feuille_id", "ValueRelation", {
        "Layer": feuilles.id(), "LayerName": feuilles.name(), "LayerSource": feuilles.source(),
        "LayerProviderName": "postgres", "Key": "id", "Value": "nom", "OrderByValue": True,
        "AllowNull": False, "AllowMulti": False, "UseCompleter": True,
    })
    default(layer, "feuille_id", f"coalesce(nullif(@feuille_active, ''), "
                                 f"array_first(overlay_within('{feuilles.id()}', \"id\")))")
    i = layer.fields().indexOf("feuille_id")
    layer.setConstraintExpression(
        i, f"\"feuille_id\" IS NULL OR within(point_on_surface($geometry), "
           f"geometry(get_feature('{feuilles.id()}', 'id', \"feuille_id\")))",
        "La géométrie sort de l'emprise de sa feuille : vérifier la feuille choisie.")
    layer.setFieldConstraint(i, QgsFieldConstraints.Constraint.ConstraintExpression,
                             QgsFieldConstraints.ConstraintStrength.ConstraintStrengthSoft)


# ----------------------------------------------------------------------------- actions
HELPERS = r'''
from qgis.core import QgsProject, QgsFeatureRequest
from qgis.utils import iface
P = QgsProject.instance()
def _refresh():
    for n in ("%(parcelles)s", "%(pending)s", "Avancement"):
        for l in P.mapLayersByName(n):
            l.triggerRepaint()
def _set(lyr, ids, values):
    attrs = {lyr.fields().indexOf(k): v for k, v in values.items()}
    if lyr.isEditable():
        for i in ids:
            for k, v in attrs.items():
                lyr.changeAttributeValue(i, k, v)
    else:
        lyr.dataProvider().changeAttributeValues({i: attrs for i in ids})
        lyr.reload()
    _refresh()
def _suivante(lyr):
    req = QgsFeatureRequest().setFilterExpression("review_status = 'pending'")
    req.addOrderBy("aire_m2", False)
    req.setLimit(1)
    f = next(lyr.getFeatures(req), None)
    if f is None:
        lyr.removeSelection()
        iface.messageBar().pushSuccess("Annotation", "Plus aucune proposition à valider ici.")
        return
    lyr.selectByIds([f.id()])
    iface.mapCanvas().setExtent(f.geometry().boundingBox().buffered(
        max(15.0, f.geometry().boundingBox().width())))
    iface.mapCanvas().refresh()
''' % {"parcelles": N_PARCELLES, "pending": N_PENDING}


def decision_code(values, scope):
    ids = "[[% $id %]]" if scope == "feature" else "list(lyr.selectedFeatureIds())"
    return HELPERS + f'''
lyr = P.mapLayer('[% @layer_id %]')
ids = {ids}
if ids:
    _set(lyr, ids, {values!r})
_suivante(lyr)
'''


def add_action(layer, name, code, scopes, short=None):
    act = QgsAction(QgsAction.ActionType.GenericPython, name, code, "", False,
                    short or name, set(scopes))
    layer.actions().addAction(act)
    return act


def subprocess_code(args_expr, label):
    clean_env = ("env = {k: v for k, v in os.environ.items() "
                 "if not k.startswith(('PYTHON', 'GDAL_', 'PROJ_', 'QGIS_', 'LD_PRELOAD'))}")
    return f'''
import os, subprocess, datetime
from qgis.utils import iface
logs = r"{LOGS}"
os.makedirs(logs, exist_ok=True)
log = os.path.join(logs, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_{label}.log")
{clean_env}
args = {args_expr}
subprocess.Popen([r"{VENV_PY}"] + args, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                 cwd=r"{ROOT}", env=env, start_new_session=True)
iface.messageBar().pushInfo("Annotation", "Tâche lancée en arrière-plan. Journal : " + log)
'''


ACTIVER = r'''
from qgis.core import QgsProject, QgsExpressionContextUtils
from qgis.utils import iface
P = QgsProject.instance()
fid = '[% "id" %]'
QgsExpressionContextUtils.setProjectVariable(P, 'feuille_active', fid)
for name in __FILTERED__:
    for lyr in P.mapLayersByName(name):
        if lyr.isEditable():
            iface.messageBar().pushWarning("Annotation", "« %s » est en édition : enregistrez puis recommencez." % name)
            continue
        lyr.setSubsetString("feuille_id = '%s'" % fid)
raster = r"""[% "raster" %]"""
grp = P.layerTreeRoot().findGroup("Scans")
if grp:
    for node in grp.findLayers():
        node.setItemVisibilityChecked(node.layer().source() == raster)
lyr = P.mapLayer('[% @layer_id %]')
f = lyr.getFeature([% $id %])
iface.mapCanvas().setExtent(f.geometry().boundingBox().buffered(20))
iface.mapCanvas().refresh()
iface.messageBar().pushSuccess("Annotation", "Feuille active : [% "nom" %]")
'''.replace("__FILTERED__", repr(FILTERED))

TOUT = r'''
from qgis.core import QgsProject, QgsExpressionContextUtils
from qgis.utils import iface
P = QgsProject.instance()
QgsExpressionContextUtils.setProjectVariable(P, 'feuille_active', '')
for name in __FILTERED__:
    for lyr in P.mapLayersByName(name):
        if not lyr.isEditable():
            lyr.setSubsetString("")
iface.messageBar().pushInfo("Annotation", "Aucune feuille active : tout est affiché.")
'''.replace("__FILTERED__", repr(FILTERED))


# ----------------------------------------------------------------------------- projet
def main():
    QgsApplication.setPrefixPath("/usr", True)
    qgs = QgsApplication([], False)
    qgs.initQgis()
    P = QgsProject.instance()
    P.clear()
    P.setCrs(QgsCoordinateReferenceSystem("EPSG:2154"))
    P.setTitle("Annotation cadastrale – vérité terrain du modèle")
    QgsExpressionContextUtils.setProjectVariable(P, "feuille_active", "")
    root = P.layerTreeRoot()

    poly = Qgis.WkbType.MultiPolygon
    feuilles = pg_layer("feuilles", N_FEUILLES)
    parcelles = pg_layer("parcelles", N_PARCELLES)
    pending = pg_layer("parcelles_pending", N_PENDING, geom="geom", geom_type=poly)
    pending_ia = pg_layer("parcelles_pending", N_PENDING_IA, geom="geom_ia", geom_type=poly)
    zv = pg_layer("zones_validees", N_ZV)
    zi = pg_layer("zones_ignorees", N_ZI)
    avancement = pg_layer("v_avancement", "Avancement")
    hors = pg_layer("v_parcelles_hors_feuille", "Parcelles hors de leur feuille")
    a_entrainer = pg_layer("v_a_entrainer", "Feuilles à (ré)entraîner")
    tables = [pg_layer(t, n, geom=None, key=k) for t, n, k in (
        ("v_qualite_ia", "Qualité de l'IA par extraction", "extraction_id"),
        ("v_rejets_par_modele", "Rejets par modèle", "id"),
        ("extractions", "Extractions", "id"), ("modeles", "Modèles", "id"),
        ("datasets", "Jeux d'entraînement", "id"))]
    for t in tables[:2]:
        t.setReadOnly(True)
    for lyr in (avancement, hors, a_entrainer, pending_ia):
        lyr.setReadOnly(True)

    # --- styles
    rules(pending, [
        ("À valider", "review_status = 'pending'", outline("#e0007a", 0.7, fill="255,0,140,30")),
        ("Validée telle quelle", "review_status = 'approved'", outline("#1a9641", 0.4)),
        ("Corrigée puis validée", "review_status = 'corrected'", outline("#2b83ba", 0.5)),
        ("Rejetée", "review_status = 'rejected'", outline("#9a9a9a", 0.3, "dash")),
        # développements : dessinés par-dessus (l'extension ajoute les mêmes règles à l'ouverture)
        ("Développement – parcelle", "\"dev_role\" = 'parcelle'", outline("#7a1fd1", 0.6, fill="150,60,220,35")),
        ("Développement – emplacement réel", "\"dev_role\" = 'emplacement'",
         outline("#7a1fd1", 1.2, "dash", fill="150,60,220,15")),
        ("Développement – contexte", "\"dev_role\" = 'contexte'", outline("#6f6880", 0.6, "dot", fill="120,110,140,35")),
    ])
    rules(pending_ia, [("Proposition d'origine", "geom_modifiee", outline("#ff8c00", 0.4, "dash"))])
    rules(parcelles, [
        ("Valide", "statut = 'valide'", outline("#006d2c", 0.4)),
        ("À vérifier (lien douteux)", "statut = 'a_verifier' AND lien = 'a_verifier'",
         outline("#ff7f00", 0.8)),
        ("À vérifier", "statut = 'a_verifier' AND lien <> 'a_verifier'", outline("#ff7f00", 0.8, "dash")),
        ("Rejetée", "statut = 'rejete'", outline("#bbbbbb", 0.3, "dot")),
    ])
    rules(feuilles, [
        ("À vérifier", "statut = 'a_verifier'", outline("#3060c0", 1.0)),
        ("En cours", "statut = 'en_cours'", outline("#f0a000", 1.4)),
        ("Validée", "statut = 'validee'", outline("#1a9641", 1.4)),
        ("Exclue", "statut = 'exclue'", outline("#999999", 0.6, "dash")),
    ])
    rules(zv, [("Zone validée", "true", outline("#1a9641", 0.3, fill="26,150,65,40"))])
    rules(zi, [("Zone ignorée", "true", outline("#d7191c", 0.3, fill="215,25,28,50"))])
    rules(hors, [("Hors feuille", "true", outline("#d7191c", 1.2))])
    rules(a_entrainer, [("Nouveau pour l'entraînement", "true", outline("#7b3294", 1.6, "dash"))])
    rules(avancement, [("Feuille", "true", outline("#3060c0", 0.2))])
    labels(feuilles, "\"nom\" || '\\n' || \"statut\" || coalesce(' · ' || \"split\", '')", 11)
    labels(avancement, "\"nom\" || '\\nvalides ' || \"parcelles_valides\" || ' · IA à valider ' "
                       "|| \"ia_a_valider\" || ' · liens à vérifier ' || \"liens_a_verifier\"", 9)
    labels(parcelles, "coalesce(\"parcelles_no\", '') || coalesce(\"parcelles_bis\", '')", 8,
           max_scale=2500)

    # --- formulaires
    value_map(pending, "review_status", STATUT_REVUE)
    value_map(pending, "motif_rejet", MOTIFS)
    for lyr_ in (pending, parcelles):
        value_map(lyr_, "dev_role", {"Parcelle d'un développement": "parcelle",
                                     "Emplacement réel (fausse parcelle)": "emplacement",
                                     "Parcelle de contexte": "contexte"})
    widget(pending, "typology_type", "UniqueValues", {"Editable": True})
    feuille_widget(pending, feuilles)
    value_map(pending, "origin", {k: v for k, v in ORIGINS.items() if v != "export"})
    read_only(pending, "feuille_id", "extraction_id", "origin", "decoupe_de", "fusion_de", "origine_grass", "aire_m2", "developpement",
              "geom_modifiee", "reviewed_by", "reviewed_at", "created_at")
    hidden(pending, "id")

    feuille_widget(parcelles, feuilles)
    value_map(parcelles, "origin", ORIGINS)
    value_map(parcelles, "statut", PARCELLE_STATUTS)
    value_map(parcelles, "lien", LIENS)
    widget(parcelles, "type", "UniqueValues", {"Editable": True})
    default(parcelles, "origin", "'qgis'")
    default(parcelles, "statut", "'valide'")
    default(parcelles, "lien", "'ok'")
    read_only(parcelles, "contraste", "pending_id", "modifie_le", "modifie_par", "developpement")
    hidden(parcelles, "id")

    value_map(feuilles, "statut", FEUILLE_STATUTS)
    value_map(feuilles, "split", SPLITS)
    value_map(feuilles, "source", {"Export GénéFoncier": "export", "Scan (emprise)": "scan",
                                   "IA": "ia", "Manuelle": "manuel"})
    read_only(feuilles, "source", "validee_le", "modifie_le", "split")
    hidden(feuilles, "id")

    for z in (zv, zi):
        feuille_widget(z, feuilles)
        hidden(z, "id")
    value_map(zi, "motif", MOTIFS_IGN)

    # --- actions : validation des propositions IA
    valider = add_action(pending, "✔ Valider puis suivante",
                         decision_code({"review_status": "approved"}, "feature"),
                         {"Feature", "Canvas"}, "Valider")
    for label, motif in MOTIFS.items():
        add_action(pending, f"✘ Rejeter – {label}",
                   decision_code({"review_status": "rejected", "motif_rejet": motif}, "feature"),
                   {"Feature", "Canvas"}, f"Rejeter ({motif})")
    add_action(pending, "↺ Remettre à valider",
               decision_code({"review_status": "pending"}, "feature"), {"Feature", "Canvas"})
    add_action(pending, "✔ Valider la sélection",
               decision_code({"review_status": "approved"}, "layer"), {"Layer"})
    add_action(pending, "✘ Rejeter la sélection (bruit)",
               decision_code({"review_status": "rejected", "motif_rejet": "bruit"}, "layer"), {"Layer"})
    add_action(pending, "➜ Aller à la prochaine proposition",
               HELPERS + "\n_suivante(P.mapLayer('[% @layer_id %]'))\n", {"Layer"})
    pending.actions().setDefaultAction("Canvas", valider.id())

    # --- actions : feuilles
    activer = add_action(feuilles, "Travailler sur cette feuille", ACTIVER, {"Feature", "Canvas"})
    add_action(feuilles, "Tout afficher (aucune feuille active)", TOUT, {"Feature", "Canvas", "Layer"})
    add_action(feuilles, "Lancer l'IA sur cette feuille (GPU)",
               subprocess_code(f"[r'{TOOLS / 'import_ia.py'}', '--feuille-id', '[% \"id\" %]', "
                               "'--remplacer']", "ia"),
               {"Feature", "Canvas"})
    add_action(feuilles, "Exporter un jeu d'entraînement", r'''
from qgis.PyQt.QtWidgets import QInputDialog
nom, ok = QInputDialog.getText(None, "Export", "Nom du jeu (ex. v3) :")
if ok and nom.strip():
''' + "\n".join("    " + line for line in subprocess_code(
        f"[r'{TOOLS / 'export_dataset.py'}', '--nom', nom.strip()]", "export").splitlines()),
               {"Layer"})
    feuilles.actions().setDefaultAction("Canvas", activer.id())

    # --- arbre des couches
    def group(name, layers, checked=True):
        g = root.addGroup(name)
        for lyr in layers:
            P.addMapLayer(lyr, False)
            g.addLayer(lyr)
        g.setItemVisibilityChecked(checked)
        return g

    group("Validation IA", [pending, pending_ia])
    root.findLayer(pending_ia.id()).setItemVisibilityChecked(False)
    group("Vérité terrain", [parcelles, hors, zv, zi])
    group("Feuilles", [feuilles, avancement, a_entrainer])
    root.findLayer(avancement.id()).setItemVisibilityChecked(False)
    root.findLayer(a_entrainer.id()).setItemVisibilityChecked(False)
    group("Suivi (tables)", tables)

    # --- relations (formulaire d'une feuille -> ses parcelles, propositions, zones)
    for child, rid, name in ((parcelles, "rel_parcelles", "Parcelles"),
                             (pending, "rel_pending", "Propositions IA"),
                             (zv, "rel_zv", "Zones validées"), (zi, "rel_zi", "Zones ignorées")):
        rel = QgsRelation()
        rel.setId(rid)
        rel.setName(name)
        rel.setReferencingLayer(child.id())
        rel.setReferencedLayer(feuilles.id())
        rel.addFieldPair("feuille_id", "id")
        if not rel.isValid():
            sys.exit(f"Relation invalide : {name} ({rel.validationError()})")
        P.relationManager().addRelation(rel)

    # scans : un par feuille, éteints (l'action « Travailler sur cette feuille » allume le bon)
    scans = root.addGroup("Scans")
    seen = set()
    for f in feuilles.getFeatures():
        path = f["raster"]
        if not path or path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        r = QgsRasterLayer(path, f["nom"])
        if r.isValid():
            P.addMapLayer(r, False)
            scans.addLayer(r).setItemVisibilityChecked(False)
    osm = QgsRasterLayer("type=xyz&url=https://tile.openstreetmap.org/{z}/{x}/{y}.png&zmax=19",
                         "OpenStreetMap", "wms")
    if osm.isValid():
        P.addMapLayer(osm, False)
        root.addGroup("Fonds").addLayer(osm).setItemVisibilityChecked(False)

    # --- accrochage + édition topologique
    cfg = P.snappingConfig()
    cfg.setEnabled(True)
    cfg.setMode(Qgis.SnappingMode.AdvancedConfiguration)
    for lyr in (parcelles, pending, zv, zi):
        cfg.setIndividualLayerSettings(lyr, QgsSnappingConfig.IndividualLayerSettings(
            True, Qgis.SnappingTypes(Qgis.SnappingType.Vertex | Qgis.SnappingType.Segment),
            12, QgsTolerance.UnitType.Pixels, 0.0, 0.0))
    cfg.setIntersectionSnapping(True)
    for lyr in (parcelles, pending, zv, zi):    # corrections automatiques à la saisie
        go = lyr.geometryOptions()
        go.setRemoveDuplicateNodes(True)       # sommet accroché sur un autre -> pas de doublon
        go.setGeometryPrecision(0.01)          # 1 cm : sommets « presque » confondus fusionnés
    P.setSnappingConfig(cfg)
    P.setTopologicalEditing(True)
    P.setAvoidIntersectionsMode(Qgis.AvoidIntersectionsMode.AllowIntersections)

    if feuilles.featureCount():
        ext = feuilles.extent()
        ext.scale(1.05)
        P.viewSettings().setDefaultViewExtent(
            __import__("qgis.core", fromlist=["QgsReferencedRectangle"]).QgsReferencedRectangle(
                ext, P.crs()))
    P.writeEntry("Paths", "/Absolute", False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not P.write(str(OUT)):
        sys.exit("Échec de l'écriture du projet.")
    print(f"Projet écrit : {OUT} ({len(seen)} scans, {feuilles.featureCount()} feuilles)")
    qgs.exitQgis()


if __name__ == "__main__":
    main()
