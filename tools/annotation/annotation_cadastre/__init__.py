"""Extension QGIS « Annotation cadastrale » : validation rapide des propositions IA."""


def classFactory(iface):  # noqa: N802 (nom imposé par QGIS)
    from .plugin import AnnotationCadastrePlugin
    return AnnotationCadastrePlugin(iface)
