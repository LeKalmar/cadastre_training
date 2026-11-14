"""Point d'entrée QGIS : installe la barre « Validation annotation » et ses touches."""
from . import validation


class AnnotationCadastrePlugin:
    def __init__(self, iface):
        self.iface = iface

    def initGui(self):  # noqa: N802
        validation.installer()

    def unload(self):
        validation.desinstaller()
