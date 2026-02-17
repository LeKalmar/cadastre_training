# Introduction
## Genèse du projet
Ce modèle a été conçu par Maxence Jeanjean en 2026 pour répondre au besoin de GénéFoncier, site web de généalogie foncière présentant à ses utilisateurs des cartes de l'ancien cadastre français. Le site GénéFoncier est lancé en janvier 2026.

## Enjeux de la généalogie foncière sur le web
Le problème principal causé par l'utilisation d'archives sur internet sont les droits d'auteur. Afin de pouvoir utiliser pleinement le contenu de ces archives, il nous faut diffuser leur contenu à partir de données créées par nos soins. Dans un second temps, il est possible de négocier avec les archives pour utiliser leurs données numérisées brutes.
# Comment entraîner le modèle
Exemple : python train.py --batch-size 12 --epochs 100 --pos-weight 8.0 --work-dir work_dirs/Jehan --resume work_dirs/Jehan/best_model.pth
- batch-size : à régler en fonction de la capacité de votre carte graphique
- epoch : nombre d'époques à traiter, mettez-en beaucoup si vous voulez faire autre chose en même temps et laisser tourner le modèle.
- pos-weight : à vérifier.
- work-dir : endroit où vous voulez enregistrer le modèle
- resume : si vous avez déjà entamé un train et que vous voulez le reprendre après l'avoir stoppé, mettez resume et choisissez le modèle que vous souhaitez continuer.

# Comment utiliser le modèle
## Préparer les données
- Préparez le fichier .tif que vous voulez utiliser sur QGIS
- Sauvegardez le fichier tif tel quel dans un dossier de votre choix
- Exportez depuis QGIS votre tif en PNG (exporter sous > image > {nom}.png)
- Sauvegardez le PNG dans le même dossier que votre tif pour plus de facilité
## Lancer le modèle
- Activez le venv
- Lancez le script inference_tiled.py exemple : (python inference_tiled.py --checkpoint work_dirs/Jehan/best_model.pth --input givenchy_b_0.png --output resultats/)
- - checkpoint = le modèle que vous voulez utiliser (ici Jehan)
- - input = {dossier/fichier.png}
- - output = {dossier/fichier_traité.png}
- Copiez les données géographiques de votre Tif directement dans le PNG traité par l'IA avec bridge.py, exemple : python bridge_tif.py --tif {dossier/fichier.tif} --png {dossier/fichier_traité.png} --out {dossier/fichier_georef.tif}

- Vous pouvez maintenant importer fichier_georef.tif directement dans QGIS et utiliser les lignes vectorisées pour les transformer en polygones
## Transformer les lignes obtenues en polygones

GRASS -> r.to.vector
v.generalize
d'abord algorithme SNAKE puis DOUGLAS (threshold 0.19)