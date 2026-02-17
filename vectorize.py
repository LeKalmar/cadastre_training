#!/usr/bin/env python3
"""
Script de vectorisation pour cadastres napoléoniennes
Convertit les masques binaires en polygones vectoriels (GeoJSON)
Compatible avec coordonnées géoréférencées (si métadonnées disponibles)
"""

import argparse
from pathlib import Path
import json

import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


# ============================================================================
# Vectorisation
# ============================================================================

def mask_to_polygons(mask_path, min_area=100, simplify_epsilon=2.0):
    """
    Convertit un masque binaire en polygones
    
    Args:
        mask_path: Chemin vers le masque PNG
        min_area: Surface minimale pour garder un polygone (pixels²)
        simplify_epsilon: Facteur de simplification Douglas-Peucker
    
    Returns:
        List de polygones au format [[x1,y1], [x2,y2], ...]
    """
    # Charger masque
    mask = np.array(Image.open(mask_path).convert('L'))
    
    # Binariser (au cas où)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Trouver contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    
    for contour in contours:
        # Filtrer par surface
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Simplifier contour (Douglas-Peucker)
        epsilon = simplify_epsilon * cv2.arcLength(contour, True) / 100.0
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convertir en liste de coordonnées
        polygon = approx.squeeze().tolist()
        
        # Vérifier que c'est bien un polygone (>= 3 points)
        if len(polygon) >= 3:
            # Si c'est un seul point [x, y], le wrapper en liste
            if isinstance(polygon[0], int):
                polygon = [polygon]
            
            polygons.append(polygon)
    
    return polygons


def polygons_to_geojson(polygons, properties=None):
    """
    Convertit une liste de polygones en GeoJSON
    
    Args:
        polygons: Liste de polygones [[x1,y1], [x2,y2], ...]
        properties: Dictionnaire de propriétés optionnelles
    
    Returns:
        Dict GeoJSON Feature
    """
    features = []
    
    for i, polygon in enumerate(polygons):
        # Fermer le polygone si nécessaire
        if polygon[0] != polygon[-1]:
            polygon.append(polygon[0])
        
        # Créer feature GeoJSON
        feature = {
            "type": "Feature",
            "properties": {
                "id": i,
                **(properties or {})
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon]
            }
        }
        
        features.append(feature)
    
    # GeoJSON FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    return geojson


def pixel_to_geo_coords(polygons, transform_params):
    """
    Convertit coordonnées pixel en coordonnées géographiques
    
    Args:
        polygons: Liste de polygones en coordonnées pixel
        transform_params: Dict avec {'x_origin', 'y_origin', 'pixel_width', 'pixel_height', 'crs'}
    
    Returns:
        Liste de polygons en coordonnées géographiques
    """
    x_origin = transform_params['x_origin']
    y_origin = transform_params['y_origin']
    pixel_width = transform_params['pixel_width']
    pixel_height = transform_params['pixel_height']
    
    geo_polygons = []
    
    for polygon in polygons:
        geo_polygon = []
        for x_pixel, y_pixel in polygon:
            # Transformer pixel -> géo
            x_geo = x_origin + (x_pixel * pixel_width)
            y_geo = y_origin + (y_pixel * pixel_height)
            geo_polygon.append([x_geo, y_geo])
        
        geo_polygons.append(geo_polygon)
    
    return geo_polygons


def vectorize_directory(mask_dir, output_dir, min_area=100, simplify_epsilon=2.0, 
                       georef_file=None, output_format='geojson'):
    """
    Vectorise tous les masques d'un dossier
    
    Args:
        mask_dir: Dossier contenant les masques PNG
        output_dir: Dossier de sortie pour les vecteurs
        min_area: Surface minimale polygones
        simplify_epsilon: Facteur de simplification
        georef_file: Fichier JSON contenant les paramètres de géoréférencement (optionnel)
        output_format: 'geojson' ou 'shapefile'
    """
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Charger paramètres géoréf si fournis
    transform_params = None
    if georef_file:
        with open(georef_file, 'r') as f:
            transform_params = json.load(f)
        print(f"📍 Géoréférencement chargé: {georef_file}")
        print(f"   CRS: {transform_params.get('crs', 'N/A')}")
    
    # Lister tous les masques
    mask_files = sorted(list(mask_dir.glob('*.png')))
    print(f"🔍 Trouvé {len(mask_files)} masques dans {mask_dir}")
    
    # Statistiques
    total_polygons = 0
    
    for mask_path in tqdm(mask_files, desc="Vectorisation"):
        # Extraire polygones
        polygons = mask_to_polygons(mask_path, min_area, simplify_epsilon)
        
        if len(polygons) == 0:
            continue
        
        total_polygons += len(polygons)
        
        # Appliquer transformation géo si disponible
        if transform_params:
            polygons = pixel_to_geo_coords(polygons, transform_params)
        
        # Créer GeoJSON
        geojson = polygons_to_geojson(
            polygons, 
            properties={
                'source_image': mask_path.stem,
                'num_vertices': sum(len(p) for p in polygons)
            }
        )
        
        # Sauvegarder
        output_path = output_dir / f"{mask_path.stem}.geojson"
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
    
    print(f"\n✅ Vectorisation terminée!")
    print(f"   Fichiers traités: {len(mask_files)}")
    print(f"   Polygones extraits: {total_polygons}")
    print(f"   Sortie: {output_dir}")


def merge_geojson_files(input_dir, output_file):
    """
    Fusionne plusieurs fichiers GeoJSON en un seul
    
    Args:
        input_dir: Dossier contenant les GeoJSON
        output_file: Fichier GeoJSON de sortie
    """
    input_dir = Path(input_dir)
    geojson_files = sorted(list(input_dir.glob('*.geojson')))
    
    print(f"🔗 Fusion de {len(geojson_files)} fichiers GeoJSON...")
    
    all_features = []
    
    for geojson_path in tqdm(geojson_files, desc="Fusion"):
        with open(geojson_path, 'r') as f:
            data = json.load(f)
            all_features.extend(data['features'])
    
    # Créer GeoJSON fusionné
    merged = {
        "type": "FeatureCollection",
        "features": all_features
    }
    
    # Sauvegarder
    with open(output_file, 'w') as f:
        json.dump(merged, f)
    
    print(f"✅ Fusion terminée: {len(all_features)} features dans {output_file}")


def visualize_vectorization(mask_path, output_path, min_area=100, simplify_epsilon=2.0):
    """
    Visualise le résultat de vectorisation (masque + contours)
    """
    import matplotlib.pyplot as plt
    
    # Charger masque
    mask = np.array(Image.open(mask_path).convert('L'))
    
    # Extraire polygones
    polygons = mask_to_polygons(mask_path, min_area, simplify_epsilon)
    
    # Créer figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Masque original
    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title('Masque original')
    axes[0].axis('off')
    
    # Masque + polygones
    axes[1].imshow(mask, cmap='gray', alpha=0.5)
    
    # Dessiner polygones
    for polygon in polygons:
        polygon_np = np.array(polygon)
        axes[1].plot(polygon_np[:, 0], polygon_np[:, 1], 'r-', linewidth=1)
        axes[1].plot([polygon_np[-1, 0], polygon_np[0, 0]], 
                     [polygon_np[-1, 1], polygon_np[0, 1]], 'r-', linewidth=1)
    
    axes[1].set_title(f'Polygones vectorisés ({len(polygons)} objets)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualisation sauvegardée: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Vectoriser masques cadastraux')
    parser.add_argument('--input', type=str, required=True,
                        help='Masque unique ou dossier de masques')
    parser.add_argument('--output', type=str, default='vectors',
                        help='Dossier de sortie pour les GeoJSON')
    parser.add_argument('--min-area', type=int, default=100,
                        help='Surface minimale polygone (pixels²)')
    parser.add_argument('--simplify', type=float, default=2.0,
                        help='Facteur simplification Douglas-Peucker')
    parser.add_argument('--georef', type=str, default=None,
                        help='Fichier JSON avec paramètres géoréférencement')
    parser.add_argument('--merge', action='store_true',
                        help='Fusionner tous les GeoJSON en un seul fichier')
    parser.add_argument('--visualize', action='store_true',
                        help='Créer visualisations masque + contours')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    # Cas 1: Fichier unique
    if input_path.is_file():
        print(f"📸 Vectorisation fichier unique: {input_path}")
        
        polygons = mask_to_polygons(input_path, args.min_area, args.simplify)
        print(f"   Polygones extraits: {len(polygons)}")
        
        # Sauvegarder GeoJSON
        output_dir.mkdir(parents=True, exist_ok=True)
        geojson = polygons_to_geojson(polygons, {'source': input_path.stem})
        output_file = output_dir / f"{input_path.stem}.geojson"
        
        with open(output_file, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"✅ GeoJSON sauvegardé: {output_file}")
        
        # Visualisation
        if args.visualize:
            vis_path = output_dir / f"{input_path.stem}_vectorized.png"
            visualize_vectorization(input_path, vis_path, args.min_area, args.simplify)
    
    # Cas 2: Dossier
    elif input_path.is_dir():
        print(f"📁 Vectorisation batch: {input_path}")
        
        vectorize_directory(
            input_path, 
            output_dir, 
            args.min_area, 
            args.simplify,
            args.georef
        )
        
        # Fusion si demandé
        if args.merge:
            merged_file = output_dir / 'cadastre_merged.geojson'
            merge_geojson_files(output_dir, merged_file)
        
        # Visualisations sur quelques exemples
        if args.visualize:
            print(f"\n🎨 Création visualisations...")
            vis_dir = output_dir / 'visualizations'
            vis_dir.mkdir(exist_ok=True)
            
            mask_files = sorted(list(input_path.glob('*.png')))
            for mask_path in mask_files[:5]:  # 5 premiers
                vis_path = vis_dir / f"{mask_path.stem}_vectorized.png"
                visualize_vectorization(mask_path, vis_path, args.min_area, args.simplify)
            
            print(f"✅ Visualisations dans: {vis_dir}")
    
    else:
        print(f"❌ Erreur: {input_path} n'existe pas")


if __name__ == '__main__':
    main()
