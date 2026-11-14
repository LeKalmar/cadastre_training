#!/usr/bin/env python3
"""
Inférence par tuiles pour grandes images cadastrales.

Deux modes de post-processing disponibles :

  --watershed   (recommandé)
    Applique un watershed de Meyer sur la heatmap de confiance brute (float 0-1).
    Garantit des polygones fermés sans graph completion.
    Paramètres : --ws-dynamic, --ws-area

  --graph-completion  (ancien mode, conservé)
    Squelettisation + connexion géométrique des endpoints.
    Utile si on veut des lignes 1px plutôt que des régions.

Dépendances :
    pip install segmentation-models-pytorch scikit-image sknw networkx
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.morphology import skeletonize
from skimage.segmentation import watershed
from skimage.filters import rank
from skimage.morphology import disk
from scipy import ndimage as ndi
from torchvision import transforms
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Watershed Meyer (Python pur — équivalent de histmapseg)
# ─────────────────────────────────────────────────────────────────────────────

def watershed_polygons(heatmap_float, dynamic=0.3, area_min=50, area_max=0,
                       threshold=0.3):
    """
    Extrait des polygones fermés depuis une heatmap de confiance de bords.

    Principe correct :
      Les lignes blanches prédites par le modèle sont des DIGUES qui séparent
      des bassins noirs. Le watershed doit :
        1. Identifier les zones noires fermées (intérieurs de parcelles)
        2. Semer une graine dans chaque zone fermée
        3. "Inonder" depuis ces graines jusqu'aux bords blancs

    Concètement :
      - On seuille la heatmap → masque binaire (blanc=bord, noir=intérieur)
      - On étiqueite chaque composante noire connexe = 1 graine potentielle
      - On filtre les graines trop petites (bruit entre deux lignes proches)
      - Le watershed propage chaque graine jusqu'aux bords blancs
      - Les frontières entre régions = les lignes des parcelles

    Paramètres :
      threshold : seuil de binarisation de la heatmap (défaut 0.3)
                  Plus bas = récupère plus de lignes faibles → moins de trous
      dynamic   : taille min du kernel d'érosion pour chercher les graines
                  (3-7 = petites parcelles urbaines, 9-15 = grandes parcelles rurales)
      area_min  : surface min d'une région finale en pixels (défaut 50)
      area_max  : surface max d'une région finale (0 = pas de limite)
    """
    h, w = heatmap_float.shape

    # ── 1. Binarisation de la heatmap ────────────────────────────────────────
    # Bords blancs (heatmap haute) = digues, zones noires = bassins
    edges_binary = (heatmap_float > threshold).astype(np.uint8)  # 1=bord, 0=intérieur

    # Dilatation légère des bords pour bien fermer les contours légèrement ouverts
    # (équivalent du paramètre dynamic de histmapseg)
    kernel_size = max(3, int(dynamic * 10) | 1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (kernel_size, kernel_size))
    edges_closed = cv2.dilate(edges_binary, kernel_close)

    # ── 2. Zones noires = intérieurs potentiels de parcelles ─────────────────
    interior = (edges_closed == 0).astype(np.uint8)

    # ── 3. Graines : composantes connexes des zones noires fermées ───────────
    # Chaque composante connexe noire = 1 future parcelle
    # On utilise une érosion pour que les graines soient bien au centre
    # et ne touchent pas les bords (évite la sur-segmentation)
    kernel_seed = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    interior_eroded = cv2.erode(interior, kernel_seed, iterations=2)

    seeds_labeled, n_seeds = ndi.label(interior_eroded)
    print(f"  Watershed : {n_seeds} graines détectées")

    if n_seeds == 0:
        print("  ⚠️  Aucune graine — essayez de baisser --ws-threshold")
        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.int32)

    # ── 4. Surface topographique pour le watershed ───────────────────────────
    # On veut que le watershed s'arrête aux bords blancs.
    # La distance transform donne à chaque pixel noir sa distance au bord
    # blanc le plus proche → les pixels proches des bords ont une valeur faible,
    # les pixels au centre des parcelles ont une valeur haute.
    # On INVERSE pour que le watershed monte des centres vers les bords.
    dist_transform = ndi.distance_transform_edt(interior)
    topo = -dist_transform  # minima aux centres, maxima aux bords

    # ── 5. Watershed ─────────────────────────────────────────────────────────
    # mask=interior : le watershed ne se propage que dans les zones noires
    labels = watershed(topo, markers=seeds_labeled, mask=interior)

    # ── 6. Filtrage des régions trop petites ou trop grandes ─────────────────
    unique, counts = np.unique(labels, return_counts=True)
    region_sizes = dict(zip(unique, counts))
    remove_mask = np.zeros((h, w), dtype=bool)
    n_removed = 0
    for label_id, size in region_sizes.items():
        if label_id == 0:
            continue
        if size < area_min:
            remove_mask |= (labels == label_id)
            n_removed += 1
        elif area_max > 0 and size > area_max:
            remove_mask |= (labels == label_id)
            n_removed += 1
    labels[remove_mask] = 0
    if n_removed:
        print(f"  Watershed : {n_removed} régions filtrées "
              f"(area<{area_min} ou area>{area_max if area_max else '∞'})")

    # ── 7. Extraire les bords entre régions adjacentes ───────────────────────
    # Méthode : un pixel de label A est un bord si un voisin a label B ≠ A
    lp = np.pad(labels, 1, mode='edge')
    edges_out = np.zeros((h, w), dtype=bool)
    edges_out |= (labels != lp[:-2, 1:-1])
    edges_out |= (labels != lp[2:,  1:-1])
    edges_out |= (labels != lp[1:-1, :-2])
    edges_out |= (labels != lp[1:-1, 2:])

    # Garder uniquement les bords entre deux parcelles (pas avec le fond=0)
    # → on veut les frontières inter-parcelles, pas les bords externes de la carte
    has_labeled_neighbor = np.zeros((h, w), dtype=bool)
    has_labeled_neighbor |= (lp[:-2, 1:-1] > 0)
    has_labeled_neighbor |= (lp[2:,  1:-1] > 0)
    has_labeled_neighbor |= (lp[1:-1, :-2] > 0)
    has_labeled_neighbor |= (lp[1:-1, 2:]  > 0)
    edges_out &= (labels > 0) | has_labeled_neighbor

    edges_uint8 = (edges_out * 255).astype(np.uint8)
    n_regions = len(np.unique(labels[labels > 0]))
    print(f"  Watershed : {n_regions} polygones extraits")

    return edges_uint8, labels


def save_watershed_debug(labels, output_path):
    """
    Sauvegarde une visualisation colorée des régions watershed.
    Utile pour déboguer et ajuster les paramètres.
    """
    if labels.max() == 0:
        return
    # Colorisation aléatoire reproductible par label
    np.random.seed(42)
    colors = np.random.randint(60, 220, (labels.max() + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # fond = noir
    colored = colors[labels]
    cv2.imwrite(str(output_path), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))


# ─────────────────────────────────────────────────────────────────────────────
# Graph completion (conservé pour le mode --graph-completion)
# ─────────────────────────────────────────────────────────────────────────────

def _check_graph_deps():
    try:
        import sknw      # noqa: F401
        import networkx  # noqa: F401
        return True
    except ImportError:
        print("  ⚠️  sknw ou networkx manquant : pip install sknw networkx")
        return False


def _endpoint_tangent(graph, node, n_sample=8):
    pos = np.array(graph.nodes[node]['o'], dtype=np.float32)
    neighbors = list(graph.neighbors(node))
    if not neighbors:
        return pos, np.array([1.0, 0.0])
    nb = neighbors[0]
    edge_pts = graph[node][nb]['pts']
    n = min(n_sample, len(edge_pts))
    if n < 2:
        tangent = np.array(graph.nodes[nb]['o'], dtype=np.float32) - pos
    else:
        dist_start = np.linalg.norm(edge_pts[0]  - pos)
        dist_end   = np.linalg.norm(edge_pts[-1] - pos)
        pts = edge_pts[:n] if dist_start < dist_end else edge_pts[-n:][::-1]
        tangent = pts[-1].astype(np.float32) - pts[0].astype(np.float32)
    norm = np.linalg.norm(tangent)
    return pos, (tangent / norm if norm > 0 else np.array([1.0, 0.0]))


def _ray_to_segment_intersection(pos, tan, a, b, max_dist):
    ab = b - a
    ab_len = np.linalg.norm(ab)
    if ab_len < 1e-6:
        return None, None
    ab_dir = ab / ab_len
    pa = pos - a
    t_along   = np.dot(pa, ab_dir)
    perp_vec  = pa - t_along * ab_dir
    perp_dist = np.linalg.norm(perp_vec)
    if perp_dist > max_dist:
        return None, None
    if t_along < -max_dist or t_along > ab_len + max_dist:
        return None, None
    t_clamped = np.clip(t_along, 0.0, ab_len)
    closest   = a + t_clamped * ab_dir
    to_closest = closest - pos
    dist_to_closest = np.linalg.norm(to_closest)
    if dist_to_closest < 1e-3:
        return None, None
    if np.dot(tan, to_closest / dist_to_closest) < 0.0:
        return None, None
    return dist_to_closest, closest


def graph_completion(skeleton_uint8, max_gap_px=20, angle_tol_deg=30):
    if not _check_graph_deps():
        return skeleton_uint8
    import sknw
    skel_bool = skeleton_uint8 > 0
    graph  = sknw.build_sknw(skel_bool.astype(np.uint16))
    result = skeleton_uint8.copy()
    endpoints = [n for n, deg in graph.degree() if deg == 1]
    if not endpoints:
        return result
    ep_data = []
    for n in endpoints:
        pos, tan = _endpoint_tangent(graph, n)
        nbs = list(graph.neighbors(n))
        ep_data.append({'node': n, 'pos': pos, 'tangent': tan,
                        'nb': nbs[0] if nbs else None})
    edges_index = []
    for u, v, data in graph.edges(data=True):
        pts = data['pts'].astype(np.float32)
        if len(pts) >= 2:
            edges_index.append((u, v, pts[::2]))
    connected_eps = set()
    n_ep2seg = 0
    for i, ei in enumerate(ep_data):
        pos = ei['pos']
        tan = ei['tangent']
        best_dist = max_gap_px
        best_closest = None
        for u, v, pts in edges_index:
            if ei['node'] in (u, v):
                continue
            r_min, c_min = pts.min(axis=0)
            r_max, c_max = pts.max(axis=0)
            if (pos[0] < r_min - max_gap_px or pos[0] > r_max + max_gap_px or
                    pos[1] < c_min - max_gap_px or pos[1] > c_max + max_gap_px):
                continue
            for k in range(len(pts) - 1):
                d, closest = _ray_to_segment_intersection(
                    pos, tan, pts[k], pts[k + 1], max_gap_px)
                if d is not None and d < best_dist:
                    best_dist = d
                    best_closest = closest
        if best_closest is not None:
            p1 = (int(round(pos[1])),          int(round(pos[0])))
            p2 = (int(round(best_closest[1])), int(round(best_closest[0])))
            cv2.line(result, p1, p2, 255, 1)
            connected_eps.add(i)
            n_ep2seg += 1
    cos_col = np.cos(np.radians(angle_tol_deg))
    n_ep2ep = 0
    for i in range(len(ep_data)):
        if i in connected_eps:
            continue
        ei = ep_data[i]
        best_dist = max_gap_px
        best_j = None
        for j in range(len(ep_data)):
            if i == j or j in connected_eps:
                continue
            ej = ep_data[j]
            diff = ej['pos'] - ei['pos']
            dist = np.linalg.norm(diff)
            if dist > best_dist or dist < 1:
                continue
            gap_dir = diff / dist
            if (np.dot(ei['tangent'],  gap_dir) < cos_col or
                    np.dot(ej['tangent'], -gap_dir) < cos_col):
                continue
            best_dist = dist
            best_j = j
        if best_j is not None:
            ej = ep_data[best_j]
            p1 = (int(round(ei['pos'][1])), int(round(ei['pos'][0])))
            p2 = (int(round(ej['pos'][1])), int(round(ej['pos'][0])))
            cv2.line(result, p1, p2, 255, 1)
            connected_eps.add(i)
            connected_eps.add(best_j)
            n_ep2ep += 1
    print(f"  Graph completion : {n_ep2seg} ep→segment  +  {n_ep2ep} ep→ep "
          f"(max_gap={max_gap_px}px, tol={angle_tol_deg}°)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Inférence tuilée
# ─────────────────────────────────────────────────────────────────────────────

class TiledInference:
    def __init__(self, checkpoint_path, tile_size=None, overlap=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(checkpoint_path, map_location=self.device,
                                weights_only=False)
        config = checkpoint.get('config', {})

        import segmentation_models_pytorch as smp
        encoder = config.get('encoder', 'resnext50_32x4d')
        self.model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            decoder_attention_type='scse'
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        ckpt_tile = config.get('tile_size', config.get('img_size', 512))
        self.tile_size = tile_size if tile_size is not None else ckpt_tile
        self.overlap   = overlap   if overlap   is not None else self.tile_size // 8
        self.stride    = self.tile_size - self.overlap

        print(f"  Encodeur  : {encoder}")
        print(f"  Tile size : {self.tile_size}px  (checkpoint={ckpt_tile}px)")
        print(f"  Overlap   : {self.overlap}px")
        if tile_size is not None and tile_size != ckpt_tile:
            print(f"  ⚠️  tile_size forcé {tile_size} ≠ {ckpt_tile} entraînement")

    @torch.no_grad()
    def _build_heatmap(self, image_path):
        """Construit la heatmap de confiance float32 [0,1] sur toute l'image."""
        Image.MAX_IMAGE_PIXELS = None
        img    = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        h, w   = img_np.shape[:2]

        full_mask  = np.zeros((h, w), dtype=np.float32)
        count_mask = np.zeros((h, w), dtype=np.float32)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        y_coords = range(0, max(1, h - self.tile_size + self.stride), self.stride)
        x_coords = range(0, max(1, w - self.tile_size + self.stride), self.stride)
        print(f"  {len(y_coords) * len(x_coords)} tuiles à traiter...")

        for y in tqdm(y_coords, desc="Inférence"):
            for x in x_coords:
                y_end = min(y + self.tile_size, h)
                x_end = min(x + self.tile_size, w)
                tile  = img_np[y:y_end, x:x_end]
                pad_h = self.tile_size - tile.shape[0]
                pad_w = self.tile_size - tile.shape[1]
                if pad_h > 0 or pad_w > 0:
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)),
                                  mode='constant')
                tile_tensor = transform(tile).unsqueeze(0).to(self.device)
                with torch.amp.autocast('cuda'):
                    pred = torch.sigmoid(self.model(tile_tensor))
                pred_np = pred.squeeze().cpu().numpy()[:(y_end - y), :(x_end - x)]
                full_mask [y:y_end, x:x_end] += pred_np
                count_mask[y:y_end, x:x_end] += 1

        full_mask /= np.maximum(count_mask, 1)
        return full_mask, img_np

    def process_image(self, image_path,
                      # Mode watershed
                      do_watershed=False,
                      ws_dynamic=0.3,
                      ws_area_min=50,
                      ws_area_max=0,
                      ws_threshold=0.3,
                      ws_debug=False,
                      ws_debug_path=None,
                      # Mode classique
                      threshold=0.4,
                      do_skeleton=False,
                      do_graph_completion=False,
                      gc_max_gap=20,
                      gc_angle_tol=30):

        heatmap, img_np = self._build_heatmap(image_path)

        if do_watershed:
            # ── Mode watershed : polygones fermés garantis ───────────────────
            print("  Watershed Meyer...")
            edges, labels = watershed_polygons(
                heatmap,
                dynamic=ws_dynamic,
                area_min=ws_area_min,
                area_max=ws_area_max,
                threshold=ws_threshold,
            )
            if ws_debug and ws_debug_path:
                save_watershed_debug(labels, ws_debug_path)
                print(f"  Debug watershed : {ws_debug_path}")
            return edges, img_np, heatmap

        else:
            # ── Mode classique : seuillage + squelette + graph completion ────
            mask_binary = (heatmap > threshold).astype(np.uint8) * 255
            if do_skeleton or do_graph_completion:
                print("  Squelettisation...")
                mask_binary = (skeletonize(mask_binary > 0) * 255).astype(np.uint8)
            if do_graph_completion:
                print("  Graph completion...")
                mask_binary = graph_completion(
                    mask_binary, max_gap_px=gc_max_gap,
                    angle_tol_deg=gc_angle_tol)
            return mask_binary, img_np, heatmap


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inférence cadastre — watershed ou graph completion",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input',      required=True)
    parser.add_argument('--output',     default='predictions')
    parser.add_argument('--tile-size',  type=int, default=None)
    parser.add_argument('--overlap',    type=int, default=None)
    parser.add_argument('--save-heatmap', action='store_true',
                        help="Sauvegarder aussi la heatmap float (PNG 8bit)")

    # ── Mode watershed ───────────────────────────────────────────────────────
    ws = parser.add_argument_group('Mode Watershed (recommandé)')
    ws.add_argument('--watershed', action='store_true',
                    help="Activer le watershed Meyer → polygones fermés garantis")
    ws.add_argument('--ws-dynamic', type=float, default=0.3,
                    help="Profondeur minimale d'un bassin versant\n"
                         "  0.1-0.2 : permissif (beaucoup de petites parcelles)\n"
                         "  0.3     : standard (défaut)\n"
                         "  0.4-0.6 : strict (grandes parcelles seulement)")
    ws.add_argument('--ws-area-min', type=int, default=50,
                    help="Surface min d'une région en pixels (défaut: 50)\n"
                         "  Augmenter si trop de petits artefacts")
    ws.add_argument('--ws-area-max', type=int, default=0,
                    help="Surface max d'une région (0=sans limite)\n"
                         "  Utile pour exclure le fond de la carte")
    ws.add_argument('--ws-threshold', type=float, default=0.3,
                    help="Seuil de confiance pour les bords (défaut: 0.3)\n"
                         "  Plus bas = récupère plus de lignes faibles")
    ws.add_argument('--ws-debug', action='store_true',
                    help="Sauvegarder une image colorée des régions watershed")

    # ── Mode graph completion ────────────────────────────────────────────────
    gc = parser.add_argument_group('Mode Graph Completion (ancien)')
    gc.add_argument('--threshold',   type=float, default=0.4)
    gc.add_argument('--skeleton',    action='store_true')
    gc.add_argument('--graph-completion', action='store_true')
    gc.add_argument('--gc-max-gap',  type=int, default=20)
    gc.add_argument('--gc-angle-tol',type=int, default=30)

    args = parser.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = TiledInference(args.checkpoint,
                               tile_size=args.tile_size,
                               overlap=args.overlap)

    input_path = Path(args.input)
    files = ([input_path] if input_path.is_file()
             else sorted(input_path.glob('*.*')))
    valid_ext = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

    for f in files:
        if f.suffix.lower() not in valid_ext:
            continue
        print(f"\n📄 {f.name}")

        debug_path = out_dir / f"{f.stem}_ws_regions.png" if args.ws_debug else None

        result, _, heatmap = processor.process_image(
            f,
            do_watershed=args.watershed,
            ws_dynamic=args.ws_dynamic,
            ws_area_min=args.ws_area_min,
            ws_area_max=args.ws_area_max,
            ws_threshold=args.ws_threshold,
            ws_debug=args.ws_debug,
            ws_debug_path=debug_path,
            threshold=args.threshold,
            do_skeleton=args.skeleton,
            do_graph_completion=args.graph_completion,
            gc_max_gap=args.gc_max_gap,
            gc_angle_tol=args.gc_angle_tol,
        )

        save_path = out_dir / f"{f.stem}_pred.png"
        cv2.imwrite(str(save_path), result)
        print(f"  Sauvegardé : {save_path}")

        if args.save_heatmap:
            hm_path = out_dir / f"{f.stem}_heatmap.png"
            cv2.imwrite(str(hm_path), (heatmap * 255).astype(np.uint8))
            print(f"  Heatmap    : {hm_path}")


if __name__ == "__main__":
    main()