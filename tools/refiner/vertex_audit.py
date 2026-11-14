"""Audit des sommets : angle de virage, longueur des segments, sommets par parcelle."""
import json, pickle, sys, numpy as np, shapely
from shapely.geometry import shape
sys.path.insert(0, "/home/lekalmar/PyCharmMiscProject/Code/Geenfoncier/GeneFoncier/terrapartus")

def ring_stats(polys):
    turn, seg = [], []
    for p in polys:
        for ring in [p.exterior, *p.interiors]:
            c = np.asarray(ring.coords)[:-1]
            if len(c) < 3: continue
            a = np.roll(c, 1, 0) - c; b = np.roll(c, -1, 0) - c
            ang = np.degrees(np.arccos(np.clip((a*b).sum(1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-12),-1,1)))
            turn.append(180 - ang); seg.append(np.linalg.norm(b, axis=1))
    turn, seg = np.concatenate(turn), np.concatenate(seg)
    return dict(n_vertices=len(turn), per_poly=round(len(turn)/len(polys),1),
                frac_turn_lt5=round(float((turn<5).mean()),3), frac_turn_lt10=round(float((turn<10).mean()),3),
                seg_med_m=round(float(np.median(seg)),2), frac_seg_lt1m=round(float((seg<1).mean()),3))

if __name__ == "__main__":
    for p in sys.argv[1:]:
        if p.endswith(".pkl"):
            r = pickle.load(open(p, "rb")); t = r["truth"]; shapely.prepare(t.zone)
            pred = [f for f in r["faces"] if t.zone.contains(f.representative_point())]
            print(p.split("/")[-2][:30], "GRASS", ring_stats(pred)); print(" "*30, "GT   ", ring_stats(t.parcels))
        else:
            fc = [shape(f["geometry"]) for f in json.load(open(p))["features"]]
            fc = [g for f in fc for g in (f.geoms if f.geom_type=="MultiPolygon" else [f])]
            print(p.split("/")[-1][:30], ring_stats(fc))


def network_stats(polys, grid=0.01):
    """Sommets du RÉSEAU de limites : degré (nb de segments incidents) et virage."""
    from collections import defaultdict
    adj = defaultdict(set)
    for p in polys:
        for ring in [p.exterior, *p.interiors]:
            c = np.round(np.asarray(ring.coords) / grid).astype(np.int64)
            for a, b in zip(map(tuple, c[:-1]), map(tuple, c[1:])):
                if a != b: adj[a].add(b); adj[b].add(a)
    deg2 = [(k, list(v)) for k, v in adj.items() if len(v) == 2]
    turn = []
    for k, (u, w) in deg2:
        a = np.subtract(u, k); b = np.subtract(w, k)
        cos = np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
        turn.append(180 - np.degrees(np.arccos(np.clip(cos, -1, 1))))
    turn = np.asarray(turn); n = len(adj)
    return dict(nodes=n, deg2=round(len(deg2)/n, 3), deg2_turn_lt3=round(float((turn < 3).sum()/n), 3),
                deg2_turn_lt8=round(float((turn < 8).sum()/n), 3))


if __name__ == "__main__":
    print("--- réseau")
    for p in sys.argv[1:]:
        if p.endswith(".pkl"):
            r = pickle.load(open(p, "rb")); t = r["truth"]; shapely.prepare(t.zone)
            pred = [f for f in r["faces"] if t.zone.contains(f.representative_point())]
            print(p.split("/")[-2][:30], "GRASS", network_stats(pred), "\n", " "*29, "GT   ", network_stats(t.parcels))
        else:
            fc = [shape(f["geometry"]) for f in json.load(open(p))["features"]]
            fc = [g for f in fc for g in (f.geoms if f.geom_type=="MultiPolygon" else [f])]
            print(p.split("/")[-1][:30], network_stats(fc))
