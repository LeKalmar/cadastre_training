"""Entraîne le classifieur d'arêtes « même parcelle principale ? » (jeu de tools/typologie/aretes.py).

Entrée 6 canaux : vignette RGB redressée + annexe (arête, polygone du dessus, du dessous).
ResNet18 (timm, poids ImageNet adaptés à 6 canaux), sortie = probabilité « même parcelle ».
Augmentations sûres pour la géométrie de l'arête : miroir gauche-droite, miroir haut-bas
(on échange alors les deux polygones), luminosité. Meilleur modèle = F1 de la classe
« même parcelle » en validation ; AUC et précision/rappel au seuil 0,5 dans le journal.

    .venv/bin/python tools/typologie/train_aretes.py --data data/regroupement_v1 --work-dir work_dirs/regroupement_v1
"""
import argparse
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

MEAN = np.array([0.485, 0.456, 0.406, 0.5, 0.5, 0.5], np.float32)
STD = np.array([0.229, 0.224, 0.225, 0.5, 0.5, 0.5], np.float32)


class Aretes(Dataset):
    def __init__(self, root, split, rows, aug=False):
        self.root = Path(root) / split
        self.rows = [r for r in rows if r["split"] == split]
        self.aug = aug

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        im = cv2.cvtColor(cv2.imread(str(self.root / "images" / f"{r['id']}.jpg")), cv2.COLOR_BGR2RGB)
        ax = cv2.cvtColor(cv2.imread(str(self.root / "aux" / f"{r['id']}.png")), cv2.COLOR_BGR2RGB)
        if self.aug:
            if np.random.rand() < 0.5:
                im, ax = im[:, ::-1], ax[:, ::-1]
            if np.random.rand() < 0.5:                 # haut <-> bas : on échange les polygones
                im, ax = im[::-1], ax[::-1][..., [0, 2, 1]]
            if np.random.rand() < 0.5:
                a, b = np.random.uniform(0.85, 1.15), np.random.uniform(-20, 20)
                im = np.clip(im.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        x = np.concatenate([im, ax], -1).astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.tensor(float(r["label"]))


@torch.no_grad()
def evaluer(model, dl, dev):
    model.eval()
    ps, ys = [], []
    for x, y in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(model(x.to(dev)).float().squeeze(1))
        ps.append(p.cpu().numpy())
        ys.append(y.numpy())
    p, y = np.concatenate(ps), np.concatenate(ys)
    pred = p >= 0.5
    tp, fp, fn = (pred & (y == 1)).sum(), (pred & (y == 0)).sum(), (~pred & (y == 1)).sum()
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    o = np.argsort(-p)                                 # AUC par les rangs
    ranks = np.empty(len(p))
    ranks[o] = np.arange(len(p), 0, -1)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)
    return dict(f1=float(f1), prec=float(prec), rec=float(rec), auc=float(auc), acc=float((pred == y).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/regroupement_v1")
    ap.add_argument("--work-dir", default="work_dirs/regroupement_v1")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    wd = Path(a.work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(Path(a.data) / "aretes.csv")))
    tr, va = Aretes(a.data, "train", rows, aug=True), Aretes(a.data, "val", rows)
    n1 = sum(r["label"] == "1" for r in tr.rows)
    pos_w = torch.tensor((len(tr) - n1) / max(n1, 1))
    print(f"{len(tr)} arêtes train ({n1} même parcelle), {len(va)} val ; pos_weight {pos_w:.2f}", flush=True)
    dl_tr = DataLoader(tr, a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True,
                       persistent_workers=True)
    dl_va = DataLoader(va, a.batch_size, num_workers=a.workers, pin_memory=True)
    dev = "cuda"
    model = timm.create_model(a.arch, pretrained=True, in_chans=6, num_classes=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 200) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    log = open(wd / "train_log.csv", "a", newline="")
    lw = csv.writer(log)
    if log.tell() == 0:
        lw.writerow(["epoch", "loss", "f1", "prec", "rec", "auc", "acc", "sec"])
    best = -1.0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x).float().squeeze(1)
            loss = F.binary_cross_entropy_with_logits(out, y, pos_weight=pos_w.to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            n += 1
        m = evaluer(model, dl_va, dev)
        dt = time.time() - t0
        lw.writerow([ep, round(tot / n, 4)] + [round(m[k], 4) for k in ("f1", "prec", "rec", "auc", "acc")] + [round(dt)])
        log.flush()
        etat = {"model": model.state_dict(), "epoch": ep, "metrics": m,
                "config": {"arch": a.arch, "data": a.data, "in_chans": 6}}
        torch.save(etat, wd / "last_model.pth")
        if m["f1"] > best:
            best = m["f1"]
            torch.save(etat, wd / "best_model.pth")
        print(f"ép {ep:3}  perte {tot / n:.4f}  F1 {m['f1']:.3f}  P {m['prec']:.3f}  R {m['rec']:.3f}  "
              f"AUC {m['auc']:.3f}  ({dt:.0f} s){'  *' if m['f1'] == best else ''}", flush=True)


if __name__ == "__main__":
    main()
