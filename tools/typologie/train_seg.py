"""Entraîne la segmentation de la typologie visible (jeu de tools/typologie/build_dataset.py).

U-Net (segmentation_models_pytorch) à 5 classes, perte entropie croisée pondérée + Dice,
pixels 255 ignorés. Tirage renforcé des tuiles qui contiennent une classe rare (bâti public,
eau, végétation). Augmentations géométriques et de luminosité seulement : la TEINTE est
l'indice (rose = bâti, rouge = bâti public, bleu = eau), on n'y touche pas.
Meilleur modèle = IoU moyenne des classes présentes en validation.

    .venv/bin/python tools/typologie/train_seg.py --data data/typologie_v1 --work-dir work_dirs/typologie_v1
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

IGNORE = 255


class Tuiles(Dataset):
    def __init__(self, root, split, classes, aug=None):
        self.img = sorted((Path(root) / split / "images").glob("*.jpg"))
        self.lab_dir = Path(root) / split / "labels"
        self.aug = aug
        self.nc = len(classes)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        p = self.img[i]
        im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        lab = cv2.imread(str(self.lab_dir / (p.stem + ".png")), cv2.IMREAD_GRAYSCALE)
        if self.aug:
            o = self.aug(image=im, mask=lab)
            im, lab = o["image"], o["mask"]
        x = torch.from_numpy(im.transpose(2, 0, 1).copy()).float() / 255.0
        x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return x, torch.from_numpy(lab.astype(np.int64))


def poids_tirage(root, ds, classes, boost):
    """Poids de tirage par tuile : boost^(classes rares présentes)."""
    fr = {r["tile"]: r for r in csv.DictReader(open(Path(root) / "tiles.csv"))}
    rares = [c for c in classes if c not in ("non_bati", "bati")]
    w = []
    for p in ds.img:
        r = fr.get(p.stem)
        n = sum(float(r[f"f_{c}"]) > 0.002 for c in rares) if r else 0
        w.append(boost ** n)
    return w


def frequences(ds, nc, n=400):
    rng = np.random.default_rng(0)
    h = np.zeros(nc)
    for i in rng.choice(len(ds), min(n, len(ds)), replace=False):
        lab = cv2.imread(str(ds.lab_dir / (ds.img[i].stem + ".png")), cv2.IMREAD_GRAYSCALE)
        h += np.bincount(lab[lab != IGNORE].ravel(), minlength=nc)[:nc]
    return h / max(h.sum(), 1)


@torch.no_grad()
def evaluer(model, dl, nc, dev):
    model.eval()
    inter, union, tp_n = np.zeros(nc), np.zeros(nc), np.zeros(nc)
    for x, y in dl:
        x, y = x.to(dev), y.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x).argmax(1)
        ok = y != IGNORE
        for k in range(nc):
            p, t = (pred == k) & ok, (y == k) & ok
            inter[k] += (p & t).sum().item()
            union[k] += (p | t).sum().item()
            tp_n[k] += t.sum().item()
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    present = tp_n > 0
    return iou, float(np.nanmean(iou[present])) if present.any() else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/typologie_v1")
    ap.add_argument("--work-dir", default="work_dirs/typologie_v1")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--boost", type=float, default=3.0, help="tirage x boost par classe rare présente")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume")
    a = ap.parse_args()
    wd = Path(a.work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(Path(a.data) / "summary.json"))
    classes = meta["classes"]
    nc = len(classes)
    dev = "cuda"

    aug = A.Compose([
        A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90(),
        A.Affine(scale=(0.85, 1.15), rotate=(-10, 10), border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
        A.GaussNoise(p=0.15), A.ImageCompression(quality_range=(60, 95), p=0.3),
    ])
    tr, va = Tuiles(a.data, "train", classes, aug), Tuiles(a.data, "val", classes)
    w = poids_tirage(a.data, tr, classes, a.boost)
    dl_tr = DataLoader(tr, a.batch_size, sampler=WeightedRandomSampler(w, len(tr)), num_workers=a.workers,
                       pin_memory=True, drop_last=True, persistent_workers=True)
    dl_va = DataLoader(va, a.batch_size, num_workers=a.workers, pin_memory=True)
    f = frequences(tr, nc)
    cw = torch.tensor(np.clip(1 / np.sqrt(np.maximum(f, 1e-4)), 1, 20), dtype=torch.float32)
    cw = (cw / cw[0]).to(dev)
    print(f"{len(tr)} tuiles train, {len(va)} val ; fréquences {np.round(f, 4).tolist()} ; poids {cw.cpu().numpy().round(2).tolist()}")

    model = smp.Unet(a.encoder, encoder_weights="imagenet", classes=nc, decoder_attention_type="scse").to(dev)
    if a.resume:
        model.load_state_dict(torch.load(a.resume, map_location=dev)["model"])
    dice = smp.losses.DiceLoss("multiclass", ignore_index=IGNORE)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 300) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    log = open(wd / "train_log.csv", "a", newline="")
    lw = csv.writer(log)
    if log.tell() == 0:
        lw.writerow(["epoch", "loss", "miou"] + [f"iou_{c}" for c in classes] + ["sec"])
    best = -1.0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
            out = out.float()
            loss = F.cross_entropy(out, y, weight=cw, ignore_index=IGNORE) + 0.5 * dice(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            n += 1
        iou, miou = evaluer(model, dl_va, nc, dev)
        dt = time.time() - t0
        lw.writerow([ep, round(tot / n, 4), round(miou, 4)] + [None if np.isnan(v) else round(v, 4) for v in iou] + [round(dt)])
        log.flush()
        etat = {"model": model.state_dict(), "epoch": ep, "miou": miou, "iou": iou.tolist(),
                "config": {"encoder": a.encoder, "classes": classes, "data": a.data}}
        torch.save(etat, wd / "last_model.pth")
        if miou > best:
            best = miou
            torch.save(etat, wd / "best_model.pth")
        print(f"ép {ep:3}  perte {tot / n:.4f}  mIoU {miou:.3f}  "
              + "  ".join(f"{c} {v:.2f}" for c, v in zip(classes, iou) if not np.isnan(v))
              + f"  ({dt:.0f} s){'  *' if miou == best else ''}", flush=True)


if __name__ == "__main__":
    main()
