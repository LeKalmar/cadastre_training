#!/usr/bin/env python3
"""
Mesure, pour chaque encodeur candidat, la mémoire GPU de pointe et la vitesse d'un
pas d'entraînement réel (U-Net scSE, bf16, perte v5, 512 px), pour savoir ce qui
tient sur la carte graphique.

    .venv/bin/python tools/bench_encoders.py --batch 8
"""
import argparse, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train import build_model, CombinedLossV5  # noqa: E402

CANDIDATES = [
    "resnet34",
    "mit_b3",                              # SegFormer (Bertrand)
    "mit_b5",
    "tu-convnext_tiny.dinov3_lvd1689m",    # DINOv3 distillé (Meta, 2025)
    "tu-convnext_small.dinov3_lvd1689m",
    "tu-convnext_base.dinov3_lvd1689m",
    "tu-convnextv2_base.fcmae_ft_in22k_in1k",
]


def bench(enc, batch, size, steps=12):
    dev = torch.device("cuda")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = build_model(enc, pretrained=False).to(dev)
    crit = CombinedLossV5(alpha=0.25, beta=0.75, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(batch, 3, size, size, device=dev)
    t = (torch.rand(batch, 1, size, size, device=dev) > 0.97).float()
    w = torch.ones_like(t)
    times = []
    for i in range(steps):
        torch.cuda.synchronize(); t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        loss = crit(out.float(), t, w); loss.backward(); opt.step()
        torch.cuda.synchronize()
        if i >= 2: times.append(time.time() - t0)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    del model, opt; torch.cuda.empty_cache()
    return params, peak, batch / (sum(times) / len(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--encoders", nargs="*", default=CANDIDATES)
    a = ap.parse_args()
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    print(f"GPU {torch.cuda.get_device_name(0)} ({total:.1f} Go) | batch {a.batch} | {a.size}px | bf16\n")
    print(f"{'encodeur':42} {'params':>8} {'mémoire':>9} {'img/s':>7}  époque*")
    for enc in a.encoders:
        try:
            p, m, ips = bench(enc, a.batch, a.size)
            print(f"{enc:42} {p:7.1f}M {m:7.2f}Go {ips:7.1f}  {9000 / ips / 60:5.1f} min", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"{enc:42}   ---    MÉMOIRE INSUFFISANTE", flush=True); torch.cuda.empty_cache()
        except Exception as e:
            print(f"{enc:42}   erreur : {str(e)[:80]}", flush=True)
    print("\n* temps de calcul d'une époque de 9 000 tuiles (hors chargement des données)")


if __name__ == "__main__":
    main()
