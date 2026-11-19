"""Train the ResU-Net denoiser at one noise level.

Run:
    uv run python scripts/train.py --overfit-one-batch       # sanity check, 1-2 minutes
    uv run python scripts/train.py --epochs 2 --limit 100    # quick run
    uv run python scripts/train.py                           # full run (medium noise)

Training data stays on the GPU; every batch gets NEW random noise (so the network
never sees the same noisy image twice). After each epoch the model is scored on the
fixed noisy validation set; the epoch with the best validation PSNR is kept.

Writes outputs/checkpoints/resunet_<level>_best.pt (and _last.pt),
outputs/results/training_log_<level>.csv and outputs/figures/training_curves_<level>.png.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from c21.config import load_config
from c21.data import clean_array_path, load_clean, load_noisy
from c21.metrics import psnr
from c21.models.resunet import ResUNet, count_parameters, denoise_numpy, save_checkpoint
from c21.noise import add_noise_torch, get_level
from c21.utils import get_device, set_seed
from c21.viz import save_training_curves


def mean_psnr(clean: np.ndarray, outputs: np.ndarray) -> float:
    return float(np.mean([psnr(c, o) for c, o in zip(clean, outputs)]))


def overfit_one_batch(cfg, args, level, device) -> int:
    """Train on 4 images with FIXED noise. A working network must fit them quickly."""
    x = torch.from_numpy(np.load(clean_array_path(cfg, "train"))[:4]).float().div(255).unsqueeze(1).to(device)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    y = add_noise_torch(x, level.lambda_photons, level.sigma_electronic, generator=gen)
    model = ResUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Overfit test: 4 images, fixed noise, {args.overfit_steps} steps")
    first = None
    for step in range(1, args.overfit_steps + 1):
        loss = F.l1_loss(model(y), x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
        if step == 1 or step % 50 == 0:
            with torch.no_grad():
                p = mean_psnr(x.squeeze(1).cpu().numpy(), model(y).clamp(0, 1).squeeze(1).cpu().numpy())
            print(f"  step {step:4d}  L1 {loss.item():.4f}  PSNR {p:6.2f} dB")
    ratio = first / max(loss.item(), 1e-12)
    ok = ratio > 3 and math.isfinite(loss.item())
    print(f"\n[{'PASS' if ok else 'FAIL'}] loss dropped {ratio:.1f}x (needs more than 3x)")
    return 0 if ok else 1


def train(cfg, args, level, device) -> int:
    tcfg = cfg["training"]
    epochs = args.epochs or tcfg["epochs"]
    batch_size = args.batch_size or tcfg["batch_size"]
    lr = args.lr

    train_u8 = torch.from_numpy(np.load(clean_array_path(cfg, "train")))
    val_clean = load_clean(cfg, "val")
    val_noisy = load_noisy(cfg, "val", level.name)
    if args.limit:
        train_u8, val_clean, val_noisy = train_u8[: args.limit], val_clean[: args.limit], val_noisy[: args.limit]
    train_u8 = train_u8.to(device)
    n = len(train_u8)
    steps = math.ceil(n / batch_size)
    noisy_val_psnr = mean_psnr(val_clean, np.clip(val_noisy, 0, 1))

    model = ResUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
    g_cpu = torch.Generator().manual_seed(args.seed)
    g_dev = torch.Generator(device=device).manual_seed(args.seed)
    flip = tcfg.get("augment_flip", True)

    print(f"Training: {n} images, batch {batch_size}, {epochs} epochs ({steps} steps each), lr {lr:g} (cosine), "
          f"flip augmentation {'on' if flip else 'off'}")
    print(f"Validation: {len(val_clean)} fixed noisy images, noisy-input PSNR {noisy_val_psnr:.2f} dB\n")

    tag = level.name
    ckpt_dir = cfg["paths"]["outputs"] / "checkpoints"
    meta = {"level": level.name, "lambda_photons": level.lambda_photons,
            "sigma_electronic": level.sigma_electronic, "seed": args.seed, "lr": lr,
            "batch_size": batch_size, "epochs": epochs, "train_images": n}
    log, best = [], -1.0
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(n, generator=g_cpu)
        total = 0.0
        for s in range(steps):
            idx = perm[s * batch_size:(s + 1) * batch_size].to(device)
            x = train_u8[idx].float().div(255).unsqueeze(1)
            if flip:
                mask = torch.rand(len(idx), generator=g_dev, device=device) < 0.5
                x = torch.where(mask[:, None, None, None], x.flip(-1), x)
            y = add_noise_torch(x, level.lambda_photons, level.sigma_electronic, generator=g_dev)
            loss = F.l1_loss(model(y), x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        train_l1 = total / n
        if not math.isfinite(train_l1):
            print(f"epoch {epoch}: loss is {train_l1}. Stopping.")
            return 1

        val_out = denoise_numpy(model, val_noisy, device)
        val_psnr = mean_psnr(val_clean, val_out)
        val_l1 = float(np.abs(val_out - val_clean).mean())
        is_best = val_psnr > best
        if is_best:
            best = val_psnr
            save_checkpoint(ckpt_dir / f"resunet_{tag}_best.pt", model, {**meta, "epoch": epoch, "val_psnr": val_psnr})
        secs = time.perf_counter() - t0
        log.append({"epoch": epoch, "train_l1": train_l1, "val_l1": val_l1, "val_psnr": val_psnr,
                    "lr": opt.param_groups[0]["lr"], "seconds": secs})
        print(f"epoch {epoch:3d}/{epochs}  train L1 {train_l1:.4f}  val L1 {val_l1:.4f}  "
              f"val PSNR {val_psnr:6.2f} dB  {secs:5.1f} s{'  *best*' if is_best else ''}", flush=True)

    save_checkpoint(ckpt_dir / f"resunet_{tag}_last.pt", model, {**meta, "epoch": epochs, "val_psnr": log[-1]["val_psnr"]})
    log_df = pd.DataFrame(log)
    res_dir = cfg["paths"]["outputs"] / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(res_dir / f"training_log_{tag}.csv", index=False, lineterminator="\n")
    fig = save_training_curves(log_df, noisy_val_psnr, cfg["paths"]["outputs"] / "figures" / f"training_curves_{tag}.png")

    best_row = log_df.loc[log_df["val_psnr"].idxmax()]
    gain = best_row["val_psnr"] - noisy_val_psnr
    print(f"\nBest epoch {int(best_row['epoch'])}: val PSNR {best_row['val_psnr']:.2f} dB "
          f"(+{gain:.2f} dB over the noisy input). Total time {log_df['seconds'].sum() / 60:.1f} min")
    print(f"Saved: {ckpt_dir / f'resunet_{tag}_best.pt'}\n       {fig}")
    if len(log_df) >= 10:
        tail_gain = log_df["val_psnr"].iloc[-1] - log_df["val_psnr"].iloc[int(0.75 * len(log_df)) - 1]
        if tail_gain > 0.3:
            print(f"Note: validation PSNR rose {tail_gain:.2f} dB in the last quarter of training. "
                  "A longer run (e.g. --epochs 60) may help.")

    checks = {"loss stayed finite": bool(np.isfinite(log_df[["train_l1", "val_l1"]]).all().all())}
    print()
    if epochs * steps >= 1000:
        checks["validation PSNR at least 5 dB above the noisy input"] = gain >= 5
    else:
        print(f"[INFO] short run ({epochs * steps} steps): the 5 dB check only applies to full runs (1000+ steps)")
    for text, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {text}")
    return 0 if all(checks.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", help="noise level (default: config primary, medium)")
    parser.add_argument("--epochs", type=int, help="default from config")
    parser.add_argument("--batch-size", type=int, help="default from config")
    parser.add_argument("--lr", type=float, help="learning rate (default from config)")
    parser.add_argument("--limit", type=int, help="use only the first N train and val images")
    parser.add_argument("--seed", type=int, help="default from config")
    parser.add_argument("--overfit-one-batch", action="store_true", help="sanity check instead of training")
    parser.add_argument("--overfit-steps", type=int, default=300)
    parser.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is available")
    parser.add_argument("--config", help="config file (default configs/default.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    args.seed = cfg["seed"] if args.seed is None else args.seed
    args.lr = cfg["training"]["lr"] if args.lr is None else args.lr
    set_seed(args.seed)
    level = get_level(cfg, args.level)
    device = get_device(prefer_gpu=not args.cpu)
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Device: {name} | noise level {level.name} (lambda {level.lambda_photons:g}, "
          f"sigma {level.sigma_electronic:g}) | ResU-Net {count_parameters(ResUNet()):,} parameters\n")

    if args.overfit_one_batch:
        return overfit_one_batch(cfg, args, level, device)
    return train(cfg, args, level, device)


if __name__ == "__main__":
    sys.exit(main())
