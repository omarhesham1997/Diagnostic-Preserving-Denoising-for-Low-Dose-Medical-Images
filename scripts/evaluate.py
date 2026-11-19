"""Evaluate denoising methods on the fixed noisy test set: pixel + diagnosis scores.

Run:
    uv run python scripts/evaluate.py --limit 20                    # quick, nothing saved
    uv run python scripts/evaluate.py --methods noisy median bm3d resunet --jobs 4
    uv run python scripts/evaluate.py --methods resunet             # add only the network

Methods: noisy, median, bm3d, resunet. The network uses
outputs/checkpoints/resunet_<level>_best.pt (or --checkpoint PATH).

Outputs are cached under outputs/denoised/; a second run reuses them (use --recompute
to force). The network is recomputed automatically when its checkpoint changes.
Each method's results are saved separately and all methods evaluated so far are
combined into outputs/results/<split>_<level>/summary.csv, summary.tex and dauc.tex.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from c21.baselines import CLASSICAL_METHODS, median_denoise
from c21.classifier import FrozenClassifier
from c21.config import load_config
from c21.data import load_clean, load_label_matrix, load_noisy, split_rows
from c21.evaluation import (
    denoised_path,
    display_name,
    format_console_table,
    load_summaries,
    predictions_path,
    results_dir,
    sanity_checks,
    save_method_results,
    score_method,
    write_combined_tables,
)
from c21.noise import get_level

ALL_METHODS = [*CLASSICAL_METHODS, "resunet"]


def run_classical(name: str, noisy: np.ndarray, level, cfg: dict, jobs: int) -> tuple[np.ndarray, float, str]:
    """Run one classical method on every image. Returns (outputs, wall seconds per image, hardware)."""
    fn = CLASSICAL_METHODS[name][0]
    if name == "median":
        fn = functools.partial(median_denoise, size=cfg["baselines"]["median_size"])
    n_jobs = jobs if name == "bm3d" else 1
    t0 = time.perf_counter()
    results = Parallel(n_jobs=n_jobs, return_as="generator")(delayed(fn)(y, level) for y in noisy)
    outs = list(tqdm(results, total=len(noisy), desc=display_name(name), ascii=True, leave=False))
    wall = time.perf_counter() - t0
    return np.stack(outs).astype(np.float32), wall / len(noisy), f"CPU, {n_jobs} worker(s)"


def checkpoint_path(cfg: dict, level_name: str, override: str | None) -> Path:
    return Path(override) if override else cfg["paths"]["outputs"] / "checkpoints" / f"resunet_{level_name}_best.pt"


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def run_resunet(noisy: np.ndarray, level, ckpt_path: Path) -> tuple[np.ndarray, float, str, dict]:
    """Denoise with the trained network. Returns (outputs, wall seconds per image, hardware, checkpoint info)."""
    import torch

    from c21.models.resunet import denoise_numpy, load_checkpoint
    from c21.utils import get_device

    device = get_device()
    model, ckpt = load_checkpoint(ckpt_path, device)
    if ckpt.get("level") not in (None, level.name):
        print(f"  WARNING: this checkpoint was trained at '{ckpt.get('level')}' noise, "
              f"but you are evaluating at '{level.name}'.")
    denoise_numpy(model, noisy[:16], device)  # warm-up, so timing excludes GPU start-up
    t0 = time.perf_counter()
    outputs = denoise_numpy(model, noisy, device, batch_size=16)
    wall = time.perf_counter() - t0
    hardware = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    info = {"checkpoint": Path(ckpt_path).name, "checkpoint_sha": file_sha(ckpt_path),
            "checkpoint_epoch": ckpt.get("epoch"), "trained_level": ckpt.get("level"),
            "checkpoint_val_psnr": ckpt.get("val_psnr")}
    return outputs, wall / len(noisy), hardware, info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS)
    parser.add_argument("--level", help="noise level (default: config primary, medium)")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--limit", type=int, help="only the first N images, nothing saved")
    parser.add_argument("--jobs", type=int, default=4, help="parallel workers for BM3D (default 4)")
    parser.add_argument("--checkpoint", help="network checkpoint (default outputs/checkpoints/resunet_<level>_best.pt)")
    parser.add_argument("--recompute", action="store_true", help="ignore cached outputs")
    parser.add_argument("--config", help="config file (default configs/default.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    level = get_level(cfg, args.level)
    split = args.split
    unknown = [m for m in args.methods if m not in ALL_METHODS]
    if unknown:
        print(f"Unknown method(s): {unknown}. Available: {ALL_METHODS}")
        return 1
    ckpt = checkpoint_path(cfg, level.name, args.checkpoint)
    if "resunet" in args.methods and not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}\nTrain first (scripts/train.py --level {level.name}) or pass --checkpoint.")
        return 1

    clean = load_clean(cfg, split)
    noisy = load_noisy(cfg, split, level.name)
    labels = load_label_matrix(cfg, split)
    names = split_rows(cfg, split)["image"].tolist()
    if args.limit:
        clean, noisy, labels, names = clean[: args.limit], noisy[: args.limit], labels[: args.limit], names[: args.limit]
    save = not args.limit
    out_dir = results_dir(cfg, split, level.name)
    print(f"Evaluating on {split}, noise level {level.name} (lambda {level.lambda_photons:g}, "
          f"sigma {level.sigma_electronic:g}), {len(clean)} images{'' if save else ' (quick run, nothing saved)'}")

    judge = FrozenClassifier.from_config(cfg)
    p_clean_path = predictions_path(cfg, split, None, "clean")
    if save and p_clean_path.exists() and len(np.load(p_clean_path)) == len(clean):
        p_clean = np.load(p_clean_path)
    else:
        p_clean = judge.predict(clean)
        if save:
            p_clean_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(p_clean_path, p_clean)

    summaries = []
    for method in args.methods:
        cache = denoised_path(cfg, split, level.name, method)
        old_path = out_dir / f"{method}_summary.json"
        old = json.loads(old_path.read_text(encoding="utf-8")) if old_path.exists() else None
        reuse = save and not args.recompute and cache.exists() and old is not None
        if reuse and method == "resunet" and old.get("checkpoint_sha") != file_sha(ckpt):
            reuse = False
            print("- ResU-Net: checkpoint changed since the last evaluation, recomputing")
        extra = {}
        if reuse:
            outputs = np.load(cache)
            secs = old.get("seconds_per_image")
            hardware = old.get("hardware") or f"CPU, {old.get('workers', 1)} worker(s)"
            extra = {k: old[k] for k in old if k.startswith("checkpoint") or k == "trained_level"}
            print(f"- {display_name(method)}: reusing cached outputs")
        elif method == "resunet":
            outputs, secs, hardware, extra = run_resunet(noisy, level, ckpt)
            print(f"- {display_name(method)}: {secs * len(noisy):.1f} s total, {secs * 1000:.1f} ms per image "
                  f"({hardware}; checkpoint epoch {extra['checkpoint_epoch']})")
        else:
            outputs, secs, hardware = run_classical(method, noisy, level, cfg, args.jobs)
            print(f"- {display_name(method)}: {secs * len(noisy):.1f} s total, {secs:.3f} s per image ({hardware})")

        p_out = judge.predict(outputs)
        per_image, auc_df, summary = score_method(method, clean, outputs, p_clean, p_out, labels, names, cfg, secs)
        summary.update({"hardware": hardware, **extra})
        summaries.append(summary)
        if save:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, outputs)
            pred_path = predictions_path(cfg, split, level.name, method)
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(pred_path, p_out)
            save_method_results(out_dir, method, per_image, auc_df, summary)

    if save:
        table = write_combined_tables(out_dir)
    else:
        import pandas as pd

        table = pd.DataFrame(summaries)
    print("\n" + format_console_table(table))
    print("\n(PSNR/SSIM/delta_p: mean +- std over images. lost/new: share of judge findings lost/added.\n"
          " macro dAUC: AUC(clean) - AUC(method), averaged over diseases with enough cases; closer to 0 is better.\n"
          " s/img: wall-clock seconds per image; BM3D runs on the CPU, the network on the GPU.)")

    print()
    checks = sanity_checks(table)
    for text, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {text}")
    if save:
        print(f"\nSaved tables: {out_dir / 'summary.csv'}, summary.tex, dauc.tex")
        print(f"Methods in the table so far: {', '.join(load_summaries(out_dir)['method'])}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
