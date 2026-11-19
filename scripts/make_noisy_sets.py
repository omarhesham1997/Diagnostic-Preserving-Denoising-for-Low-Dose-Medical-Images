"""Create the fixed noisy validation and test sets for every noise level.

Run:
    uv run python scripts/make_noisy_sets.py --limit 20   # quick check, nothing saved
    uv run python scripts/make_noisy_sets.py

Writes data/processed/noisy/{val,test}_{low,medium,high}.npy (float32, not clipped),
outputs/results/noise_levels_test.csv and outputs/figures/noise_levels.png.
Image i of a split always gets seed (split seed base + i), so re-running gives
identical files (compare the printed checksums).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from c21.config import load_config
from c21.data import array_checksum, load_clean, noisy_array_path
from c21.noise import all_levels, make_noisy_set
from c21.viz import save_panels_with_zoom


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="config file (default configs/default.yaml)")
    parser.add_argument("--limit", type=int, help="only the first N images of each split, nothing saved")
    args = parser.parse_args()

    cfg = load_config(args.config)
    levels = all_levels(cfg)
    seed_bases = {"val": cfg["noise"]["val_seed_base"], "test": cfg["noise"]["test_seed_base"]}

    print(f"{'split':6s}{'level':8s}{'images':>8s}  checksum")
    test_noisy = {}
    for split, base in seed_bases.items():
        clean = load_clean(cfg, split)
        if args.limit:
            clean = clean[: args.limit]
        for lv in levels:
            noisy = make_noisy_set(clean, lv, base)
            status = "(not saved: --limit)"
            if not args.limit:
                out = noisy_array_path(cfg, split, lv.name)
                out.parent.mkdir(parents=True, exist_ok=True)
                np.save(out, noisy)
                status = array_checksum(noisy)
            print(f"{split:6s}{lv.name:8s}{len(noisy):8d}  {status}")
            if split == "test":
                test_noisy[lv.name] = noisy
        if split == "test":
            test_clean = clean

    # How noisy is each level on the real test images? (noisy input clipped to [0, 1] for display/metrics)
    rows = []
    for lv in levels:
        noisy = np.clip(test_noisy[lv.name], 0, 1)
        psnr = [peak_signal_noise_ratio(c, n, data_range=1.0) for c, n in zip(test_clean, noisy)]
        ssim = [structural_similarity(c, n, data_range=1.0) for c, n in zip(test_clean, noisy)]
        rows.append({"level": lv.name, "lambda": lv.lambda_photons, "sigma": lv.sigma_electronic,
                     "psnr_mean": np.mean(psnr), "psnr_std": np.std(psnr),
                     "ssim_mean": np.mean(ssim), "ssim_std": np.std(ssim), "n_images": len(psnr)})
    table = pd.DataFrame(rows)

    print(f"\nNoisy test images vs clean ({len(test_clean)} images):")
    print(f"{'level':8s}{'lambda':>8s}{'sigma':>7s}{'PSNR dB':>16s}{'SSIM':>16s}")
    for r in rows:
        print(f"{r['level']:8s}{r['lambda']:8.0f}{r['sigma']:7.2f}"
              f"{r['psnr_mean']:9.2f} +- {r['psnr_std']:4.2f}{r['ssim_mean']:9.3f} +- {r['ssim_std']:5.3f}")

    if not args.limit:
        out_csv = cfg["paths"]["outputs"] / "results" / "noise_levels_test.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_csv, index=False, lineterminator="\n")
        print(f"\nSaved table: {out_csv}")

    panels = [("clean", test_clean[0])] + [
        (f"{r['level']} ({peak_signal_noise_ratio(test_clean[0], np.clip(test_noisy[r['level']][0], 0, 1), data_range=1.0):.1f} dB)",
         test_noisy[r["level"]][0]) for r in rows
    ]
    fig = save_panels_with_zoom(panels, cfg["paths"]["outputs"] / "figures" / "noise_levels.png")
    print(f"Saved figure: {fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
