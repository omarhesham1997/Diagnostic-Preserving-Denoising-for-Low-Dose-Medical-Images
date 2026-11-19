"""Report figures: qualitative comparisons and the pixel-vs-diagnosis chart.

Run (after evaluate.py has scored the methods):
    uv run python scripts/make_figures.py

Example images are chosen by fixed rules (not by hand), using the ResU-Net scores
(BM3D if the network has not been evaluated yet):
  1. typical:      the image with the median PSNR
  2. small finding: among images labelled Nodule or Mass (small findings are the most
                   at risk of being smoothed away), the one with the median PSNR
  3. worst case:   the image with the largest delta_p (biggest change in the judge's opinion)
The zoom box is placed where the clean image has the most local detail.

Writes outputs/figures/qualitative_<level>_{1_typical,2_small_finding,3_worst_case}.png
and outputs/figures/psnr_vs_delta_p_<level>.png
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from c21.config import load_config
from c21.data import load_clean, load_noisy, split_rows
from c21.evaluation import METHOD_ORDER, denoised_path, display_name, load_summaries, pick_examples, results_dir
from c21.noise import get_level
from c21.viz import save_qualitative, save_tradeoff_chart


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", help="noise level (default: config primary, medium)")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--config", help="config file (default configs/default.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    level = get_level(cfg, args.level)
    out_dir = results_dir(cfg, args.split, level.name)
    summary = load_summaries(out_dir)
    if summary.empty:
        print(f"No results in {out_dir}. Run scripts/evaluate.py first.")
        return 1
    methods = [m for m in METHOD_ORDER if m in set(summary["method"])
               and denoised_path(cfg, args.split, level.name, m).exists()]
    fig_dir = cfg["paths"]["outputs"] / "figures"

    clean = load_clean(cfg, args.split)
    rows = split_rows(cfg, args.split)
    outputs = {m: np.load(denoised_path(cfg, args.split, level.name, m)) for m in methods}
    if "noisy" not in outputs:
        outputs["noisy"] = np.clip(load_noisy(cfg, args.split, level.name), 0, 1)
        methods = ["noisy", *methods]
    per = {m: pd.read_csv(out_dir / f"{m}_per_image.csv") for m in methods if (out_dir / f"{m}_per_image.csv").exists()}
    ref = "resunet" if "resunet" in per else "bm3d"
    print(f"Methods: {', '.join(display_name(m) for m in methods)}. Examples chosen with {display_name(ref)} scores.\n")

    for tag, i, reason in pick_examples(per[ref], rows):
        name, findings = rows.loc[i, "image"], rows.loc[i, "finding_labels"]
        panels = [(f"Clean\n{name}\n{findings[:34]}", clean[i])]
        for m in methods:
            p = per[m].loc[i] if m in per else None
            label = display_name(m) if p is None else f"{display_name(m)}\nPSNR {p['psnr']:.1f} dB\n$\\Delta p$ {p['delta_p']:.3f}"
            panels.append((label, outputs[m][i]))
        reason_tex = reason.replace("delta_p", r"$\Delta p$")
        title = f"Example {tag[0]}: {reason_tex}; noise level {level.name}"
        path = save_qualitative(clean[i], panels, title, fig_dir / f"qualitative_{level.name}_{tag}.png")
        print(f"- {tag}: {name} ({findings}) -> {path.name}")

    ci = {m: (1.96 * per[m]["psnr"].std() / np.sqrt(len(per[m])), 1.96 * per[m]["delta_p"].std() / np.sqrt(len(per[m])))
          for m in per}
    chart = save_tradeoff_chart(summary[summary["method"].isin(per)], ci, fig_dir / f"psnr_vs_delta_p_{level.name}.png",
                                f"Pixel fidelity vs diagnostic change ({args.split} set, {level.name} noise; 95% CI)")
    print(f"- chart: {chart.name}")
    print(f"\nSaved in {fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
