"""Paired comparison of two evaluated methods (default: ResU-Net vs Anscombe-BM3D).

Run (after evaluate.py has scored both methods):
    uv run python scripts/compare.py
    uv run python scripts/compare.py --a bm3d --b median

For every metric: mean of A, mean of B, difference A - B with a paired bootstrap 95%
confidence interval, Wilcoxon signed-rank p-value (per-image metrics), and which method
is better when the interval excludes 0. Both methods saw exactly the same noisy images,
so the comparison is paired.

Writes outputs/results/<split>_<level>/compare_<a>_vs_<b>.csv and .tex
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from c21.config import load_config
from c21.data import load_label_matrix
from c21.evaluation import comparison_to_latex, display_name, paired_comparison, predictions_path, results_dir
from c21.metrics import eligible_diseases
from c21.noise import get_level


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", default="resunet")
    parser.add_argument("--b", default="bm3d")
    parser.add_argument("--level", help="noise level (default: config primary, medium)")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--config", help="config file (default configs/default.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    level = get_level(cfg, args.level)
    out_dir = results_dir(cfg, args.split, level.name)
    missing = [m for m in (args.a, args.b) if not (out_dir / f"{m}_per_image.csv").exists()]
    if missing:
        print(f"No saved results for {missing} at level {level.name}. Run scripts/evaluate.py --methods {' '.join(missing)}")
        return 1

    per_a = pd.read_csv(out_dir / f"{args.a}_per_image.csv")
    per_b = pd.read_csv(out_dir / f"{args.b}_per_image.csv")
    labels = load_label_matrix(cfg, args.split)
    p_clean = np.load(predictions_path(cfg, args.split, None, "clean"))
    p_a = np.load(predictions_path(cfg, args.split, level.name, args.a))
    p_b = np.load(predictions_path(cfg, args.split, level.name, args.b))
    ccfg = cfg["classifier"]
    diseases = eligible_diseases(labels, ccfg["min_positives_for_auc"])

    df = paired_comparison(per_a, per_b, labels, p_clean, p_a, p_b, diseases, ccfg["threshold"],
                           n_boot=ccfg.get("bootstrap_samples", 1000), seed=cfg["seed"])

    name_a, name_b = display_name(args.a), display_name(args.b)
    print(f"{name_a} (A) vs {name_b} (B): {args.split} set, noise level {level.name}, {len(per_a)} images\n")
    print(f"{'metric':15s}{'A':>9s}{'B':>9s}{'A - B [95% CI]':>30s}{'Wilcoxon p':>12s}{'A better on':>13s}  better")
    for _, r in df.iterrows():
        p = "" if pd.isna(r["p_value"]) else ("<0.001" if r["p_value"] < 0.001 else f"{r['p_value']:.3f}")
        share = "" if pd.isna(r["a_better_share"]) else f"{100 * r['a_better_share']:.0f}% imgs"
        ci = f"{r['diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        print(f"{r['metric']:15s}{r['a']:9.4f}{r['b']:9.4f}{ci:>30s}{p:>12s}{share:>13s}  {r['better']}")

    stem = f"compare_{args.a}_vs_{args.b}"
    df.to_csv(out_dir / f"{stem}.csv", index=False, lineterminator="\n")
    (out_dir / f"{stem}.tex").write_text(comparison_to_latex(df), encoding="utf-8")
    print(f"\nSaved: {out_dir / (stem + '.csv')} and .tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
