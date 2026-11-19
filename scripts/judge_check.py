"""Batch 3 check: is the frozen classifier ("the judge") wired up correctly?

Run:
    uv run python scripts/judge_check.py --limit 50   # quick
    uv run python scripts/judge_check.py

1) AUC of the judge on the CLEAN test images, per disease. Most should be well above
   0.5; values near 0.5 everywhere mean the input preparation is wrong.
2) Sanity: clean vs the same clean images must give delta_p 0 and agreement 1.
3) Noisy images (no denoising) vs clean, per noise level. Damage must grow with noise:
   lost findings and macro dAUC rise from low to high. delta_p does NOT have to rise:
   the judge is so sensitive to noise that delta_p already levels off (about 0.3)
   at our weakest level.
Saves the clean-image predictions to outputs/predictions/test_clean.npy (reused later).
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from c21.classifier import FrozenClassifier
from c21.config import load_config
from c21.constants import NIH_LABELS
from c21.data import load_clean, load_label_matrix, load_noisy
from c21.metrics import auc, delta_auc, diagnosis_scores, eligible_diseases
from c21.noise import all_levels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="config file (default configs/default.yaml)")
    parser.add_argument("--limit", type=int, help="only the first N test images, nothing saved")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ccfg = cfg["classifier"]
    clean = load_clean(cfg, "test")
    labels = load_label_matrix(cfg, "test")
    if args.limit:
        clean, labels = clean[: args.limit], labels[: args.limit]

    judge = FrozenClassifier.from_config(cfg)
    print(f"Judge: {judge.weights} on {judge.device}, {len(clean)} test images")
    t0 = time.perf_counter()
    p_clean = judge.predict(clean)
    print(f"Prediction time: {time.perf_counter() - t0:.1f} s\n")

    # 1) AUC on clean images
    min_pos = ccfg["min_positives_for_auc"]
    used = eligible_diseases(labels, min_pos)
    print(f"1) Judge AUC on clean test images (dAUC uses diseases with {min_pos}+ cases)")
    print(f"   {'disease':20s}{'cases':>7s}{'AUC':>8s}")
    aucs = []
    for k, name in enumerate(NIH_LABELS):
        a = auc(labels[:, k], p_clean[:, k])
        n_pos = int(labels[:, k].sum())
        if not np.isnan(a):
            aucs.append(a)
        note = "used for dAUC" if k in used else "too few cases"
        a_txt = "   n/a" if np.isnan(a) else f"{a:6.3f}"
        print(f"   {name:20s}{n_pos:7d}  {a_txt}   {note}")
    median_auc = float(np.median(aucs)) if aucs else float("nan")
    print(f"   median AUC: {median_auc:.3f}")

    # 2) Sanity check
    same = diagnosis_scores(p_clean, p_clean, ccfg["threshold"])
    print("\n2) Sanity, clean vs clean: "
          f"delta_p {same['delta_p'].mean():.4f}, agreement {same['agreement'].mean():.3f}, "
          f"lost {same['lost_findings']:.3f}, new {same['new_findings']:.3f}")
    print(f"   Findings the judge sees on clean images: {same['n_clean_findings']} "
          f"(out of {p_clean.size} image-disease pairs)")

    # 3) Noisy images, no denoising
    print("\n3) Noisy images (no denoising) vs clean")
    print(f"   {'level':8s}{'delta_p':>9s}{'agree':>8s}{'lost':>7s}{'new':>7s}{'macro dAUC [95% CI]':>28s}")
    dps, losts, daucs = [], [], []
    for lv in all_levels(cfg):
        noisy = load_noisy(cfg, "test", lv.name)[: len(clean)]
        p_noisy = judge.predict(noisy)
        d = diagnosis_scores(p_clean, p_noisy, ccfg["threshold"])
        m = delta_auc(labels, p_clean, p_noisy, used, n_boot=ccfg.get("bootstrap_samples", 1000))["macro"]
        dps.append(d["delta_p"].mean())
        losts.append(d["lost_findings"])
        daucs.append(m["delta_auc"])
        ci = f"{m['delta_auc']:+.3f} [{m['ci_low']:+.3f}, {m['ci_high']:+.3f}]" if used else "n/a"
        print(f"   {lv.name:8s}{d['delta_p'].mean():9.4f}{d['agreement'].mean():8.3f}"
              f"{d['lost_findings']:7.3f}{d['new_findings']:7.3f}{ci:>28s}")

    print("   (delta_p levels off at these noise levels; lost findings and dAUC show the growing damage)")

    # Verdict
    increasing = lambda v: all(a < b for a, b in zip(v, v[1:]))  # noqa: E731
    checks = {
        "clean vs clean is exactly unchanged": same["delta_p"].mean() == 0 and same["agreement"].mean() == 1,
        "median clean AUC above 0.6": median_auc > 0.6,
        "noise changes the judge's output (delta_p above 0.05 at every level)": min(dps) > 0.05,
        "lost findings grow with noise (low < medium < high)": increasing(losts),
    }
    if used:
        checks["macro dAUC grows with noise (low < medium < high)"] = increasing(daucs)
    print()
    for text, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {text}")

    if not args.limit:
        out = cfg["paths"]["outputs"] / "predictions" / "test_clean.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, p_clean)
        print(f"\nSaved clean-image predictions: {out}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
