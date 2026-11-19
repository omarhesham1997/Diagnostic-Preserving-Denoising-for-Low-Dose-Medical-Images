"""Shared evaluation: score one method, save its results, and build comparison tables.

Each method is scored on exactly the same fixed noisy images. Results are saved per
method, so methods can be evaluated at different times (for example the network in
Batch 6) and still end up in one table.

Files (under outputs/):
    denoised/{split}_{level}_{method}.npy         method outputs (float32, in [0, 1])
    predictions/{split}_{level}_{method}.npy      judge probabilities [N, 14]
    predictions/{split}_clean.npy                 judge probabilities on clean images
    results/{split}_{level}/{method}_per_image.csv
    results/{split}_{level}/{method}_auc.csv      per-disease and macro dAUC with 95% CI
    results/{split}_{level}/{method}_summary.json
    results/{split}_{level}/summary.csv, summary.tex, dauc.tex   (all methods together)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from c21.constants import NIH_LABELS
from c21.metrics import delta_auc, diagnosis_scores, eligible_diseases, pixel_scores

METHOD_ORDER = ["noisy", "median", "bm3d", "resunet"]
DISPLAY = {
    "noisy": ("Noisy (no denoising)", "Noisy (no denoising)"),
    "median": ("Median 3x3", r"Median $3\times3$"),
    "bm3d": ("Anscombe-BM3D", "Anscombe-BM3D"),
    "resunet": ("ResU-Net (ours)", "ResU-Net (ours)"),
}


def display_name(method: str, latex: bool = False) -> str:
    plain, tex = DISPLAY.get(method, (method, method))
    return tex if latex else plain


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def results_dir(cfg: dict, split: str, level: str) -> Path:
    return cfg["paths"]["outputs"] / "results" / f"{split}_{level}"


def denoised_path(cfg: dict, split: str, level: str, method: str) -> Path:
    return cfg["paths"]["outputs"] / "denoised" / f"{split}_{level}_{method}.npy"


def predictions_path(cfg: dict, split: str, level: str | None, method: str) -> Path:
    name = f"{split}_clean.npy" if method == "clean" else f"{split}_{level}_{method}.npy"
    return cfg["paths"]["outputs"] / "predictions" / name


# ---------------------------------------------------------------------------
# Scoring one method
# ---------------------------------------------------------------------------


def score_method(method: str, clean: np.ndarray, outputs: np.ndarray, p_clean: np.ndarray,
                 p_out: np.ndarray, labels: np.ndarray, image_names: list[str], cfg: dict,
                 seconds_per_image: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """All scores for one method. Returns (per-image table, dAUC table, summary dict)."""
    ccfg = cfg["classifier"]
    pix = pixel_scores(clean, outputs)
    diag = diagnosis_scores(p_clean, p_out, ccfg["threshold"])

    per_image = pd.DataFrame({
        "image": list(image_names), "method": method,
        "psnr": pix["psnr"], "ssim": pix["ssim"],
        "delta_p": diag["delta_p"], "agreement": diag["agreement"],
    })

    used = eligible_diseases(labels, ccfg["min_positives_for_auc"])
    dauc = delta_auc(labels, p_clean, p_out, used,
                     n_boot=ccfg.get("bootstrap_samples", 1000), seed=cfg["seed"])
    auc_rows = [{"method": method, "disease": NIH_LABELS[k], **v} for k, v in dauc["per_disease"].items()]
    auc_rows.append({"method": method, "disease": "macro", **dauc["macro"]})
    auc_df = pd.DataFrame(auc_rows)

    summary = {
        "method": method, "name": display_name(method), "n_images": len(per_image),
        "psnr_mean": float(pix["psnr"].mean()), "psnr_std": float(pix["psnr"].std()),
        "ssim_mean": float(pix["ssim"].mean()), "ssim_std": float(pix["ssim"].std()),
        "delta_p_mean": float(diag["delta_p"].mean()), "delta_p_std": float(diag["delta_p"].std()),
        "agreement_mean": float(diag["agreement"].mean()),
        "lost_findings": diag["lost_findings"], "new_findings": diag["new_findings"],
        "macro_dauc": dauc["macro"]["delta_auc"], "macro_dauc_ci_low": dauc["macro"]["ci_low"],
        "macro_dauc_ci_high": dauc["macro"]["ci_high"], "dauc_diseases": dauc["macro"]["n_diseases"],
        "seconds_per_image": seconds_per_image,
    }
    return per_image, auc_df, summary


def save_method_results(out_dir: Path, method: str, per_image: pd.DataFrame,
                        auc_df: pd.DataFrame, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(out_dir / f"{method}_per_image.csv", index=False, lineterminator="\n")
    auc_df.to_csv(out_dir / f"{method}_auc.csv", index=False, lineterminator="\n")
    (out_dir / f"{method}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _method_sort_key(method: str):
    return (METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER), method)


def load_summaries(out_dir: Path) -> pd.DataFrame:
    """All saved method summaries in this folder, in the standard method order."""
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in out_dir.glob("*_summary.json")]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.iloc[sorted(range(len(df)), key=lambda i: _method_sort_key(df.loc[i, "method"]))].reset_index(drop=True)


def load_auc_tables(out_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(out_dir.glob("*_auc.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def format_console_table(summary: pd.DataFrame) -> str:
    head = (f"{'method':22s}{'PSNR dB':>15s}{'SSIM':>15s}{'delta_p':>9s}{'agree':>7s}"
            f"{'lost':>7s}{'new':>7s}{'macro dAUC [95% CI]':>27s}{'s/img':>7s}")
    lines = [head, "-" * len(head)]
    for _, r in summary.iterrows():
        ci = ("n/a (too few cases)" if pd.isna(r["macro_dauc"]) else
              f"{r['macro_dauc']:+.3f} [{r['macro_dauc_ci_low']:+.3f}, {r['macro_dauc_ci_high']:+.3f}]")
        sec = "" if pd.isna(r.get("seconds_per_image")) else f"{r['seconds_per_image']:.3f}"
        lines.append(f"{r['name']:22s}{r['psnr_mean']:8.2f} +- {r['psnr_std']:4.2f}"
                     f"{r['ssim_mean']:8.3f} +-{r['ssim_std']:5.3f}{r['delta_p_mean']:9.4f}"
                     f"{r['agreement_mean']:7.3f}{r['lost_findings']:7.3f}{r['new_findings']:7.3f}"
                     f"{ci:>27s}{sec:>7s}")
    return "\n".join(lines)


def _tex_ci(value: float, low: float, high: float) -> str:
    """Signed value with 95% CI, using proper minus signs in LaTeX."""
    if pd.isna(value):
        return "n/a"
    return f"${value:+.3f}$ [${low:+.3f}$, ${high:+.3f}$]"


def summary_to_latex(summary: pd.DataFrame) -> str:
    """Main comparison table (tabular only; \\input it inside a table environment)."""
    lines = [
        r"\begin{tabular}{@{}lccccccc@{}}",
        r"\toprule",
        r"\textbf{Method} & \textbf{PSNR (dB)} $\uparrow$ & \textbf{SSIM} $\uparrow$ & "
        r"$\boldsymbol{\Delta p}$ $\downarrow$ & \textbf{Agr.} $\uparrow$ & "
        r"\textbf{Lost} $\downarrow$ & \textbf{New} $\downarrow$ & \textbf{Macro $\Delta$AUC} \\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"{display_name(r['method'], latex=True)} & "
            f"{r['psnr_mean']:.2f} $\\pm$ {r['psnr_std']:.2f} & {r['ssim_mean']:.3f} $\\pm$ {r['ssim_std']:.3f} & "
            f"{r['delta_p_mean']:.4f} & {r['agreement_mean']:.3f} & {r['lost_findings']:.3f} & "
            f"{r['new_findings']:.3f} & "
            f"{_tex_ci(r['macro_dauc'], r['macro_dauc_ci_low'], r['macro_dauc_ci_high'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def dauc_to_latex(auc_df: pd.DataFrame) -> str:
    """Per-disease dAUC with 95% CI: one row per disease, one column per method."""
    methods = sorted(auc_df["method"].unique(), key=_method_sort_key)
    diseases = [d for d in auc_df["disease"].unique() if d != "macro"] + ["macro"]
    cols = "l" + "c" * len(methods)
    lines = [f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}", r"\toprule",
             r"\textbf{Disease (cases)} & " + " & ".join(f"\\textbf{{{display_name(m, latex=True)}}}"
                                                            for m in methods) + r" \\",
             r"\midrule"]
    for d in diseases:
        part = auc_df[auc_df["disease"] == d].set_index("method")
        if d == "macro":
            lines.append(r"\midrule")
            label = r"\textit{Macro average}"
        else:
            n_pos = int(part["n_pos"].iloc[0])
            label = f"{d.replace('_', ' ')} ({n_pos})"
        cells = []
        for m in methods:
            if m in part.index:
                r = part.loc[m]
                cells.append(_tex_ci(r["delta_auc"], r["ci_low"], r["ci_high"]))
            else:
                cells.append("--")
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def write_combined_tables(out_dir: Path) -> pd.DataFrame:
    """Rebuild summary.csv, summary.tex and dauc.tex from every method saved in out_dir."""
    summary = load_summaries(out_dir)
    if summary.empty:
        return summary
    summary.to_csv(out_dir / "summary.csv", index=False, lineterminator="\n")
    (out_dir / "summary.tex").write_text(summary_to_latex(summary), encoding="utf-8")
    auc_df = load_auc_tables(out_dir)
    if not auc_df.empty:
        (out_dir / "dauc.tex").write_text(dauc_to_latex(auc_df), encoding="utf-8")
    return summary


def sanity_checks(summary: pd.DataFrame) -> dict[str, bool]:
    """Basic expectations for the classical baselines (only checks methods that are present)."""
    s = summary.set_index("method")
    checks = {}
    if "noisy" in s.index:
        for m in s.index.drop("noisy"):
            checks[f"{display_name(m)} PSNR above Noisy"] = s.loc[m, "psnr_mean"] > s.loc["noisy", "psnr_mean"]
        if "bm3d" in s.index:
            gain = s.loc["bm3d", "psnr_mean"] - s.loc["noisy", "psnr_mean"]
            checks[f"BM3D gains 5+ dB over Noisy (gain {gain:.1f} dB)"] = gain >= 5
    if "bm3d" in s.index and "median" in s.index:
        checks["BM3D PSNR above Median"] = s.loc["bm3d", "psnr_mean"] > s.loc["median", "psnr_mean"]
    return checks


# ---------------------------------------------------------------------------
# Paired comparison of two methods (same images, same noise)
# ---------------------------------------------------------------------------


def paired_comparison(per_a: pd.DataFrame, per_b: pd.DataFrame, labels: np.ndarray, p_clean: np.ndarray,
                      p_a: np.ndarray, p_b: np.ndarray, diseases: list[int], threshold: float = 0.5,
                      n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Method A vs method B on the same images: mean difference (A - B), paired bootstrap 95% CI,
    Wilcoxon signed-rank p-value (per-image metrics), and the share of images where A is better.

    `better` says which method wins when the CI excludes 0 ("no clear difference" otherwise).
    """
    from scipy.stats import wilcoxon

    from c21.metrics import auc

    if list(per_a["image"]) != list(per_b["image"]):
        raise ValueError("per-image tables are not in the same image order")
    n = len(per_a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    name_a, name_b = per_a["method"].iloc[0], per_b["method"].iloc[0]
    rows = []

    def add(metric, a_val, b_val, boot_diffs, higher_is_better, p_value=np.nan, a_better_share=np.nan):
        lo, hi = np.nanpercentile(boot_diffs, [2.5, 97.5])
        diff = a_val - b_val
        if lo > 0 or hi < 0:
            a_wins = (diff > 0) == higher_is_better
            better = display_name(name_a if a_wins else name_b)
        else:
            better = "no clear difference"
        rows.append({"metric": metric, "a": a_val, "b": b_val, "diff": diff, "ci_low": lo, "ci_high": hi,
                     "p_value": p_value, "a_better_share": a_better_share,
                     "higher_is_better": higher_is_better, "better": better})

    for col, label, higher in [("psnr", "PSNR (dB)", True), ("ssim", "SSIM", True),
                               ("delta_p", "delta_p", False), ("agreement", "Agreement", True)]:
        a, b = per_a[col].to_numpy(), per_b[col].to_numpy()
        d = a - b
        try:
            p = float(wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0
        except ValueError:
            p = np.nan
        share = float(np.mean(d > 0) if higher else np.mean(d < 0))
        add(label, float(a.mean()), float(b.mean()), d[idx].mean(axis=1), higher, p, share)

    # Pooled lost / new findings (lower is better)
    yes_clean = p_clean >= threshold
    for label, mask_fn in [("Lost findings", lambda p: yes_clean & (p < threshold)),
                           ("New findings", lambda p: ~yes_clean & (p >= threshold))]:
        base = yes_clean.sum(axis=1) if label == "Lost findings" else (~yes_clean).sum(axis=1)
        ca, cb = mask_fn(p_a).sum(axis=1), mask_fn(p_b).sum(axis=1)
        rate = lambda c, i: c[i].sum(axis=-1) / np.maximum(base[i].sum(axis=-1), 1)  # noqa: E731
        add(label, float(ca.sum() / base.sum()), float(cb.sum() / base.sum()),
            rate(ca, idx) - rate(cb, idx), higher_is_better=False)

    # Macro AUC against the doctor labels (higher is better)
    if diseases:
        def macro_auc(p, i=slice(None)):
            return np.nanmean([auc(labels[i, k], p[i, k]) for k in diseases])

        boot = np.array([macro_auc(p_a, i) - macro_auc(p_b, i) for i in idx])
        add("Macro AUC", float(macro_auc(p_a)), float(macro_auc(p_b)), boot, higher_is_better=True)

    out = pd.DataFrame(rows)
    out.insert(0, "method_b", name_b)
    out.insert(0, "method_a", name_a)
    return out


def comparison_to_latex(df: pd.DataFrame) -> str:
    name_a = display_name(df["method_a"].iloc[0], latex=True)
    name_b = display_name(df["method_b"].iloc[0], latex=True)
    fmt = {"PSNR (dB)": "{:.2f}", "SSIM": "{:.3f}", "delta_p": "{:.4f}", "Agreement": "{:.3f}",
           "Lost findings": "{:.3f}", "New findings": "{:.3f}", "Macro AUC": "{:.3f}"}
    tex_name = {"delta_p": r"$\Delta p$", "PSNR (dB)": "PSNR (dB)"}
    lines = [r"\begin{tabular}{@{}lccccl@{}}", r"\toprule",
             f"\\textbf{{Metric}} & \\textbf{{{name_a}}} & \\textbf{{{name_b}}} & "
             r"\textbf{Difference [95\% CI]} & \textbf{Wilcoxon $p$} & \textbf{Better} \\", r"\midrule"]
    for _, r in df.iterrows():
        f = fmt.get(r["metric"], "{:.3f}")
        arrow = r"$\uparrow$" if r["higher_is_better"] else r"$\downarrow$"
        p = "--" if pd.isna(r["p_value"]) else ("$<0.001$" if r["p_value"] < 0.001 else f"{r['p_value']:.3f}")
        diff = f"${r['diff']:+.3f}$ [${r['ci_low']:+.3f}$, ${r['ci_high']:+.3f}$]"
        better = r["better"].replace("ResU-Net (ours)", "ResU-Net")
        lines.append(f"{tex_name.get(r['metric'], r['metric'])} {arrow} & {f.format(r['a'])} & {f.format(r['b'])} & "
                     f"{diff} & {p} & {better} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Example selection for figures
# ---------------------------------------------------------------------------


def pick_examples(per_ref: pd.DataFrame, rows: pd.DataFrame) -> list[tuple[str, int, str]]:
    """Return [(tag, index, reason)] for the three example images (distinct indices)."""
    chosen = []

    def median_index(idx):
        idx = list(idx)
        order = per_ref.loc[idx, "psnr"].sort_values().index
        return int(order[len(order) // 2])

    chosen.append(("1_typical", median_index(per_ref.index), "median PSNR over the test set"))
    small = rows.index[(rows["Nodule"] == 1) | (rows["Mass"] == 1)]
    small = [i for i in small if i not in {c[1] for c in chosen}]
    if small:
        chosen.append(("2_small_finding", median_index(small), "median PSNR among Nodule/Mass images"))
    rest = per_ref.drop(index=[c[1] for c in chosen])
    chosen.append(("3_worst_case", int(rest["delta_p"].idxmax()), "largest delta_p, the worst case"))
    return chosen
