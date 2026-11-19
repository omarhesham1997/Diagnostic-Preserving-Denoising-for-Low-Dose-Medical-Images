"""Turn experiment outputs into report assets: LaTeX tables and number macros.

Every number quoted in the report text is a macro defined in generated/numbers.tex
(for example \\OursPSNR), written by scripts/export_report_assets.py from the saved
results. Re-running the experiments and the export updates the report consistently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from c21.constants import NIH_LABELS

PREFIX = {"noisy": "Noisy", "median": "Median", "bm3d": "BM", "resunet": "Ours"}


# ---------------------------------------------------------------------------
# Formatting helpers (all return LaTeX-ready text)
# ---------------------------------------------------------------------------


def num(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}"


def signed(x: float, d: int = 3) -> str:
    return f"${x:+.{d}f}$"


def ci(lo: float, hi: float, d: int = 3) -> str:
    return f"[${lo:+.{d}f}$, ${hi:+.{d}f}$]"


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}\\%"


def integer(n: int) -> str:
    return f"{int(n):,}".replace(",", "{,}")


def macro(name: str, value: str) -> str:
    if not name.isalpha():
        raise ValueError(f"LaTeX macro names may only contain letters: {name}")
    return f"\\newcommand{{\\{name}}}{{{value}}}"


# ---------------------------------------------------------------------------
# Analyses computed from saved predictions
# ---------------------------------------------------------------------------


def findings_by_disease(p_clean: np.ndarray, preds: dict[str, np.ndarray], threshold: float = 0.5) -> pd.DataFrame:
    """Per disease: how many findings the judge sees on clean images, and the share lost / new per method."""
    yes = p_clean >= threshold
    rows = []
    for k, name in enumerate(NIH_LABELS):
        row = {"disease": name, "clean_findings": int(yes[:, k].sum())}
        n_pos, n_neg = max(int(yes[:, k].sum()), 1), max(int((~yes[:, k]).sum()), 1)
        for m, p in preds.items():
            out = p[:, k] >= threshold
            row[f"{m}_lost"] = float((yes[:, k] & ~out).sum() / n_pos)
            row[f"{m}_new"] = float((~yes[:, k] & out).sum() / n_neg)
        rows.append(row)
    return pd.DataFrame(rows)


def probability_shift(p_clean: np.ndarray, p_out: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Mean signed change of the judge's probabilities: overall, on its clean findings, on non-findings."""
    d = p_out - p_clean
    yes = p_clean >= threshold
    return {"all": float(d.mean()), "findings": float(d[yes].mean()), "non_findings": float(d[~yes].mean())}


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _tabular(cols: str, header: list[str], rows: list[list[str]], midrule_before: set[int] = frozenset()) -> str:
    lines = [f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if i in midrule_before:
            lines.append(r"\midrule")
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def findings_table_tex(df: pd.DataFrame, methods: tuple[str, str] = ("bm3d", "resunet"),
                       names: tuple[str, str] = ("BM3D", "ResU-Net")) -> str:
    """Lost / new findings per disease for two methods; bold marks the lower (better) value."""
    a, b = methods
    lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"& \textbf{Clean} & \multicolumn{2}{c}{\textbf{Lost findings}} & \multicolumn{2}{c}{\textbf{New findings}} \\",
             r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
             f"\\textbf{{Disease}} & \\textbf{{findings}} & {names[0]} & {names[1]} & {names[0]} & {names[1]} \\\\",
             r"\midrule"]
    for _, r in df.sort_values("clean_findings", ascending=False).iterrows():
        cells = [r["disease"].replace("_", " "), integer(r["clean_findings"])]
        for kind in ("lost", "new"):
            va, vb = r[f"{a}_{kind}"], r[f"{b}_{kind}"]
            cells += [f"\\textbf{{{pct(va, 0)}}}" if va < vb else pct(va, 0),
                      f"\\textbf{{{pct(vb, 0)}}}" if vb < va else pct(vb, 0)]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def noise_levels_tex(df: pd.DataFrame) -> str:
    rows = [[r["level"].capitalize(), f"{r['lambda']:g}", f"{r['sigma']:g}",
             f"{r['psnr_mean']:.2f} $\\pm$ {r['psnr_std']:.2f}", f"{r['ssim_mean']:.3f} $\\pm$ {r['ssim_std']:.3f}"]
            for _, r in df.iterrows()]
    return _tabular("lcccc", [r"\textbf{Level}", r"$\boldsymbol{\lambda}$", r"$\boldsymbol{\sigma}$",
                              r"\textbf{PSNR (dB)}", r"\textbf{SSIM}"], rows)


def judge_levels_tex(df: pd.DataFrame) -> str:
    rows = [[r["level"].capitalize(), f"{r['delta_p']:.3f}", f"{r['agreement']:.3f}", pct(r["lost_findings"]),
             pct(r["new_findings"]), f"{signed(r['macro_dauc'])} {ci(r['ci_low'], r['ci_high'])}"]
            for _, r in df.iterrows()]
    return _tabular("lccccc", [r"\textbf{Noise level}", r"$\boldsymbol{\Delta p}$", r"\textbf{Agr.}",
                               r"\textbf{Lost}", r"\textbf{New}", r"\textbf{Macro $\Delta$AUC [95\% CI]}"], rows)


def judge_auc_tex(df: pd.DataFrame) -> str:
    rows = []
    for _, r in df.sort_values("cases", ascending=False).iterrows():
        a = "--" if pd.isna(r["auc_clean"]) else f"{r['auc_clean']:.3f}"
        rows.append([r["disease"].replace("_", " "), str(int(r["cases"])), a, "yes" if r["used_for_dauc"] else "no"])
    return _tabular("lrcc", [r"\textbf{Disease}", r"\textbf{Test cases}", r"\textbf{AUC (clean)}",
                             r"\textbf{Used for $\Delta$AUC}"], rows)


def dataset_tex(split: pd.DataFrame, diseases: list[str]) -> str:
    order = ["train", "val", "test"]
    header = [r"\textbf{Split}", r"\textbf{Images}", r"\textbf{Patients}", r"\textbf{No Finding}"] + \
             [f"\\textbf{{{d.replace('_', ' ')}}}" for d in diseases]
    rows = []
    for s in order:
        part = split[split["split"] == s]
        rows.append([s.capitalize(), integer(len(part)), integer(part["patient_id"].nunique()),
                     pct(part["no_finding"].mean())] + [str(int(part[d].sum())) for d in diseases])
    rows.append([r"\textit{Total}", integer(len(split)), integer(split["patient_id"].nunique()),
                 pct(split["no_finding"].mean())] + [str(int(split[d].sum())) for d in diseases])
    return _tabular("l" + "r" * (3 + len(diseases)), header, rows, midrule_before={3})


def join_names(names: list[str]) -> str:
    names = [n.replace("_", " ") for n in names]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
