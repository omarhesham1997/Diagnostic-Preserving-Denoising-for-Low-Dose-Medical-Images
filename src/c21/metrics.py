"""All scores used in the project.

Pixel scores (compare a method's output with the clean image):
    PSNR (dB, higher is better), SSIM (0..1, higher is better)

Diagnosis scores (compare the frozen classifier's outputs on clean vs processed images):
    delta_p        mean |p_out - p_clean| over the 14 diseases          (lower is better)
    agreement      share of yes/no answers (p >= threshold) unchanged  (higher is better)
    lost_findings  of the diseases the judge saw on the clean image, share it no longer sees
    new_findings   of the diseases it did not see on the clean image, share it now sees
    delta_auc      AUC(clean) - AUC(processed) against the doctor labels, per disease,
                   with a paired bootstrap 95% confidence interval (close to 0 is better)

Every method's output is clipped to [0, 1] before any score is computed.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from skimage.metrics import structural_similarity

PSNR_CAP = 100.0  # PSNR of identical images is infinite; cap it so averages stay finite


# ---------------------------------------------------------------------------
# Pixel scores
# ---------------------------------------------------------------------------


def psnr(clean: np.ndarray, test: np.ndarray) -> float:
    """PSNR in dB for one image pair (data range 1)."""
    test = np.clip(test, 0.0, 1.0)
    mse = float(np.mean((np.asarray(clean, np.float64) - test) ** 2))
    return PSNR_CAP if mse == 0 else min(PSNR_CAP, 10.0 * np.log10(1.0 / mse))


def ssim(clean: np.ndarray, test: np.ndarray) -> float:
    """SSIM for one image pair (data range 1)."""
    return float(structural_similarity(np.asarray(clean, np.float64), np.clip(test, 0.0, 1.0).astype(np.float64),
                                       data_range=1.0))


def pixel_scores(clean: np.ndarray, outputs: np.ndarray) -> dict[str, np.ndarray]:
    """Per-image PSNR and SSIM for stacks [N, H, W]."""
    return {
        "psnr": np.array([psnr(c, o) for c, o in zip(clean, outputs)]),
        "ssim": np.array([ssim(c, o) for c, o in zip(clean, outputs)]),
    }


# ---------------------------------------------------------------------------
# Diagnosis scores
# ---------------------------------------------------------------------------


def diagnosis_scores(p_clean: np.ndarray, p_out: np.ndarray, threshold: float = 0.5) -> dict:
    """Compare classifier probabilities [N, 14] on clean vs processed images.

    Returns per-image arrays (delta_p, agreement) and pooled rates (lost/new findings).
    """
    p_clean = np.asarray(p_clean, np.float64)
    p_out = np.asarray(p_out, np.float64)
    yes_clean = p_clean >= threshold
    yes_out = p_out >= threshold
    n_pos = int(yes_clean.sum())
    n_neg = int((~yes_clean).sum())
    return {
        "delta_p": np.abs(p_out - p_clean).mean(axis=1),
        "agreement": (yes_clean == yes_out).mean(axis=1),
        "lost_findings": float((yes_clean & ~yes_out).sum() / n_pos) if n_pos else float("nan"),
        "new_findings": float((~yes_clean & yes_out).sum() / n_neg) if n_neg else float("nan"),
        "n_clean_findings": n_pos,
    }


def auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC via ranks (Mann-Whitney U); ties count half. NaN if only one class present."""
    y = np.asarray(y_true).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def eligible_diseases(labels: np.ndarray, min_positives: int) -> list[int]:
    """Indices of diseases with enough positive (and negative) cases for a stable AUC."""
    pos = labels.sum(axis=0)
    neg = len(labels) - pos
    return [k for k in range(labels.shape[1]) if pos[k] >= min_positives and neg[k] >= min_positives]


def delta_auc(labels: np.ndarray, p_clean: np.ndarray, p_out: np.ndarray, diseases: list[int],
              n_boot: int = 1000, seed: int = 0) -> dict:
    """dAUC = AUC(clean) - AUC(processed), per disease and averaged (macro), with 95% CIs.

    Paired bootstrap: the same resampled images are used for clean and processed, so the
    CI reflects the uncertainty of the difference, not of each AUC separately.
    Returns {"per_disease": {k: {...}}, "macro": {...}}.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    boot_idx = rng.integers(0, n, size=(n_boot, n))
    diffs = np.full((n_boot, len(diseases)), np.nan)
    per_disease = {}
    for j, k in enumerate(diseases):
        y, pc, po = labels[:, k], p_clean[:, k], p_out[:, k]
        a_clean, a_out = auc(y, pc), auc(y, po)
        diffs[:, j] = [auc(y[i], pc[i]) - auc(y[i], po[i]) for i in boot_idx]
        lo, hi = np.nanpercentile(diffs[:, j], [2.5, 97.5])
        per_disease[k] = {"auc_clean": a_clean, "auc_out": a_out, "delta_auc": a_clean - a_out,
                          "ci_low": float(lo), "ci_high": float(hi), "n_pos": int(y.sum())}
    if diseases:
        macro_boot = np.nanmean(diffs, axis=1)
        lo, hi = np.nanpercentile(macro_boot, [2.5, 97.5])
        macro = {"delta_auc": float(np.mean([v["delta_auc"] for v in per_disease.values()])),
                 "ci_low": float(lo), "ci_high": float(hi), "n_diseases": len(diseases)}
    else:
        macro = {"delta_auc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_diseases": 0}
    return {"per_disease": per_disease, "macro": macro}
