"""Batch 4 tests: classical baselines and the evaluation tables."""

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter

from c21.baselines import CLASSICAL_METHODS, bm3d_gat_denoise, median_denoise, no_denoising
from c21.evaluation import (
    dauc_to_latex,
    load_summaries,
    sanity_checks,
    save_method_results,
    score_method,
    summary_to_latex,
    write_combined_tables,
)
from c21.metrics import psnr
from c21.noise import NoiseLevel, add_noise

MEDIUM = NoiseLevel("medium", 50.0, 0.02)


@pytest.fixture(scope="module")
def smooth_image():
    """A smooth X-ray-like test image (values 0.1 to 0.9)."""
    rng = np.random.default_rng(0)
    img = gaussian_filter(rng.random((96, 96)), 6)
    img = (img - img.min()) / (img.max() - img.min())
    return (0.1 + 0.8 * img).astype(np.float32)


def test_registry_has_the_three_classical_methods():
    assert list(CLASSICAL_METHODS) == ["noisy", "median", "bm3d"]


def test_no_denoising_only_clips():
    y = np.array([[-0.2, 0.5, 1.3]], dtype=np.float32)
    assert np.array_equal(no_denoising(y), [[0.0, 0.5, 1.0]])


def test_median_removes_single_bright_pixel():
    y = np.full((9, 9), 0.3, dtype=np.float32)
    y[4, 4] = 1.0
    out = median_denoise(y)
    assert out.shape == y.shape and out[4, 4] == pytest.approx(0.3)


@pytest.mark.parametrize("fn", [median_denoise, bm3d_gat_denoise])
def test_denoisers_return_valid_images(fn, smooth_image):
    y = add_noise(smooth_image, MEDIUM, seed=1)
    out = fn(y, MEDIUM)
    assert out.shape == y.shape and out.dtype == np.float32
    assert out.min() >= 0 and out.max() <= 1


def test_bm3d_gat_improves_psnr_a_lot_and_keeps_brightness(smooth_image):
    y = add_noise(smooth_image, MEDIUM, seed=2)
    out = bm3d_gat_denoise(y, MEDIUM)
    gain = psnr(smooth_image, out) - psnr(smooth_image, y)
    assert gain > 5  # if the Anscombe step were wrong, the gain would be much smaller
    assert abs(out.mean() - smooth_image.mean()) < 0.01  # unbiased inverse: no darkening


def test_bm3d_beats_median_on_this_noise(smooth_image):
    y = add_noise(smooth_image, MEDIUM, seed=3)
    assert psnr(smooth_image, bm3d_gat_denoise(y, MEDIUM)) > psnr(smooth_image, median_denoise(y))


# --- evaluation tables ------------------------------------------------------------


def _fake_method_results(method, quality, cfg_small):
    """Fake outputs/predictions whose quality decreases as `quality` drops."""
    rng = np.random.default_rng(len(method))
    clean = rng.random((40, 32, 32)).astype(np.float32)
    outputs = np.clip(clean + rng.normal(0, 0.2 * (1 - quality) + 0.01, clean.shape), 0, 1)
    labels = (rng.random((40, 14)) < 0.5).astype(int)
    p_clean = np.clip(labels * 0.6 + rng.random((40, 14)) * 0.4, 0, 1)
    p_out = np.clip(p_clean + rng.normal(0, 0.3 * (1 - quality), p_clean.shape), 0, 1)
    names = [f"img{i}.png" for i in range(40)]
    return score_method(method, clean, outputs, p_clean, p_out, labels, names, cfg_small, 0.01)


@pytest.fixture
def cfg_small():
    return {"seed": 0, "classifier": {"threshold": 0.5, "min_positives_for_auc": 10, "bootstrap_samples": 100}}


def test_score_method_outputs(cfg_small):
    per_image, auc_df, summary = _fake_method_results("median", 0.5, cfg_small)
    assert len(per_image) == 40 and set(per_image.columns) >= {"psnr", "ssim", "delta_p", "agreement"}
    assert "macro" in set(auc_df["disease"])
    assert summary["n_images"] == 40 and summary["dauc_diseases"] > 0


def test_tables_combine_methods_in_standard_order(tmp_path, cfg_small):
    for method, quality in [("bm3d", 0.9), ("noisy", 0.0), ("median", 0.5)]:  # saved out of order
        save_method_results(tmp_path, method, *_fake_method_results(method, quality, cfg_small))
    summary = write_combined_tables(tmp_path)
    assert list(summary["method"]) == ["noisy", "median", "bm3d"]
    assert (tmp_path / "summary.tex").exists() and (tmp_path / "dauc.tex").exists()
    assert list(load_summaries(tmp_path)["method"]) == ["noisy", "median", "bm3d"]
    tex = (tmp_path / "summary.tex").read_text(encoding="utf-8")
    assert tex.count(r"\\") == 4 and "Anscombe-BM3D" in tex  # header + 3 method rows
    assert all(sanity_checks(summary).values())


def test_latex_escapes_disease_names():
    auc_df = pd.DataFrame([
        {"method": "bm3d", "disease": "Pleural_Thickening", "n_pos": 20, "delta_auc": 0.01, "ci_low": -0.02, "ci_high": 0.04},
        {"method": "bm3d", "disease": "macro", "n_pos": np.nan, "delta_auc": 0.01, "ci_low": -0.02, "ci_high": 0.04},
    ])
    tex = dauc_to_latex(auc_df)
    assert "Pleural Thickening (20)" in tex and "Pleural_Thickening" not in tex


def test_summary_latex_has_one_row_per_method(cfg_small):
    rows = [_fake_method_results(m, q, cfg_small)[2] for m, q in [("noisy", 0.0), ("bm3d", 0.9)]]
    tex = summary_to_latex(pd.DataFrame(rows))
    assert tex.startswith(r"\begin{tabular}") and tex.count(r"\pm") == 4
