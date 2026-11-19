"""Batch 6 tests: paired comparison, example selection, and figure helpers."""

import numpy as np
import pandas as pd
import pytest

from c21.evaluation import comparison_to_latex, paired_comparison, pick_examples
from c21.viz import detail_center, save_qualitative, save_tradeoff_chart

rng = np.random.default_rng(0)
N = 120


def per_image(method, psnr, delta_p):
    return pd.DataFrame({"image": [f"img{i}.png" for i in range(N)], "method": method, "psnr": psnr,
                         "ssim": np.clip(psnr / 40, 0, 1), "delta_p": delta_p, "agreement": 1 - delta_p})


@pytest.fixture(scope="module")
def judge_data():
    labels = (rng.random((N, 3)) < 0.4).astype(int)
    p_clean = np.clip(labels * 0.5 + rng.random((N, 3)) * 0.5, 0, 1)
    return labels, p_clean


# --- paired comparison ------------------------------------------------------------


def test_clearly_better_method_wins_on_every_metric(judge_data):
    labels, p_clean = judge_data
    base = rng.normal(30, 1, N)
    a = per_image("resunet", base + 1.0, np.full(N, 0.05) + rng.normal(0, 0.005, N))
    b = per_image("bm3d", base, np.full(N, 0.15) + rng.normal(0, 0.005, N))
    p_a = p_clean.copy()  # A keeps the judge's opinion exactly
    p_b = np.clip(p_clean + rng.normal(0, 0.4, p_clean.shape), 0, 1)  # B scrambles it
    df = paired_comparison(a, b, labels, p_clean, p_a, p_b, diseases=[0, 1, 2], n_boot=300)
    res = df.set_index("metric")
    assert res.loc["PSNR (dB)", "diff"] == pytest.approx(1.0)
    assert res.loc["PSNR (dB)", "better"] == "ResU-Net (ours)"
    assert res.loc["delta_p", "better"] == "ResU-Net (ours)"  # lower delta_p is better
    assert res.loc["Lost findings", "a"] == 0 and res.loc["Lost findings", "better"] == "ResU-Net (ours)"
    assert res.loc["Macro AUC", "diff"] > 0


def test_identical_methods_show_no_clear_difference(judge_data):
    labels, p_clean = judge_data
    a = per_image("resunet", np.full(N, 30.0), np.full(N, 0.1))
    b = per_image("bm3d", np.full(N, 30.0), np.full(N, 0.1))
    df = paired_comparison(a, b, labels, p_clean, p_clean, p_clean, diseases=[0], n_boot=100)
    assert set(df["better"]) == {"no clear difference"}
    assert (df["diff"] == 0).all()


def test_comparison_requires_same_image_order(judge_data):
    labels, p_clean = judge_data
    a = per_image("resunet", np.full(N, 30.0), np.full(N, 0.1))
    b = a.iloc[::-1].assign(method="bm3d")
    with pytest.raises(ValueError):
        paired_comparison(a, b, labels, p_clean, p_clean, p_clean, diseases=[])


def test_comparison_latex_has_one_row_per_metric(judge_data):
    labels, p_clean = judge_data
    a = per_image("resunet", rng.normal(31, 1, N), rng.random(N) * 0.1)
    b = per_image("bm3d", rng.normal(30, 1, N), rng.random(N) * 0.1)
    df = paired_comparison(a, b, labels, p_clean, p_clean, p_clean, diseases=[0], n_boot=50)
    tex = comparison_to_latex(df)
    assert tex.count(r"\\") == len(df) + 1 and r"$\Delta p$" in tex


# --- example selection and figures -------------------------------------------------


def test_examples_follow_the_fixed_rules():
    per = pd.DataFrame({"psnr": np.arange(10, dtype=float), "delta_p": np.linspace(0, 1, 10)})
    rows = pd.DataFrame({"Nodule": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0], "Mass": [0] * 10})
    chosen = {tag: i for tag, i, _ in pick_examples(per, rows)}
    assert chosen["1_typical"] == 5  # median PSNR
    assert chosen["2_small_finding"] in (2, 6)  # a Nodule image
    assert chosen["3_worst_case"] == 9  # largest delta_p
    assert len(set(chosen.values())) == 3


def test_detail_center_finds_the_textured_area():
    img = np.full((256, 256), 0.5)
    img[150:220, 40:110] += rng.normal(0, 0.2, (70, 70))  # texture only here
    r, c = detail_center(img)
    assert 150 <= r <= 220 and 40 <= c <= 110


def test_figure_helpers_write_files(tmp_path):
    clean = rng.random((128, 128))
    panels = [("clean", clean), ("noisy", clean + rng.normal(0, 0.1, clean.shape)), ("ours", clean)]
    assert save_qualitative(clean, panels, "test", tmp_path / "q.png").stat().st_size > 0
    summary = pd.DataFrame({"method": ["noisy", "bm3d", "resunet"], "psnr_mean": [20.0, 33.5, 34.5],
                            "delta_p_mean": [0.3, 0.1, 0.08]})
    ci = {"noisy": (0.1, 0.01), "bm3d": (0.1, 0.01), "resunet": (0.1, 0.01)}
    assert save_tradeoff_chart(summary, ci, tmp_path / "c.png", "test").stat().st_size > 0
