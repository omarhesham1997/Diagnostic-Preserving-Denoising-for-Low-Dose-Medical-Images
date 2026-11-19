"""Batch 3 tests: pixel scores, diagnosis scores, AUC, and the frozen classifier."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from c21.classifier import to_xrv_input
from c21.metrics import (
    PSNR_CAP,
    auc,
    delta_auc,
    diagnosis_scores,
    eligible_diseases,
    pixel_scores,
    psnr,
    ssim,
)

rng = np.random.default_rng(0)


# --- pixel scores -------------------------------------------------------------


def test_psnr_of_identical_images_is_capped():
    img = rng.random((32, 32))
    assert psnr(img, img) == PSNR_CAP


def test_psnr_known_value():
    clean = np.full((16, 16), 0.5)
    assert psnr(clean, clean + 0.1) == pytest.approx(20.0)  # mse 0.01 -> 20 dB


def test_psnr_clips_output_to_valid_range():
    clean = np.ones((8, 8))
    assert psnr(clean, clean + 5.0) == PSNR_CAP  # 6.0 is clipped to 1.0


def test_ssim_of_identical_images_is_one():
    img = rng.random((64, 64))
    assert ssim(img, img) == pytest.approx(1.0)


def test_pixel_scores_shapes_and_order():
    clean = rng.random((3, 32, 32))
    noisy = clean + rng.normal(0, 0.1, clean.shape)
    s = pixel_scores(clean, noisy)
    assert s["psnr"].shape == (3,) and s["ssim"].shape == (3,)
    assert np.all(s["psnr"] < PSNR_CAP)


# --- diagnosis scores -----------------------------------------------------------


def test_same_predictions_mean_nothing_changed():
    p = rng.random((50, 14))
    d = diagnosis_scores(p, p)
    assert d["delta_p"].mean() == 0 and d["agreement"].mean() == 1
    assert d["lost_findings"] == 0 and d["new_findings"] == 0


def test_flip_counting_on_small_example():
    p_clean = np.array([[0.9, 0.8, 0.1, 0.2]])
    p_out = np.array([[0.4, 0.8, 0.6, 0.2]])  # disease 0 lost, disease 2 new
    d = diagnosis_scores(p_clean, p_out, threshold=0.5)
    assert d["agreement"][0] == pytest.approx(0.5)
    assert d["lost_findings"] == pytest.approx(0.5)  # 1 of 2 clean findings lost
    assert d["new_findings"] == pytest.approx(0.5)  # 1 of 2 clean non-findings became findings
    assert d["delta_p"][0] == pytest.approx((0.5 + 0 + 0.5 + 0) / 4)


def test_auc_matches_sklearn_including_ties():
    y = rng.integers(0, 2, 300)
    s = np.round(rng.random(300), 1)  # many ties
    assert auc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_auc_is_nan_with_one_class():
    assert np.isnan(auc(np.zeros(10), rng.random(10)))


def test_eligible_diseases_need_enough_cases():
    labels = np.zeros((100, 3), dtype=int)
    labels[:30, 0] = 1  # 30 cases
    labels[:5, 1] = 1  # 5 cases
    assert eligible_diseases(labels, 20) == [0]


def test_delta_auc_is_zero_for_identical_predictions():
    labels = (rng.random((200, 2)) < 0.3).astype(int)
    p = rng.random((200, 2))
    res = delta_auc(labels, p, p, diseases=[0, 1], n_boot=200)
    assert res["macro"]["delta_auc"] == 0
    assert res["per_disease"][0]["ci_low"] == 0 and res["per_disease"][0]["ci_high"] == 0


def test_delta_auc_detects_destroyed_predictions():
    labels = (rng.random((400, 1)) < 0.3).astype(int)
    p_good = np.clip(labels + rng.normal(0, 0.3, labels.shape), 0, 1)  # informative
    p_bad = rng.random(labels.shape)  # random guessing
    res = delta_auc(labels, p_good, p_bad, diseases=[0], n_boot=300)
    assert res["macro"]["delta_auc"] > 0.2
    assert res["macro"]["ci_low"] > 0  # the drop is clearly real


# --- classifier input and the real model ---------------------------------------------


def test_xrv_input_scaling_and_size():
    images = np.stack([np.zeros((256, 256)), np.ones((256, 256)), np.full((256, 256), 3.0)])
    x = to_xrv_input(images, 224)
    assert tuple(x.shape) == (3, 1, 224, 224)
    assert x[0].min().item() == pytest.approx(-1024)  # black -> -1024
    assert x[1].max().item() == pytest.approx(1024)  # white -> +1024
    assert x[2].max().item() == pytest.approx(1024)  # values above 1 are clipped


@pytest.fixture(scope="module")
def judge():
    try:
        from c21.classifier import FrozenClassifier

        return FrozenClassifier(device="cpu")
    except Exception as exc:  # weights not downloadable here
        pytest.skip(f"classifier not available: {exc}")


def test_judge_outputs_14_probabilities(judge):
    p = judge.predict(rng.random((3, 256, 256)).astype(np.float32))
    assert p.shape == (3, 14)
    assert np.all((p >= 0) & (p <= 1))


def test_judge_is_deterministic_and_frozen(judge):
    imgs = rng.random((2, 256, 256)).astype(np.float32)
    assert np.allclose(judge.predict(imgs), judge.predict(imgs), atol=1e-6)
    assert not any(p.requires_grad for p in judge.model.parameters())
    assert not judge.model.training
