"""Batch 1 tests: noise model, seeds, and the Anscombe transform.

The statistical tests use large flat images and fixed seeds, so they are exact
repeats every run and finish in a few seconds.
"""

import numpy as np
import pytest

from c21.config import load_config
from c21.noise import (
    NoiseLevel,
    add_noise,
    add_noise_torch,
    add_poisson_gaussian_noise,
    all_levels,
    gat_forward,
    gat_inverse_algebraic,
    gat_inverse_unbiased,
    get_level,
    image_seed,
)

LEVELS = all_levels(load_config())
BRIGHTNESS = [0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
N = 1_000_000  # pixels per flat test image


def flat(x):
    return np.full(N, x, dtype=np.float32)


# --- config and seeds -------------------------------------------------------


def test_three_levels_from_config():
    assert [lv.name for lv in LEVELS] == ["low", "medium", "high"]
    assert get_level(load_config()).name == "medium"  # primary level


def test_same_seed_gives_identical_noise():
    img = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    a = add_poisson_gaussian_noise(img, 50, 0.02, seed=123)
    b = add_poisson_gaussian_noise(img, 50, 0.02, seed=123)
    assert np.array_equal(a, b)


def test_different_images_get_different_noise():
    """The M1 bug: one fixed seed gave every image the same noise pattern."""
    img = np.full((64, 64), 0.5, dtype=np.float32)
    a = add_poisson_gaussian_noise(img, 50, 0.02, seed=image_seed(100000, 0)) - img
    b = add_poisson_gaussian_noise(img, 50, 0.02, seed=image_seed(100000, 1)) - img
    assert not np.allclose(a, b)
    assert image_seed(100000, 7) == 100007


# --- noise statistics ---------------------------------------------------------


def test_output_is_float32_and_same_shape():
    img = np.zeros((10, 20), dtype=np.float32)
    out = add_poisson_gaussian_noise(img, 50, 0.02, seed=0)
    assert out.shape == img.shape and out.dtype == np.float32


def test_noise_is_not_clipped_by_default():
    y = add_poisson_gaussian_noise(flat(1.0), 50, 0.02, seed=0)
    assert y.max() > 1.0 and add_poisson_gaussian_noise(flat(0.0), 50, 0.02, seed=0).min() < 0.0
    y_clip = add_poisson_gaussian_noise(flat(1.0), 50, 0.02, seed=0, clip=True)
    assert y_clip.max() <= 1.0


@pytest.mark.parametrize("level", LEVELS, ids=lambda lv: lv.name)
@pytest.mark.parametrize("x", [0.1, 0.5, 0.9])
def test_noise_mean_and_variance_match_the_model(level, x):
    y = add_noise(flat(x), level, seed=1).astype(np.float64)
    expected_var = x / level.lambda_photons + level.sigma_electronic**2
    assert abs(y.mean() - x) < 0.002  # noise does not shift brightness
    assert abs(y.var() / expected_var - 1) < 0.02  # variance = x/lam + sigma^2


def test_brighter_pixels_are_noisier():
    lv = LEVELS[1]
    stds = [add_noise(flat(x), lv, seed=2).std() for x in (0.1, 0.5, 0.9)]
    assert stds[0] < stds[1] < stds[2]


def test_torch_noise_matches_numpy_statistics():
    torch = pytest.importorskip("torch")
    lv = LEVELS[1]
    x = torch.full((N,), 0.4)
    y = add_noise_torch(x, lv.lambda_photons, lv.sigma_electronic,
                        generator=torch.Generator().manual_seed(0)).double()
    y_np = add_noise(np.full(N, 0.4, np.float32), lv, seed=0).astype(np.float64)
    assert abs(y.mean().item() - y_np.mean()) < 0.002
    assert abs(y.std().item() / y_np.std() - 1) < 0.02


def test_torch_noise_accepts_one_level_per_image():
    torch = pytest.importorskip("torch")
    x = torch.full((2, 1, 32, 32), 0.5)
    lam = torch.tensor([100.0, 25.0]).view(2, 1, 1, 1)
    sig = torch.tensor([0.01, 0.04]).view(2, 1, 1, 1)
    y = add_noise_torch(x, lam, sig, generator=torch.Generator().manual_seed(0))
    assert y[0].std() < y[1].std()  # low-noise image is cleaner than high-noise image


# --- Anscombe transform -------------------------------------------------------


@pytest.mark.parametrize("level", LEVELS, ids=lambda lv: lv.name)
@pytest.mark.parametrize("x", BRIGHTNESS)
def test_gat_makes_noise_std_close_to_one(level, x):
    """The M1 bug: std came out 0.07 to 0.11 and changed with brightness."""
    y = add_noise(flat(x), level, seed=3)
    d = gat_forward(y, level.lambda_photons, level.sigma_electronic)
    assert 0.95 < d.std() < 1.05


@pytest.mark.parametrize("level", LEVELS, ids=lambda lv: lv.name)
@pytest.mark.parametrize("x", BRIGHTNESS)
def test_unbiased_inverse_has_almost_no_bias(level, x):
    """An ideal denoiser outputs the mean of the stabilized values; inverting it must give back x."""
    y = add_noise(flat(x), level, seed=4)
    mean_d = gat_forward(y, level.lambda_photons, level.sigma_electronic).mean()
    x_hat = gat_inverse_unbiased(mean_d, level.lambda_photons, level.sigma_electronic)
    assert abs(x_hat - x) < 0.0015


@pytest.mark.parametrize("level", LEVELS, ids=lambda lv: lv.name)
def test_unbiased_inverse_beats_algebraic_inverse(level):
    y = add_noise(flat(0.5), level, seed=5)
    mean_d = gat_forward(y, level.lambda_photons, level.sigma_electronic).mean()
    err_unbiased = abs(gat_inverse_unbiased(mean_d, level.lambda_photons, level.sigma_electronic) - 0.5)
    err_algebraic = abs(gat_inverse_algebraic(mean_d, level.lambda_photons, level.sigma_electronic) - 0.5)
    assert err_unbiased * 3 < err_algebraic


def test_algebraic_inverse_undoes_forward_exactly():
    y = np.linspace(0.0, 1.2, 1000)
    lv = NoiseLevel("t", 50.0, 0.02)
    back = gat_inverse_algebraic(gat_forward(y, lv.lambda_photons, lv.sigma_electronic),
                                 lv.lambda_photons, lv.sigma_electronic)
    assert np.allclose(back, y, atol=1e-9)


def test_unbiased_inverse_is_monotonic_and_never_negative():
    d = np.linspace(0.0, 25.0, 25001)
    v = gat_inverse_unbiased(d, 50.0, 0.02)
    assert np.all(v >= 0)
    assert np.all(np.diff(v) >= -1e-12)
