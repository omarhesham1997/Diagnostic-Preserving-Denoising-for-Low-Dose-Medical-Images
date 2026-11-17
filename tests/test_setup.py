"""Batch 0 tests: project layout, config and seeds. No network or data needed."""

import numpy as np
import pytest

from c21.config import load_config, repo_root
from c21.constants import NIH_LABELS
from c21.utils import set_seed


def test_repo_root_has_pyproject():
    assert (repo_root() / "pyproject.toml").exists()


def test_config_loads_and_split_sizes_add_up():
    cfg = load_config()
    sizes = cfg["data"]["split_sizes"]
    assert sum(sizes.values()) == cfg["data"]["n_images"]
    assert set(sizes) == {"train", "val", "test"}


def test_config_paths_are_absolute():
    cfg = load_config()
    for key, path in cfg["paths"].items():
        assert path.is_absolute(), key


def test_noise_levels_get_stronger_from_low_to_high():
    levels = load_config()["noise"]["levels"]
    lam = [levels[k]["lambda_photons"] for k in ("low", "medium", "high")]
    sig = [levels[k]["sigma_electronic"] for k in ("low", "medium", "high")]
    assert lam[0] > lam[1] > lam[2]  # fewer photons = lower dose = more noise
    assert sig[0] < sig[1] < sig[2]


def test_fourteen_unique_nih_labels():
    assert len(NIH_LABELS) == 14
    assert len(set(NIH_LABELS)) == 14


def test_set_seed_makes_numpy_repeatable():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_set_seed_makes_torch_repeatable():
    torch = pytest.importorskip("torch")
    set_seed(7)
    a = torch.rand(5)
    set_seed(7)
    b = torch.rand(5)
    assert torch.equal(a, b)
