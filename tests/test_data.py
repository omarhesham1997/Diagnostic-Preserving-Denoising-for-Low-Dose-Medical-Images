"""Batch 2 tests: label parsing, patient-level split, preprocessing, fixed noisy sets.

Most tests build a small fake dataset in a temporary folder with the same layout as
Kaggle's NIH sample. The last test checks the real committed split file, if present.
"""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from c21.config import load_config
from c21.constants import NIH_LABELS
from c21.data import (
    check_split,
    find_label_csv,
    index_images,
    load_and_resize,
    load_labels,
    make_patient_split,
    split_csv_path,
)
from c21.noise import get_level, make_noisy_set

SIZES = {"train": 60, "val": 15, "test": 25}
FINDINGS = ["No Finding", "Effusion", "Cardiomegaly|Effusion", "Hernia", "Pleural_Thickening|Mass"]


@pytest.fixture(scope="module")
def fake_raw(tmp_path_factory):
    """150 patients with 1 to 4 images each, plus 10 patients listed in the CSV but not on disk."""
    root = tmp_path_factory.mktemp("raw")
    img_dir = root / "nih-sample" / "sample" / "images"
    img_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for pid in range(1, 161):
        for k in range(int(rng.integers(1, 5))):
            name = f"{pid:08d}_{k:03d}.png"
            rows.append({"Image Index": name, "Finding Labels": FINDINGS[(pid + k) % len(FINDINGS)],
                         "Follow-up #": k, "Patient ID": pid, "Patient Age": "045Y",
                         "Patient Gender": "M" if pid % 2 else "F", "View Position": "PA"})
            if pid <= 150:  # patients 151-160 have no files on disk
                pixels = rng.integers(0, 256, (96, 96), dtype=np.uint8)
                mode = "RGBA" if k == 3 else "L"  # a few NIH images are RGBA
                Image.fromarray(pixels).convert(mode).save(img_dir / name)
    pd.DataFrame(rows).to_csv(root / "nih-sample" / "sample_labels.csv", index=False)
    return root


@pytest.fixture(scope="module")
def labels(fake_raw):
    return load_labels(find_label_csv(fake_raw))


@pytest.fixture(scope="module")
def available(fake_raw):
    return set(index_images(fake_raw))


# --- raw files and labels -------------------------------------------------------


def test_finds_label_file_and_images(fake_raw, labels, available):
    assert find_label_csv(fake_raw).name == "sample_labels.csv"
    assert len(available) == len(labels[labels["patient_id"] <= 150])


def test_labels_are_parsed_into_14_columns(labels):
    assert all(label in labels.columns for label in NIH_LABELS)
    row = labels[labels["finding_labels"] == "Cardiomegaly|Effusion"].iloc[0]
    assert row["Cardiomegaly"] == 1 and row["Effusion"] == 1 and row["Mass"] == 0
    healthy = labels[labels["finding_labels"] == "No Finding"].iloc[0]
    assert healthy[NIH_LABELS].sum() == 0 and healthy["no_finding"] == 1
    assert labels.loc[labels["finding_labels"] == "Pleural_Thickening|Mass", "Pleural_Thickening"].all()


# --- patient-level split ----------------------------------------------------------


def test_split_has_exact_sizes_and_no_patient_overlap(labels, available):
    split = make_patient_split(labels, available, SIZES, seed=42, max_per_patient=2)
    assert check_split(split, SIZES) == []


def test_split_is_reproducible_and_depends_on_seed(labels, available):
    a = make_patient_split(labels, available, SIZES, seed=42)
    b = make_patient_split(labels, available, SIZES, seed=42)
    c = make_patient_split(labels, available, SIZES, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["image"]) != set(c["image"])


def test_split_uses_only_images_on_disk(labels, available):
    split = make_patient_split(labels, available, SIZES, seed=42)
    assert set(split["image"]) <= available


def test_split_respects_max_images_per_patient(labels, available):
    split = make_patient_split(labels, available, SIZES, seed=42, max_per_patient=1)
    assert split["patient_id"].value_counts().max() == 1


def test_split_fails_clearly_when_too_few_images(labels, available):
    with pytest.raises(ValueError, match="Download more images"):
        make_patient_split(labels, available, {"train": 1000, "val": 10, "test": 10}, seed=42)


def test_check_split_detects_patient_leak():
    bad = pd.DataFrame({"image": ["a.png", "b.png"], "split": ["train", "test"], "patient_id": [5, 5]})
    assert any("both" in p for p in check_split(bad))


# --- preprocessing ---------------------------------------------------------------------


def test_load_and_resize_gives_square_uint8_even_for_rgba(fake_raw):
    rgba = sorted((fake_raw / "nih-sample" / "sample" / "images").glob("*_003.png"))[0]
    arr = load_and_resize(rgba, 64)
    assert arr.shape == (64, 64) and arr.dtype == np.uint8


# --- fixed noisy sets ----------------------------------------------------------------


def test_noisy_set_is_identical_on_rerun_and_differs_between_images():
    clean = np.full((3, 32, 32), 0.5, dtype=np.float32)
    level = get_level(load_config(), "medium")
    a = make_noisy_set(clean, level, seed_base=100000)
    b = make_noisy_set(clean, level, seed_base=100000)
    assert a.dtype == np.float32 and np.array_equal(a, b)
    assert not np.allclose(a[0], a[1])  # each image has its own noise


# --- the real split file (only after scripts/make_split.py has been run) --------------


def test_committed_split_file_is_valid():
    cfg = load_config()
    path = split_csv_path(cfg)
    if not path.exists():
        pytest.skip(f"{path.name} not created yet (run scripts/make_split.py)")
    split = pd.read_csv(path)
    assert check_split(split, cfg["data"]["split_sizes"]) == []
    assert all(label in split.columns for label in NIH_LABELS)
