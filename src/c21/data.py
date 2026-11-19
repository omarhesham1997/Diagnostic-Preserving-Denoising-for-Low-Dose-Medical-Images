"""NIH ChestX-ray14: labels, patient-level split, preprocessing and loading.

Works with any of these layouts under the raw-data folder (searched recursively):
  - Kaggle "nih-chest-xrays/sample": sample_labels.csv + sample/images/*.png
  - Kaggle "nih-chest-xrays/data":   Data_Entry_2017.csv + images_0XX/images/*.png
  - NIH Box download:                Data_Entry_2017_v2020.csv + extracted images/*.png
Only images that are actually on disk are used.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from c21.constants import NIH_LABELS

LABEL_CSV_NAMES = ["Data_Entry_2017_v2020.csv", "Data_Entry_2017.csv", "sample_labels.csv"]
SPLITS = ("train", "val", "test")

# ---------------------------------------------------------------------------
# Raw files
# ---------------------------------------------------------------------------


def find_label_csv(raw_dir: Path) -> Path:
    """Find the NIH label file anywhere under raw_dir (full-dataset files preferred)."""
    for name in LABEL_CSV_NAMES:
        hits = sorted(Path(raw_dir).rglob(name))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No label file ({', '.join(LABEL_CSV_NAMES)}) found under {raw_dir}. "
        "Run scripts/download_data.py first."
    )


def index_images(raw_dir: Path) -> dict[str, Path]:
    """Map image file name (e.g. '00000013_005.png') to its path, for every PNG under raw_dir."""
    return {p.name: p for p in sorted(Path(raw_dir).rglob("*.png"))}


def load_labels(csv_path: Path) -> pd.DataFrame:
    """Read the NIH CSV into a tidy table: one row per image, one 0/1 column per disease."""
    raw = pd.read_csv(csv_path)
    raw.columns = [c.strip() for c in raw.columns]

    def col(*names):
        for n in names:
            if n in raw.columns:
                return raw[n]
        return pd.Series([None] * len(raw))

    df = pd.DataFrame({
        "image": raw["Image Index"].astype(str),
        "patient_id": raw["Patient ID"].astype(int),
        "follow_up": col("Follow-up #"),
        "age": col("Patient Age"),
        "sex": col("Patient Gender", "Patient Sex"),
        "view": col("View Position"),
        "finding_labels": raw["Finding Labels"].astype(str),
    })
    findings = df["finding_labels"].str.split("|")
    for label in NIH_LABELS:
        df[label] = findings.apply(lambda f, lab=label: int(lab in f))
    df["no_finding"] = (df[NIH_LABELS].sum(axis=1) == 0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Patient-level split
# ---------------------------------------------------------------------------


def make_patient_split(
    labels: pd.DataFrame,
    available: set[str] | None,
    sizes: dict[str, int],
    seed: int,
    max_per_patient: int = 2,
) -> pd.DataFrame:
    """Pick sum(sizes) images and assign them to splits so that no patient is in two splits.

    Patients are visited in a random (seeded) order; each patient contributes at most
    `max_per_patient` randomly chosen images. Splits are filled in the order
    test, val, train, so the test set is never short. Sizes are exact.
    """
    df = labels if available is None else labels[labels["image"].isin(available)]
    df = df.sort_values("image").reset_index(drop=True)
    need = sum(sizes.values())

    rng = np.random.default_rng(seed)
    patients = rng.permutation(np.sort(df["patient_id"].unique()))
    by_patient = {pid: idx.to_numpy() for pid, idx in df.groupby("patient_id").groups.items()}

    remaining = {s: sizes[s] for s in ("test", "val", "train")}
    chosen_rows, chosen_split = [], []
    for pid in patients:
        open_splits = [s for s, n in remaining.items() if n > 0]
        if not open_splits:
            break
        split = open_splits[0]
        rows = rng.permutation(by_patient[pid])[:max_per_patient]
        rows = rows[: remaining[split]]
        chosen_rows.extend(rows.tolist())
        chosen_split.extend([split] * len(rows))
        remaining[split] -= len(rows)

    if len(chosen_rows) < need:
        raise ValueError(
            f"Only {len(chosen_rows)} images could be selected but {need} are needed "
            f"({len(df)} images from {df['patient_id'].nunique()} patients available, "
            f"max {max_per_patient} per patient). Download more images or lower n_images."
        )

    out = df.loc[chosen_rows].copy()
    out.insert(1, "split", chosen_split)
    order = {s: i for i, s in enumerate(SPLITS)}
    out = out.sort_values(["split", "image"], key=lambda c: c.map(order) if c.name == "split" else c)
    return out.reset_index(drop=True)


def check_split(split_df: pd.DataFrame, sizes: dict[str, int] | None = None) -> list[str]:
    """Return a list of problems (empty list = split is valid)."""
    problems = []
    groups = split_df.groupby("split")["patient_id"].apply(set).to_dict()
    for a in SPLITS:
        for b in SPLITS:
            if a < b and a in groups and b in groups and groups[a] & groups[b]:
                problems.append(f"{len(groups[a] & groups[b])} patient(s) in both {a} and {b}")
    if split_df["image"].duplicated().any():
        problems.append("duplicate images")
    if sizes:
        counts = split_df["split"].value_counts().to_dict()
        for s, n in sizes.items():
            if counts.get(s, 0) != n:
                problems.append(f"{s} has {counts.get(s, 0)} images, expected {n}")
    return problems


def split_csv_path(cfg: dict) -> Path:
    return cfg["paths"]["splits"] / f"split_seed{cfg['seed']}.csv"


def read_split(cfg: dict) -> pd.DataFrame:
    path = split_csv_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/make_split.py first.")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Preprocessing and loading
# ---------------------------------------------------------------------------


def load_and_resize(path: Path, size: int) -> np.ndarray:
    """Read a PNG as grayscale and resize to size x size. Returns uint8."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("L")  # a few NIH images are RGBA
        if im.size != (size, size):
            im = im.resize((size, size), Image.Resampling.LANCZOS)
        return np.asarray(im, dtype=np.uint8)


def clean_array_path(cfg: dict, split: str) -> Path:
    return cfg["paths"]["processed"] / f"{split}_{cfg['data']['image_size']}.npy"


def noisy_array_path(cfg: dict, split: str, level: str) -> Path:
    return cfg["paths"]["processed"] / "noisy" / f"{split}_{level}.npy"


def split_rows(cfg: dict, split: str) -> pd.DataFrame:
    """Rows of one split, in the same order as the preprocessed arrays."""
    df = read_split(cfg)
    return df[df["split"] == split].reset_index(drop=True)


def load_label_matrix(cfg: dict, split: str) -> np.ndarray:
    """Doctor labels of one split as a 0/1 matrix [N, 14] (NIH_LABELS order)."""
    return split_rows(cfg, split)[NIH_LABELS].to_numpy(dtype=np.int64)


def load_clean(cfg: dict, split: str) -> np.ndarray:
    """Clean images of one split as float32 in [0, 1], shape [N, H, W], in split-CSV order."""
    path = clean_array_path(cfg, split)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/preprocess.py first.")
    return np.load(path).astype(np.float32) / 255.0


def load_noisy(cfg: dict, split: str, level: str) -> np.ndarray:
    """Fixed noisy images (float32, NOT clipped) for val/test, same order as load_clean."""
    path = noisy_array_path(cfg, split, level)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/make_noisy_sets.py first.")
    return np.load(path)


def array_checksum(arr: np.ndarray) -> str:
    """Short SHA-256 of an array's bytes: identical arrays give identical checksums."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]
