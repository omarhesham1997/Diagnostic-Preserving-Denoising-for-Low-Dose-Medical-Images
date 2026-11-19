"""Pick the image subset and split it by patient into train / val / test.

Run:
    uv run python scripts/make_split.py

Writes splits/split_seed<seed>.csv (image IDs, patient IDs, labels; no images).
Commit that file: it defines the experiment. Re-running with the same seed and the
same downloaded images gives a byte-identical file (compare the printed checksum).
"""

from __future__ import annotations

import argparse
import hashlib
import sys

from c21.config import load_config
from c21.constants import NIH_LABELS
from c21.data import (
    SPLITS,
    check_split,
    find_label_csv,
    index_images,
    load_labels,
    make_patient_split,
    split_csv_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="config file (default configs/default.yaml)")
    parser.add_argument("--seed", type=int, help="override the config seed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    raw = cfg["paths"]["raw_data"]
    sizes = cfg["data"]["split_sizes"]
    max_pp = cfg["data"]["max_images_per_patient"]

    csv_path = find_label_csv(raw)
    labels = load_labels(csv_path)
    images = index_images(raw)
    available = set(images) & set(labels["image"])
    print(f"Label file: {csv_path.name} ({len(labels)} rows, {labels['patient_id'].nunique()} patients)")
    print(f"Images on disk with labels: {len(available)}")

    split = make_patient_split(labels, available, sizes, seed=cfg["seed"], max_per_patient=max_pp)
    problems = check_split(split, sizes)
    if problems:
        print("SPLIT INVALID: " + "; ".join(problems))
        return 1

    out = split_csv_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(out, index=False, lineterminator="\n")
    checksum = hashlib.sha256(out.read_bytes()).hexdigest()[:12]

    print(f"\nSaved {out} (seed {cfg['seed']}, max {max_pp} images per patient, checksum {checksum})")
    print("No patient appears in two splits: OK\n")
    print(f"{'split':6s}{'images':>8s}{'patients':>10s}{'No Finding':>12s}")
    for s in SPLITS:
        part = split[split["split"] == s]
        print(f"{s:6s}{len(part):8d}{part['patient_id'].nunique():10d}{100 * part['no_finding'].mean():11.1f}%")

    print("\nImages per disease (a disease needs about 25+ test cases for a stable AUC):")
    print(f"{'disease':20s}" + "".join(f"{s:>7s}" for s in SPLITS))
    for label in NIH_LABELS:
        counts = [int(split.loc[split["split"] == s, label].sum()) for s in SPLITS]
        print(f"{label:20s}" + "".join(f"{c:7d}" for c in counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
