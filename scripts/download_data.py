"""Download NIH ChestX-ray14 from Kaggle into data/raw (replaces data/dataset_download.py).

Recommended: Kaggle's official random 5% sample (5,606 images, one download):
    uv run python scripts/download_data.py

Alternative: the label file plus chosen image folders of the full dataset:
    uv run python scripts/download_data.py --source full --folders images_001 images_002

The later scripts search data/raw recursively, so the exact folder layout does not matter.
If Kaggle asks for credentials, create an API token on kaggle.com (Settings > API)
and save it as %USERPROFILE%/.kaggle/kaggle.json (never commit it).
"""

from __future__ import annotations

import argparse
import sys

from c21.config import load_config
from c21.data import find_label_csv, index_images

HANDLES = {"sample": "nih-chest-xrays/sample", "full": "nih-chest-xrays/data"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=HANDLES, default="sample")
    parser.add_argument("--folders", nargs="+", default=["images_001", "images_002"],
                        help="image folders to fetch with --source full")
    parser.add_argument("--config", help="config file (default configs/default.yaml)")
    args = parser.parse_args()

    import kagglehub

    raw = load_config(args.config)["paths"]["raw_data"]
    target = raw / f"nih-{args.source}"
    target.mkdir(parents=True, exist_ok=True)
    handle = HANDLES[args.source]
    print(f"Downloading {handle} into {target} (this can take a while) ...")

    try:
        if args.source == "sample":
            kagglehub.dataset_download(handle, output_dir=str(target))
        else:
            for item in ["Data_Entry_2017.csv", *args.folders]:
                print(f"  - {item}")
                kagglehub.dataset_download(handle, path=item, output_dir=str(target / item.split(".")[0]))
    except Exception as exc:  # noqa: BLE001
        print(f"Download failed: {type(exc).__name__}: {exc}")
        print("If this is an authentication error, set up %USERPROFILE%/.kaggle/kaggle.json (see docstring).")
        return 1

    n_png = len(index_images(raw))
    print(f"Done. PNG images under {raw}: {n_png}")
    print(f"Label file: {find_label_csv(raw)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
