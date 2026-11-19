"""Convert the split images to 256x256 grayscale arrays.

Run:
    uv run python scripts/preprocess.py --limit 20   # quick check: picture only, nothing saved
    uv run python scripts/preprocess.py              # all images

Writes data/processed/{train,val,test}_256.npy (uint8, same order as the split CSV)
and outputs/figures/preprocess_check.png.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from joblib import Parallel, delayed

from c21.config import load_config
from c21.data import SPLITS, array_checksum, clean_array_path, index_images, load_and_resize, read_split
from c21.viz import save_grid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="config file (default configs/default.yaml)")
    parser.add_argument("--limit", type=int, help="only N images per split, nothing saved except the picture")
    parser.add_argument("--jobs", type=int, default=4, help="parallel workers (default 4)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    size = cfg["data"]["image_size"]
    split = read_split(cfg)
    images = index_images(cfg["paths"]["raw_data"])
    missing = [name for name in split["image"] if name not in images]
    if missing:
        print(f"{len(missing)} images in the split CSV are not on disk, e.g. {missing[:3]}")
        return 1

    print(f"{'split':6s}{'images':>8s}{'shape':>16s}{'mean':>7s}{'min':>5s}{'max':>5s}  checksum")
    preview, titles = [], []
    for s in SPLITS:
        rows = split[split["split"] == s]
        if args.limit:
            rows = rows.head(args.limit)
        arrays = Parallel(n_jobs=args.jobs)(
            delayed(load_and_resize)(images[name], size) for name in rows["image"]
        )
        stack = np.stack(arrays)
        status = "(not saved: --limit)"
        if not args.limit:
            out = clean_array_path(cfg, s)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, stack)
            status = array_checksum(stack)
        print(f"{s:6s}{len(stack):8d}{str(stack.shape):>16s}{stack.mean() / 255:7.3f}"
              f"{stack.min():5d}{stack.max():5d}  {status}")
        if s == "test":
            preview = [a / 255.0 for a in stack[:8]]
            titles = [f"{n}\n{f[:30]}" for n, f in zip(rows["image"][:8], rows["finding_labels"][:8])]

    fig = save_grid(preview, titles, cfg["paths"]["outputs"] / "figures" / "preprocess_check.png")
    print(f"\nSaved figure: {fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
