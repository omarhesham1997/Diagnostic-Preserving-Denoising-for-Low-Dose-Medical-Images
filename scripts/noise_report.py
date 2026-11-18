"""Batch 1 report: noise levels, Anscombe check, inverse bias, and a picture.

Run from the repository root:
    uv run python scripts/noise_report.py
    uv run python scripts/noise_report.py --image path/to/any_xray.png

Without --image it uses a standard test photo (scikit-image "camera"), so the
PSNR values are only a guide. Batch 2 measures the real values on the chest
X-ray test set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from skimage import data, transform  # noqa: E402
from skimage.metrics import peak_signal_noise_ratio, structural_similarity  # noqa: E402

from c21.config import load_config  # noqa: E402
from c21.noise import (  # noqa: E402
    add_noise,
    all_levels,
    gat_forward,
    gat_inverse_algebraic,
    gat_inverse_unbiased,
)

M1_CLAIMED_PSNR = {"low": 30.0, "medium": 24.0, "high": 19.0}


def load_image(path: str | None, size: int) -> tuple[np.ndarray, str]:
    if path:
        from PIL import Image

        img = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
        name = Path(path).name
    else:
        img = data.camera().astype(np.float64) / 255.0
        name = "scikit-image camera (test photo)"
    img = transform.resize(img, (size, size), anti_aliasing=True)
    return img.astype(np.float32), name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", help="optional grayscale image to use instead of the test photo")
    parser.add_argument("--seeds", type=int, default=5, help="noise repeats to average (default 5)")
    args = parser.parse_args()

    cfg = load_config()
    levels = all_levels(cfg)
    clean, name = load_image(args.image, cfg["data"]["image_size"])

    # 1) How noisy is each level?
    print(f"Image: {name}, {clean.shape[0]}x{clean.shape[1]}, mean brightness {clean.mean():.3f}\n")
    print("1) Noisy-image quality per level (average over", args.seeds, "noise seeds)")
    print(f"   {'level':8s}{'lambda':>8s}{'sigma':>8s}{'PSNR dB':>10s}{'SSIM':>8s}{'M1 claimed':>12s}")
    noisy_examples = {}
    for lv in levels:
        psnrs, ssims = [], []
        for s in range(args.seeds):
            y = np.clip(add_noise(clean, lv, seed=s), 0, 1)
            psnrs.append(peak_signal_noise_ratio(clean, y, data_range=1.0))
            ssims.append(structural_similarity(clean, y, data_range=1.0))
            if s == 0:
                noisy_examples[lv.name] = (y, psnrs[0])
        print(f"   {lv.name:8s}{lv.lambda_photons:8.0f}{lv.sigma_electronic:8.2f}"
              f"{np.mean(psnrs):10.2f}{np.mean(ssims):8.3f}{'~' + str(M1_CLAIMED_PSNR.get(lv.name, '?')):>12s}")

    # 2) Does the Anscombe transform make the noise std ~1 everywhere?
    flat_pixels = 500_000
    print("\n2) Anscombe check: noise std after the transform (target 1.00)")
    print(f"   {'level':8s}{'x=0.1':>8s}{'x=0.5':>8s}{'x=0.9':>8s}")
    for lv in levels:
        stds = []
        for x in (0.1, 0.5, 0.9):
            y = add_noise(np.full(flat_pixels, x, np.float32), lv, seed=10)
            stds.append(gat_forward(y, lv.lambda_photons, lv.sigma_electronic).std())
        print(f"   {lv.name:8s}" + "".join(f"{s:8.3f}" for s in stds))

    # 3) Bias of the two inverse transforms (ideal denoiser, flat x = 0.5)
    print("\n3) Inverse-transform bias at x = 0.5 (closer to 0 is better)")
    print(f"   {'level':8s}{'algebraic':>12s}{'unbiased':>12s}")
    for lv in levels:
        y = add_noise(np.full(flat_pixels, 0.5, np.float32), lv, seed=11)
        mean_d = gat_forward(y, lv.lambda_photons, lv.sigma_electronic).mean()
        b_alg = gat_inverse_algebraic(mean_d, lv.lambda_photons, lv.sigma_electronic) - 0.5
        b_unb = gat_inverse_unbiased(mean_d, lv.lambda_photons, lv.sigma_electronic) - 0.5
        print(f"   {lv.name:8s}{b_alg:+12.4f}{b_unb:+12.4f}")

    # 4) Picture: clean + each level, with a zoomed crop
    out = cfg["paths"]["outputs"] / "figures" / "noise_demo.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    panels = [("clean", clean, None)] + [(k, v[0], v[1]) for k, v in noisy_examples.items()]
    h, w = clean.shape
    crop = (slice(h // 2 - 32, h // 2 + 32), slice(w // 2 - 32, w // 2 + 32))
    fig, axes = plt.subplots(2, len(panels), figsize=(3 * len(panels), 6.2))
    for col, (title, img, psnr) in enumerate(panels):
        label = title if psnr is None else f"{title} ({psnr:.1f} dB)"
        axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(label)
        axes[1, col].imshow(img[crop], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[1, col].set_title("zoom (center 64x64)", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
