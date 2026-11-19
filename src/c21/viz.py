"""Small plotting helpers shared by the scripts (no screen needed)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def save_panels_with_zoom(panels: list[tuple[str, np.ndarray]], path: Path, zoom: int = 64) -> Path:
    """Save images side by side (top row) with a zoomed center crop below each."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = panels[0][1].shape
    crop = (slice(h // 2 - zoom // 2, h // 2 + zoom // 2), slice(w // 2 - zoom // 2, w // 2 + zoom // 2))
    fig, axes = plt.subplots(2, len(panels), figsize=(3 * len(panels), 6.2), squeeze=False)
    for col, (title, img) in enumerate(panels):
        axes[0, col].imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(title)
        axes[1, col].imshow(np.clip(img[crop], 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[1, col].set_title(f"zoom (center {zoom}x{zoom})", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_grid(images: list[np.ndarray], titles: list[str], path: Path, cols: int = 4) -> Path:
    """Save a simple grid of grayscale images."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, img, title in zip(axes.ravel(), images, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
