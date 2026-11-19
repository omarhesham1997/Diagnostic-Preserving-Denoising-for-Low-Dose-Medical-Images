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


def save_training_curves(log, noisy_psnr: float, path: Path) -> Path:
    """Two panels: L1 loss (train/val) and validation PSNR per epoch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(log["epoch"], log["train_l1"], label="train (new noise each batch)")
    ax1.plot(log["epoch"], log["val_l1"], label="validation (fixed noise)")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("L1 loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(log["epoch"], log["val_psnr"], color="tab:green", label="ResU-Net")
    ax2.axhline(noisy_psnr, color="gray", linestyle="--", label="noisy input")
    best = log["val_psnr"].idxmax()
    ax2.scatter([log.loc[best, "epoch"]], [log.loc[best, "val_psnr"]], color="tab:red", zorder=3, label="best")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("validation PSNR (dB)")
    ax2.legend()
    ax2.grid(alpha=0.3)
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


# ---------------------------------------------------------------------------
# Batch 6 figures
# ---------------------------------------------------------------------------

# Validated categorical palette (dataviz reference, light mode): slots 1-3 pass all-pairs CVD checks.
# "Noisy" is a neutral reference point, drawn in the muted gray.
METHOD_STYLE = {
    "noisy": {"color": "#898781", "marker": "o"},
    "median": {"color": "#1baf7a", "marker": "s"},
    "bm3d": {"color": "#eb6834", "marker": "D"},
    "resunet": {"color": "#2a78d6", "marker": "^"},
}
INK, INK_2 = "#0b0b0b", "#52514e"


def detail_center(clean: np.ndarray, window: int = 64, margin: int = 40) -> tuple[int, int]:
    """Center of the window with the most local detail (highest local std) in the clean image.

    An objective rule for where to zoom, so crops are not hand-picked.
    """
    from scipy.ndimage import uniform_filter

    x = clean.astype(np.float64)
    var = uniform_filter(x * x, window) - uniform_filter(x, window) ** 2
    h, w = x.shape
    inner = var[margin:h - margin, margin:w - margin]
    r, c = np.unravel_index(np.argmax(inner), inner.shape)
    return int(r + margin), int(c + margin)


def save_qualitative(clean: np.ndarray, panels: list[tuple[str, np.ndarray]], title: str, path: Path,
                     zoom: int = 64, err_vmax: float = 0.15) -> Path:
    """Rows: full image (crop marked), zoomed detail, absolute error |output - clean|."""
    from matplotlib.patches import Rectangle

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    r0, c0 = detail_center(clean, zoom)
    rs, cs = slice(r0 - zoom // 2, r0 + zoom // 2), slice(c0 - zoom // 2, c0 + zoom // 2)
    n = len(panels)
    fig, axes = plt.subplots(3, n, figsize=(2.7 * n, 8.6), squeeze=False)
    err_im = None
    for col, (label, img) in enumerate(panels):
        img = np.clip(img, 0, 1)
        axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[0, col].add_patch(Rectangle((cs.start, rs.start), zoom, zoom, fill=False, edgecolor="#eda100", lw=1.5))
        axes[0, col].set_title(label, fontsize=8.5, color=INK)
        axes[1, col].imshow(img[rs, cs], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        if col == 0:
            axes[2, col].text(0.5, 0.5, "absolute error\n|output - clean|\n(darker = larger)", ha="center",
                              va="center", fontsize=9, color=INK_2, transform=axes[2, col].transAxes)
        else:
            err_im = axes[2, col].imshow(np.abs(img - clean), cmap="Blues", vmin=0, vmax=err_vmax)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[1, 0].set_ylabel(f"zoom {zoom}x{zoom}", fontsize=8.5, color=INK_2)
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 0.94, 0.97))
    if err_im is not None:
        cax = fig.add_axes((0.945, 0.06, 0.012, 0.25))
        cb = fig.colorbar(err_im, cax=cax)
        cb.ax.tick_params(labelsize=7, colors=INK_2)
        cb.outline.set_visible(False)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _place_labels(ax, fig, points: list[tuple[float, float, str]]) -> None:
    """Direct labels next to points, trying a few offsets so labels do not overlap each other or the points."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = []
    pts_disp = [ax.transData.transform((x, y)) for x, y, _ in points]
    offsets = [(9, 7, "left"), (9, -15, "left"), (-9, 7, "right"), (-9, -15, "right"), (9, 18, "left"), (9, -26, "left")]
    for (x, y, text), _ in zip(points, pts_disp):
        for dx, dy, ha in offsets:
            t = ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha,
                            fontsize=9, color=INK)
            bb = t.get_window_extent(renderer).expanded(1.05, 1.15)
            hits_label = any(bb.overlaps(other) for other in placed)
            hits_point = any(bb.contains(px, py) for px, py in pts_disp)
            if not hits_label and not hits_point:
                placed.append(bb)
                break
            t.remove()
        else:  # nothing free: keep the first position
            t = ax.annotate(text, (x, y), textcoords="offset points", xytext=offsets[0][:2], fontsize=9, color=INK)
            placed.append(t.get_window_extent(renderer))


def save_tradeoff_chart(summary, ci: dict, path: Path, title: str) -> Path:
    """Pixel fidelity (PSNR, right = better) vs diagnostic change (delta_p, down = better), one point per method.

    `ci` maps method -> (psnr_halfwidth, delta_p_halfwidth) for 95% CI error bars.
    """
    from matplotlib.lines import Line2D

    from c21.evaluation import display_name

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    handles, points = [], []
    for _, r in summary.iterrows():
        m = r["method"]
        style = METHOD_STYLE.get(m, {"color": INK_2, "marker": "o"})
        xerr, yerr = ci.get(m, (0, 0))
        ax.errorbar(r["psnr_mean"], r["delta_p_mean"], xerr=xerr, yerr=yerr, fmt="none",
                    ecolor=style["color"], elinewidth=2, capsize=0, zorder=2)
        ax.plot(r["psnr_mean"], r["delta_p_mean"], linestyle="none", marker=style["marker"], color=style["color"],
                markersize=9, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        handles.append(Line2D([], [], linestyle="none", marker=style["marker"], color=style["color"],
                              markersize=8, markeredgecolor="white", label=display_name(m)))
        points.append((r["psnr_mean"], r["delta_p_mean"], display_name(m)))
    ax.set_xlabel("PSNR (dB)  -  higher is better", color=INK_2)
    ax.set_ylabel("mean $\\Delta p$  -  lower is better", color=INK_2)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors=INK_2, labelsize=8.5)
    x, y = summary["psnr_mean"], summary["delta_p_mean"]
    ax.set_xlim(x.min() - 2.5, x.max() + 4.5)
    ax.set_ylim(0, max(y.max() * 1.25, 0.05))
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=len(handles), handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    _place_labels(ax, fig, points)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
