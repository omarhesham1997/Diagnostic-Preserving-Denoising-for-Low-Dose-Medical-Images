"""Batch 0 check: Python packages, GPU, BM3D and the frozen classifier.

Run from the repository root:
    uv run python scripts/check_env.py

Prints one line per check:
    [PASS] works   [WARN] works but not ideal   [FAIL] must be fixed   [INFO] note only
Exit code 0 means no FAIL.
"""

from __future__ import annotations

import platform
import sys
import time
import traceback
from importlib import metadata

RESULTS: list[tuple[str, str]] = []  # (status, message)


def report(status: str, message: str) -> None:
    RESULTS.append((status, message))
    print(f"[{status}] {message}", flush=True)


def run(name: str, fn) -> None:
    """Run one check; any exception becomes a FAIL with a short reason."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 (we want to report every failure)
        report("FAIL", f"{name}: {type(exc).__name__}: {exc}")
        if "--verbose" in sys.argv:
            traceback.print_exc()


def check_python() -> None:
    v = sys.version_info
    status = "PASS" if v >= (3, 10) else "FAIL"
    report(status, f"Python {platform.python_version()} on {platform.platform()}")


def check_packages() -> None:
    names = [
        "numpy", "scipy", "scikit-image", "pandas", "pyyaml", "matplotlib",
        "torch", "torchvision", "torchxrayvision", "bm3d", "joblib", "tqdm", "pytest",
    ]
    found, missing = [], []
    for name in names:
        try:
            found.append(f"{name} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            missing.append(name)
    if missing:
        report("FAIL", f"Missing packages: {', '.join(missing)} (run `uv sync`)")
    report("PASS" if not missing else "INFO", "Packages: " + ", ".join(found))


def check_config() -> None:
    from c21.config import load_config

    cfg = load_config()
    sizes = cfg["data"]["split_sizes"]
    total = sum(sizes.values())
    if total != cfg["data"]["n_images"]:
        raise ValueError(f"split sizes add up to {total}, expected {cfg['data']['n_images']}")
    report("PASS", f"Config: {cfg['config_path'].name} (seed {cfg['seed']}, "
                   f"{sizes['train']}/{sizes['val']}/{sizes['test']} split)")

    raw = cfg["paths"]["raw_data"]
    if raw.exists() and any(raw.iterdir()):
        report("INFO", f"Raw data folder found: {raw}")
    else:
        report("INFO", f"Raw data folder not filled yet: {raw} (needed in Batch 2)")


def check_torch_gpu() -> None:
    import torch

    cuda_build = torch.version.cuda
    if cuda_build is None:
        report("WARN", f"PyTorch {torch.__version__} is a CPU-only build: training will be slow. "
                       "On Windows/Linux, `uv sync` should install the CUDA 13.0 build.")
        return
    report("PASS", f"PyTorch {torch.__version__} built with CUDA {cuda_build}")

    if not torch.cuda.is_available():
        report("WARN", "CUDA build installed but no usable GPU found (check the NVIDIA driver)")
        return

    props = torch.cuda.get_device_properties(0)
    arch = f"sm_{props.major}{props.minor}"
    supported = torch.cuda.get_arch_list()
    if arch not in supported:
        raise RuntimeError(f"GPU is {arch} but this PyTorch only supports {supported}. "
                           "Install the CUDA 13.0 build of PyTorch (run `uv sync`).")

    # Run a real convolution on the GPU and compare with the CPU result.
    x = torch.randn(4, 1, 64, 64)
    w = torch.randn(8, 1, 3, 3)
    cpu = torch.nn.functional.conv2d(x, w, padding=1)
    gpu = torch.nn.functional.conv2d(x.cuda(), w.cuda(), padding=1).cpu()
    diff = (cpu - gpu).abs().max().item()
    if diff > 1e-3:
        raise RuntimeError(f"GPU result differs from CPU by {diff:.2e}")

    mem_gb = props.total_memory / 1024**3
    report("PASS", f"GPU: {props.name}, {mem_gb:.1f} GB, {arch}, test convolution OK")


def check_bm3d() -> None:
    import bm3d
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio as psnr

    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:64, 0:64] / 63.0
    clean = 0.2 + 0.6 * (xx * yy)  # smooth test pattern
    noisy = clean + rng.normal(0, 0.1, clean.shape)
    t0 = time.perf_counter()
    denoised = bm3d.bm3d(noisy, sigma_psd=0.1)
    secs = time.perf_counter() - t0
    before, after = psnr(clean, noisy, data_range=1), psnr(clean, denoised, data_range=1)
    if after <= before:
        raise RuntimeError(f"BM3D did not improve the test image ({before:.1f} -> {after:.1f} dB)")
    report("PASS", f"BM3D: runs, test pattern {before:.1f} -> {after:.1f} dB ({secs:.2f} s)")


def check_classifier() -> None:
    import torch
    import torchxrayvision as xrv

    from c21.config import load_config
    from c21.constants import NIH_LABELS

    weights = load_config()["classifier"]["weights"]
    t0 = time.perf_counter()
    model = xrv.models.DenseNet(weights=weights)  # downloads once, then cached
    model.eval()
    # TorchXRayVision expects pixel values scaled to [-1024, 1024].
    dummy = torch.rand(1, 1, 224, 224, generator=torch.Generator().manual_seed(0)) * 2048 - 1024
    with torch.no_grad():
        out = model(dummy)

    labels = list(model.pathologies)
    found = [name for name in NIH_LABELS if name in labels]
    if out.shape != (1, len(labels)):
        raise RuntimeError(f"unexpected output shape {tuple(out.shape)}")
    if len(found) != len(NIH_LABELS):
        missing = sorted(set(NIH_LABELS) - set(found))
        raise RuntimeError(f"NIH labels missing from the model outputs: {missing}")
    report("PASS", f"Classifier {weights}: {len(labels)} outputs, "
                   f"{len(found)}/{len(NIH_LABELS)} NIH labels found ({time.perf_counter() - t0:.1f} s)")


def main() -> int:
    print("C21 environment check")
    print("=" * 60)
    run("Python", check_python)
    run("Packages", check_packages)
    run("Config", check_config)
    run("PyTorch/GPU", check_torch_gpu)
    run("BM3D", check_bm3d)
    run("Classifier", check_classifier)
    print("=" * 60)

    fails = sum(1 for status, _ in RESULTS if status == "FAIL")
    warns = sum(1 for status, _ in RESULTS if status == "WARN")
    if fails:
        print(f"Result: {fails} check(s) FAILED. Paste this output to fix them.")
    else:
        print(f"Result: all required checks passed ({warns} warning(s)).")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
