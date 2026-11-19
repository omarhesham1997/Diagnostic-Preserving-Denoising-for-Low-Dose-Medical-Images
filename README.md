# Diagnostic-Preserving Denoising for Low-Dose Medical Images

CSE681 Digital Image Processing, term project C21.
Team: Omar Hesham Hanafy Mohamed Nasr (2401812), Mohamed Ahmed Abdelsalaam (2400033).

We add simulated low-dose (Poisson-Gaussian) noise to NIH ChestX-ray14 images, denoise them with
a median filter, Anscombe-BM3D and a lightweight residual U-Net, and evaluate each method on
(1) pixel fidelity (PSNR, SSIM) and (2) diagnostic preservation: how much the predictions of a
frozen TorchXRayVision DenseNet-121 change after denoising.

## Setup

Requires [uv](https://docs.astral.sh/uv/). On Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then, from the repository root:

```powershell
uv sync                                   # creates .venv; first run downloads PyTorch (CUDA 13.0, ~3 GB)
uv run python scripts/check_env.py        # checks packages, GPU, BM3D and the classifier
uv run pytest                             # automatic tests
```

On Windows and Linux, `pyproject.toml` installs PyTorch from the official CUDA 13.0 index
(supports NVIDIA Blackwell GPUs; needs an NVIDIA driver with CUDA 13.0 or newer). macOS uses the default PyPI build.

`requirements.txt` is exported from the lock file for users without uv:

```powershell
uv export --no-hashes --emit-index-url --no-emit-project -o requirements.txt
```

## Data pipeline

```powershell
uv run python scripts/download_data.py      # Kaggle NIH random 5% sample -> data/raw
uv run python scripts/make_split.py         # 3,000 images, patient-level 2000/400/600 split -> splits/
uv run python scripts/preprocess.py         # 256x256 grayscale arrays -> data/processed/
uv run python scripts/make_noisy_sets.py    # fixed noisy val/test sets for low/medium/high
```

Each step prints checksums; re-running a step gives identical checksums.

## Repository layout

```
configs/default.yaml   all settings (paths, seeds, noise levels, training)
src/c21/               project code (imported as `c21`)
scripts/               runnable steps
tests/                 pytest checks
data/                  local dataset files (not committed; NIH data usage terms apply)
splits/                patient-level split CSV (committed, image IDs only)
outputs/               results, checkpoints, figures (not committed)
report/                milestone reports (LaTeX)
```

## Data and ethics

NIH ChestX-ray14 is used under its data usage terms via Kaggle. Images are never committed.
We make no attempt to re-identify patients, and results are computational benchmarks on
simulated noise, not clinically validated findings.

## Reproducibility

All randomness is seeded from `configs/default.yaml` (`seed: 42`). Evaluation noise is fixed
per image (`seed = base + image index`), so every method sees identical noisy test images.

## AI-tool disclosure

Parts of the code scaffolding were written with AI assistance (Claude). All results are run
and verified by the team.
