# Course Code
CSE681



# Diagnostic-Preserving Denoising for Low-Dose Medical Images

**Objectives**
- Train a denoiser for simulated low-dose X-ray/CT noise.
- Evaluate both by pixel-fidelity metrics and by whether a frozen diagnostic classifier's predictions are preserved after denoising.

**References**
- Team-driven: survey 2024–2026 low-dose medical denoising literature during M1.
- Methodology reference: "Deep Learning Based Medical Image Compression Using Cross Attention Learning and Wavelet Transform," *Scientific Reports* 2025 (ROI-relevance evaluation idea).

**Dataset:** NIH ChestX-ray14 public sample — [link](https://www.kaggle.com/datasets/nih-chest-xrays/data), with synthetically added low-dose-style noise (paired real low/high-dose public data is scarce — state this limitation explicitly in the report).

**Baseline:** a classical denoising filter (BM3D/median) at the same noise level. **Compute tier:** CPU-OK / GPU-light.

> **Ethics note:** same as Project 10 — respect the dataset's license, no re-identification attempts (see *project-instructions.md*, §9.3).

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
|---|---|---|---|---|---|
| 1 | 3 | ChestX-ray14 sample + license review; noise-simulation pipeline | Design denoiser architecture | Select & freeze a pretrained diagnostic classifier for the preservation metric | Proposal + license/ethics statement |
| 2 | 6 | Classical-filter baseline, both pixel and classifier-preservation metrics | Denoiser, first checkpoint | Preliminary pixel + classifier-preservation comparison | Progress report, 1 comparison |
| 3 | 9 | Final baseline table | Final denoiser + noise-level ablation | Full comparison on both metric families | Final report, repo, demo |
