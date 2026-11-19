"""Batch 5 tests: ResU-Net architecture, inference helper, checkpoints, one training step."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from c21.models.resunet import (
    ResUNet,
    count_parameters,
    denoise_numpy,
    load_checkpoint,
    save_checkpoint,
)
from c21.noise import add_noise_torch


def test_parameter_count_matches_the_m1_design():
    # 16/32/64/128 channels, two 3x3 convs per level, 1x1 up-convs, 1x1 head (biases included).
    # M1 estimated ~384k; this is the exact count of the described design.
    assert count_parameters(ResUNet()) == 449_489


def test_output_has_input_shape():
    y = torch.rand(2, 1, 256, 256)
    assert ResUNet()(y).shape == y.shape


def test_untrained_network_is_the_identity():
    """The last layer starts at zero, so x_hat = y - 0 = y before training."""
    y = torch.rand(1, 1, 64, 64)
    assert torch.equal(ResUNet()(y), y)


def test_no_normalization_layers_and_leaky_relu_slope():
    model = ResUNet()
    assert not any(isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)) for m in model.modules())
    slopes = {m.negative_slope for m in model.modules() if isinstance(m, nn.LeakyReLU)}
    assert slopes == {0.1}


def test_denoise_numpy_handles_odd_sizes_and_clips():
    model = ResUNet()
    noisy = np.random.default_rng(0).normal(0.5, 0.5, (3, 50, 70)).astype(np.float32)  # not multiples of 8
    out = denoise_numpy(model, noisy, device="cpu", batch_size=2)
    assert out.shape == noisy.shape and out.dtype == np.float32
    assert out.min() >= 0 and out.max() <= 1
    assert np.allclose(out, np.clip(noisy, 0, 1), atol=1e-6)  # identity at init, then clipped


def test_one_training_step_reduces_the_loss():
    torch.manual_seed(0)
    model = ResUNet()
    x = torch.rand(4, 1, 64, 64)
    y = add_noise_torch(x, 50.0, 0.02, generator=torch.Generator().manual_seed(0))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(20):
        loss = F.l1_loss(model(y), x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_checkpoint_round_trip_gives_identical_outputs(tmp_path):
    model = ResUNet()
    with torch.no_grad():  # make the head non-zero so the test is meaningful
        model.head.weight.normal_(0, 0.1)
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, {"epoch": 3, "val_psnr": 30.0})
    loaded, ckpt = load_checkpoint(path)
    y = torch.rand(1, 1, 64, 64)
    model.eval()
    assert torch.allclose(model(y), loaded(y))
    assert ckpt["epoch"] == 3 and ckpt["parameters"] == 449_489


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_model_runs_on_gpu():
    model = ResUNet().cuda()
    y = torch.rand(2, 1, 256, 256, device="cuda")
    assert model(y).shape == y.shape
