"""
segmentation_module/segmentation/models/baselines.py

Standard segmentation baseline architectures for fair comparison with LAURA.
All models accept [B, 1, H, W] single-channel ultrasound input and produce
[B, 1, H, W] sigmoid-ready logits (NOT probabilities).

Supported architectures:
  - UNet            : Classic U-Net (Ronneberger et al. 2015)
  - AttentionUNet   : Attention U-Net (Oktay et al. 2018)
  - DeepLabV3Plus   : DeepLabV3+ with MobileNetV3-Large encoder (Chen et al. 2018)
  - UNetPlusPlus    : U-Net++ with dense skip connections (Zhou et al. 2018)

Usage
-----
from segmentation.models.baselines import get_baseline_model

model = get_baseline_model("unet", in_channels=1, num_classes=1)

All models trained and evaluated with EXACTLY the same patient-level split
and preprocessing pipeline as LAURA (see train_segmentation.py).

Dependencies: torch, torchvision.
Optional: segmentation_models_pytorch (smp) for DeepLabV3+ and U-Net++.
If smp is not installed, a lightweight pure-PyTorch stub is used that
raises NotImplementedError at forward() time with installation instructions.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper blocks
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks (shared by UNet and AttentionUNet)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Down(nn.Module):
    """MaxPool2x2 followed by DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Up(nn.Module):
    """Bilinear upsampling + DoubleConv (used by plain U-Net)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if needed (handles non-power-of-two inputs)
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        x  = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """Classic U-Net (Ronneberger et al. 2015).

    Reference: https://arxiv.org/abs/1505.04597

    Args:
        in_channels:  Number of input channels (1 for grayscale ultrasound).
        num_classes:  Number of output channels (1 for binary segmentation).
        features:     Channel widths for each encoder stage.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        features: List[int] = None,
    ) -> None:
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.inc  = _DoubleConv(in_channels, features[0])
        self.down = nn.ModuleList([
            _Down(features[i], features[i + 1])
            for i in range(len(features) - 1)
        ])
        # Bottleneck
        self.bottleneck = _Down(features[-1], features[-1] * 2)

        # Decoder (reversed)
        up_chs = [features[-1] * 2] + list(reversed(features))
        self.up = nn.ModuleList([
            _Up(up_chs[i] + up_chs[i + 1], up_chs[i + 1])
            for i in range(len(up_chs) - 1)
        ])
        self.outc = nn.Conv2d(features[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for d in self.down:
            skips.append(d(skips[-1]))
        x = self.bottleneck(skips[-1])
        for up, skip in zip(self.up, reversed(skips)):
            x = up(x, skip)
        return self.outc(x)


# ---------------------------------------------------------------------------
# Attention U-Net
# ---------------------------------------------------------------------------

class _AttentionGate(nn.Module):
    """Attention gate from Oktay et al. 2018.

    Computes a spatial attention map from gating signal (g) and
    skip connection (x), then scales x element-wise.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Align spatial dims
        if g1.shape != x1.shape:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear",
                               align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class _AttUp(nn.Module):
    """Attention U-Net decoder block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up      = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.att     = _AttentionGate(F_g=in_ch, F_l=skip_ch, F_int=out_ch // 2)
        self.conv    = _DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x    = self.up(x)
        skip = self.att(g=x, x=skip)
        dh   = skip.size(2) - x.size(2)
        dw   = skip.size(3) - x.size(3)
        x    = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class AttentionUNet(nn.Module):
    """Attention U-Net (Oktay et al. 2018).

    Reference: https://arxiv.org/abs/1804.03999

    Architecture identical to U-Net but with attention gates on every
    skip connection in the decoder path.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output channels.
        features:    Encoder channel widths.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        features: List[int] = None,
    ) -> None:
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.inc  = _DoubleConv(in_channels, features[0])
        self.down = nn.ModuleList([
            _Down(features[i], features[i + 1])
            for i in range(len(features) - 1)
        ])
        self.bottleneck = _Down(features[-1], features[-1] * 2)

        up_in   = [features[-1] * 2] + list(reversed(features))
        up_skip = list(reversed(features))
        self.up = nn.ModuleList([
            _AttUp(up_in[i], up_skip[i], up_skip[i])
            for i in range(len(up_skip))
        ])
        self.outc = nn.Conv2d(features[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for d in self.down:
            skips.append(d(skips[-1]))
        x = self.bottleneck(skips[-1])
        for up, skip in zip(self.up, reversed(skips)):
            x = up(x, skip)
        return self.outc(x)


# ---------------------------------------------------------------------------
# U-Net++ and DeepLabV3+  via segmentation_models_pytorch (smp)
# ---------------------------------------------------------------------------

def _try_import_smp():
    try:
        import segmentation_models_pytorch as smp  # noqa: F401
        return smp
    except ImportError:
        return None


def _smp_stub(name: str):
    """Return a module that raises NotImplementedError at forward time."""
    class _Stub(nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()
            self._name = name

        def forward(self, x):
            raise NotImplementedError(
                f"{self._name} requires `segmentation_models_pytorch`.\n"
                "Install it with:  pip install segmentation-models-pytorch"
            )
    return _Stub


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ with MobileNetV3-Large encoder (Chen et al. 2018).

    Requires segmentation_models_pytorch:
        pip install segmentation-models-pytorch

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output channels.
        encoder_weights: None (random init) or 'imagenet'.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        encoder_weights=None,
    ) -> None:
        super().__init__()
        smp = _try_import_smp()
        if smp is None:
            self._model = _smp_stub("DeepLabV3Plus")()
        else:
            self._model = smp.DeepLabV3Plus(
                encoder_name="mobilenet_v2",
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                activation=None,  # raw logits
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class UNetPlusPlus(nn.Module):
    """U-Net++ with MobileNetV2 encoder (Zhou et al. 2018).

    Requires segmentation_models_pytorch:
        pip install segmentation-models-pytorch

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output channels.
        encoder_weights: None (random init) or 'imagenet'.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        encoder_weights=None,
    ) -> None:
        super().__init__()
        smp = _try_import_smp()
        if smp is None:
            self._model = _smp_stub("UNetPlusPlus")()
        else:
            self._model = smp.UnetPlusPlus(
                encoder_name="mobilenet_v2",
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                activation=None,  # raw logits
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY = {
    "unet":          UNet,
    "attention_unet": AttentionUNet,
    "deeplabv3plus": DeepLabV3Plus,
    "unetplusplus":  UNetPlusPlus,
}


def get_baseline_model(
    name: str,
    in_channels: int = 1,
    num_classes: int = 1,
    **kwargs,
) -> nn.Module:
    """Instantiate a baseline segmentation model by name.

    Args:
        name:        One of 'unet', 'attention_unet', 'deeplabv3plus',
                     'unetplusplus'.
        in_channels: Input channels (1 for grayscale ultrasound).
        num_classes: Output channels (1 for binary segmentation).
        **kwargs:    Additional keyword arguments forwarded to the constructor.

    Returns:
        nn.Module ready for training/inference (outputs raw logits).
    """
    name = name.lower().replace("-", "_").replace(" ", "_")
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown baseline model '{name}'. "
            f"Choose from: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](in_channels=in_channels, num_classes=num_classes, **kwargs)


def count_parameters(model: nn.Module) -> dict:
    """Return total and trainable parameter counts (in millions)."""
    total    = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params_M":     total     / 1e6,
        "trainable_params_M": trainable / 1e6,
    }
