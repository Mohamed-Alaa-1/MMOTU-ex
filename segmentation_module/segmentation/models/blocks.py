"""
segmentation/models/blocks.py

Ported from https://github.com/Mohamed-Alaa-1/AuraVIT (src/models/blocks.py),
adapted for the MMOTU ovarian tumor segmentation extension.

Two fixes applied relative to the original source, both documented in
segmentation_extension_plan.md Section 4 and Section 2/3.1:

1. DeconvBlock: the original used nn.ConvTranspose2d(kernel_size=2, stride=2)
   for all upsampling. Kernel size divisible by stride is the textbook case
   for checkerboard artifacts, and this decoder chains ten of these calls
   across the four stages, concentrated exactly at high-frequency regions,
   i.e. the tumor boundary itself. Replaced with bilinear upsampling followed
   by a convolution, which is artifact-free and parameter-neutral to
   slightly cheaper.

2. LightweightASPP: the original hardcoded nn.Dropout2d(0.1) inside
   output_conv, disconnected from any config value. Now accepts a
   dropout_rate constructor argument (default 0.1 preserves the original
   behavior exactly for anyone not explicitly overriding it).

Everything else (DepthwiseSeparableConv, LightweightResBlock,
LightweightAtrousConv, LightweightAttentionGate, and the non-lightweight
ASPP/AttentionGate variants kept for reference/compatibility) is ported
unchanged, since it was already using depthwise separable convolutions
throughout and no bug was found there.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution - reduces parameters by 70-80%"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.depthwise.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.pointwise.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class LightweightResBlock(nn.Module):
    """Lightweight Residual Block using depthwise separable convolutions"""
    def __init__(self, in_c, out_c, stride=1, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            DepthwiseSeparableConv(in_c, out_c, stride=stride),
            nn.Dropout2d(dropout_rate),
            DepthwiseSeparableConv(out_c, out_c),
        )

        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.relu(self.layers(x) + self.shortcut(x))


class ResBlock(nn.Module):
    """Standard Residual Block, kept for reference/compatibility. Not used
    by LightweightAuraViT (see LightweightResBlock above)."""
    def __init__(self, in_c, out_c, stride=1, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.layers(x)
        x = x + shortcut
        return self.relu(x)


class DeconvBlock(nn.Module):
    """Upsampling block. FIXED relative to the original AuraViT source:
    bilinear upsample + conv instead of ConvTranspose2d(kernel_size=2,
    stride=2), to remove checkerboard artifact risk at tumor boundaries.
    Interface (in_c, out_c) -> forward(x) unchanged, so this is a drop-in
    replacement everywhere DeconvBlock is used in lightweight_auravit.py."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class LightweightAtrousConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.atrous_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=dilation,
                      dilation=dilation, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.atrous_conv(x)


class LightweightASPP(nn.Module):
    """FIXED relative to the original AuraViT source: dropout_rate is now a
    constructor argument threaded into output_conv's Dropout2d, instead of
    a hardcoded 0.1 disconnected from the rest of the model's config."""
    def __init__(self, in_channels, out_channels, rates, dropout_rate=0.1):
        super().__init__()
        intermediate_channels = out_channels // 2

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True)
        )
        self.atrous_blocks = nn.ModuleList([
            LightweightAtrousConv(in_channels, intermediate_channels, rate) for rate in rates
        ])
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True)
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(intermediate_channels * (len(rates) + 2), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        size = x.shape[2:]
        x_1x1 = self.conv_1x1(x)
        x_atrous = [block(x) for block in self.atrous_blocks]
        x_global = self.global_avg_pool(x)
        x_global = F.interpolate(x_global, size=size, mode='bilinear', align_corners=False)
        x_cat = torch.cat([x_1x1] + x_atrous + [x_global], dim=1)
        return self.output_conv(x_cat)


class ASPP(nn.Module):
    """Standard (non-lightweight) ASPP, kept for reference/compatibility.
    Not used by LightweightAuraViT."""
    def __init__(self, in_channels, out_channels, rates, dropout_rate=0.1):
        super().__init__()
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.atrous_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for rate in rates
        ])
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        size = x.shape[2:]
        x_1x1 = self.conv_1x1(x)
        x_atrous = [block(x) for block in self.atrous_blocks]
        x_global = self.global_avg_pool(x)
        x_global = F.interpolate(x_global, size=size, mode='bilinear', align_corners=False)
        x_cat = torch.cat([x_1x1] + x_atrous + [x_global], dim=1)
        return self.output_conv(x_cat)


class LightweightAttentionGate(nn.Module):
    def __init__(self, gate_channels, in_channels, inter_channels):
        super().__init__()
        inter_channels = inter_channels // 2

        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, gate, x):
        g1 = self.W_gate(gate)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # Sigmoid output is already bounded in (0, 1); this clamp is inert
        # in practice and kept only for parity with the original source.
        psi = torch.clamp(psi, min=1e-8, max=1.0)
        return x * psi


class AttentionGate(nn.Module):
    """Standard (non-lightweight) Attention Gate, kept for reference/
    compatibility. Not used by LightweightAuraViT."""
    def __init__(self, gate_channels, in_channels, inter_channels):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, gate, x):
        g1 = self.W_gate(gate)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        psi = torch.clamp(psi, min=1e-8, max=1.0)
        return x * psi
