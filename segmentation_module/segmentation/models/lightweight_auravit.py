"""
segmentation/models/lightweight_auravit.py

Ported from https://github.com/Mohamed-Alaa-1/AuraVIT
(src/models/lightweight_auravit.py), adapted for MMOTU ovarian tumor
segmentation. See segmentation_extension_plan.md for the full adaptation
and lightweighting rationale.

Fix applied relative to the original source (in addition to the
DeconvBlock and LightweightASPP fixes already made in blocks.py):

- dropout_rate is no longer read inconsistently. The original used
  cf["dropout_rate"] for the encoder and cf.get("block_dropout_rate", 0.1)
  for the decoder, where "block_dropout_rate" was never actually present
  in the example config, so the decoder silently always used 0.1
  regardless of what dropout_rate was set to. This version requires
  cf["block_dropout_rate"] and cf["aspp_dropout_rate"] to be present
  (auravit_config.build_config always sets them, defaulting both equal to
  dropout_rate unless explicitly overridden), so there is no silent
  fallback anymore. Building a config dict by hand instead of through
  build_config() will now raise a KeyError immediately rather than
  silently drifting, which is the intended behavior.

NaN handling is preserved unchanged from the original (raises ValueError
at the input, after every transformer layer, and at the final output).
This is intentional defensive behavior, not a bug, but it does not match
the soft skip-and-continue pattern used elsewhere in this project's
training loop for the classification backbones. The segmentation trainer
(segmentation/trainer.py) is responsible for catching this ValueError and
skipping the batch, exactly the way GradientMonitor already does for
classification. See segmentation_extension_plan.md Section 1, point 4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .blocks import (
    LightweightASPP, LightweightAttentionGate, DeconvBlock, LightweightResBlock
)

_REQUIRED_KEYS = (
    "image_size", "num_layers", "hidden_dim", "mlp_dim", "num_heads",
    "dropout_rate", "patch_size", "num_channels", "num_patches",
    "block_dropout_rate", "aspp_dropout_rate",
)


class LightweightAuraViT(nn.Module):
    """
    Lightweight ViT encoder + ASPP + attention-gated U-Net decoder for
    binary tumor segmentation. Build the config dict via
    segmentation.models.auravit_config.build_config() (or one of the
    PRESETS), do not hand-write it, since several keys used here
    (num_patches, block_dropout_rate, aspp_dropout_rate) are derived
    values with correctness constraints described in auravit_config.py.
    """

    def __init__(self, cf: dict):
        super().__init__()
        missing = [k for k in _REQUIRED_KEYS if k not in cf]
        if missing:
            raise KeyError(
                f"Config missing required keys {missing}. Build configs via "
                f"segmentation.models.auravit_config.build_config(...) rather "
                f"than a hand-written dict, so these are always derived "
                f"consistently."
            )
        self.cf = cf

        # ViT Encoder
        self.patch_embed = nn.Sequential(
            nn.Linear(cf["patch_size"] * cf["patch_size"] * cf["num_channels"], cf["hidden_dim"]),
            nn.LayerNorm(cf["hidden_dim"]),
            nn.Dropout(cf["dropout_rate"])
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, cf["num_patches"], cf["hidden_dim"]))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_dropout = nn.Dropout(cf["dropout_rate"])

        self.trans_encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cf["hidden_dim"], nhead=cf["num_heads"],
                dim_feedforward=cf["mlp_dim"], dropout=cf["dropout_rate"],
                activation=F.gelu, batch_first=True, norm_first=True
            ) for _ in range(cf["num_layers"])
        ])

        self.skip_norms = nn.ModuleList([
            nn.LayerNorm(cf["hidden_dim"]) for _ in range(4)
        ])

        # ASPP (dropout_rate now threaded through, see module docstring)
        aspp_rates = cf.get("aspp_rates", [6, 12, 18])
        self.aspp = LightweightASPP(
            cf["hidden_dim"], cf["hidden_dim"], rates=aspp_rates,
            dropout_rate=cf["aspp_dropout_rate"],
        )

        # Attention Gates
        self.att_gate_1 = LightweightAttentionGate(256, 256, 128)
        self.att_gate_2 = LightweightAttentionGate(128, 128, 64)
        self.att_gate_3 = LightweightAttentionGate(64, 64, 32)
        self.att_gate_4 = LightweightAttentionGate(32, 32, 16)

        # Decoder (block_dropout_rate now threaded consistently, see module docstring)
        block_dropout_rate = cf["block_dropout_rate"]

        self.seg_d1 = DeconvBlock(cf["hidden_dim"], 256)
        self.seg_s1 = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 256),
            LightweightResBlock(256, 256, dropout_rate=block_dropout_rate)
        )
        self.seg_c1 = nn.Sequential(
            LightweightResBlock(512, 256, dropout_rate=block_dropout_rate),
            LightweightResBlock(256, 256, dropout_rate=block_dropout_rate)
        )

        self.seg_d2 = DeconvBlock(256, 128)
        self.seg_s2 = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 128),
            LightweightResBlock(128, 128, dropout_rate=block_dropout_rate),
            DeconvBlock(128, 128),
            LightweightResBlock(128, 128, dropout_rate=block_dropout_rate)
        )
        self.seg_c2 = nn.Sequential(
            LightweightResBlock(256, 128, dropout_rate=block_dropout_rate),
            LightweightResBlock(128, 128, dropout_rate=block_dropout_rate)
        )

        self.seg_d3 = DeconvBlock(128, 64)
        self.seg_s3 = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 64),
            LightweightResBlock(64, 64, dropout_rate=block_dropout_rate),
            DeconvBlock(64, 64),
            LightweightResBlock(64, 64, dropout_rate=block_dropout_rate),
            DeconvBlock(64, 64),
            LightweightResBlock(64, 64, dropout_rate=block_dropout_rate)
        )
        self.seg_c3 = nn.Sequential(
            LightweightResBlock(128, 64, dropout_rate=block_dropout_rate),
            LightweightResBlock(64, 64, dropout_rate=block_dropout_rate)
        )

        self.seg_d4 = DeconvBlock(64, 32)
        self.seg_s4 = nn.Sequential(
            LightweightResBlock(cf["num_channels"], 32, dropout_rate=block_dropout_rate),
            LightweightResBlock(32, 32, dropout_rate=block_dropout_rate)
        )
        self.seg_c4 = nn.Sequential(
            LightweightResBlock(64, 32, dropout_rate=block_dropout_rate),
            LightweightResBlock(32, 32, dropout_rate=block_dropout_rate)
        )

        self.seg_output = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        nn.init.xavier_uniform_(self.seg_output.weight, gain=0.1)
        nn.init.constant_(self.seg_output.bias, 0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if torch.isnan(inputs).any():
            raise ValueError("NaN detected in input tensor")
        if inputs.size(1) != self.cf["num_channels"]:
            raise ValueError(
                f"Input has {inputs.size(1)} channels but model was built "
                f"with num_channels={self.cf['num_channels']}. If MMOTU "
                f"images are still RGB at this point in the pipeline, "
                f"convert to grayscale before calling forward (see "
                f"segmentation_extension_plan.md Section 1, point 1), or "
                f"rebuild the config with num_channels=3."
            )

        # ViT Encoder: split into non-overlapping patches, flatten, project
        p = self.cf["patch_size"]
        patches = inputs.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(inputs.size(0), inputs.size(1), -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4)
        patches = patches.contiguous().view(inputs.size(0), self.cf["num_patches"], -1)
        patch_embed = self.patch_embed(patches)

        x = self.pos_dropout(patch_embed + self.pos_embed)

        num_layers = len(self.trans_encoder_layers)
        if num_layers == 8:
            skip_connection_index = [1, 3, 5, 7]
        elif num_layers == 6:
            skip_connection_index = [1, 2, 4, 5]
        elif num_layers == 4:
            skip_connection_index = [0, 1, 2, 3]
        else:  # 12 layers
            skip_connection_index = [2, 5, 8, 11]

        skip_connections = []
        for i, layer in enumerate(self.trans_encoder_layers):
            x = layer(x)
            if torch.isnan(x).any():
                raise ValueError(f"NaN detected after transformer layer {i}")
            if i in skip_connection_index:
                norm_idx = len(skip_connections)
                normalized_skip = self.skip_norms[norm_idx](x)
                skip_connections.append(normalized_skip)

        z3, z6, z9, z12_features = skip_connections

        # Reshape token sequences (B, num_patches, hidden_dim) back to a
        # spatial grid (B, hidden_dim, H, W). All four skip connections are
        # at identical spatial resolution here since the ViT encoder does
        # not spatially downsample across layers; the decoder below builds
        # the resolution pyramid itself via repeated DeconvBlock calls.
        batch, num_patches, hidden_dim = z12_features.shape
        patches_per_side = int(np.sqrt(num_patches))
        shape = (batch, hidden_dim, patches_per_side, patches_per_side)

        z0 = inputs
        z3 = z3.permute(0, 2, 1).contiguous().view(shape)
        z6 = z6.permute(0, 2, 1).contiguous().view(shape)
        z9 = z9.permute(0, 2, 1).contiguous().view(shape)
        z12_reshaped = z12_features.permute(0, 2, 1).contiguous().view(shape)

        # ASPP on the deepest feature map
        aspp_out = self.aspp(z12_reshaped)

        # Decoder: four stages, each upsampling the main path and the
        # corresponding skip branch to the same resolution before fusing
        # through an attention gate.
        x_seg = self.seg_d1(aspp_out)
        s = self.seg_s1(z9)
        s = self.att_gate_1(gate=x_seg, x=s)
        x_seg = torch.cat([x_seg, s], dim=1)
        x_seg = self.seg_c1(x_seg)

        x_seg = self.seg_d2(x_seg)
        s = self.seg_s2(z6)
        s = self.att_gate_2(gate=x_seg, x=s)
        x_seg = torch.cat([x_seg, s], dim=1)
        x_seg = self.seg_c2(x_seg)

        x_seg = self.seg_d3(x_seg)
        s = self.seg_s3(z3)
        s = self.att_gate_3(gate=x_seg, x=s)
        x_seg = torch.cat([x_seg, s], dim=1)
        x_seg = self.seg_c3(x_seg)

        x_seg = self.seg_d4(x_seg)
        s = self.seg_s4(z0)
        s = self.att_gate_4(gate=x_seg, x=s)
        x_seg = torch.cat([x_seg, s], dim=1)
        x_seg = self.seg_c4(x_seg)

        seg_output = self.seg_output(x_seg)

        if torch.isnan(seg_output).any():
            raise ValueError("NaN detected in model output")

        return seg_output
