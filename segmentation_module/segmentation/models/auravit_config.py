"""
segmentation/models/auravit_config.py

Config factory for LightweightAuraViT variants. Two fixes applied relative
to hand-writing the config dict directly, both documented in
segmentation_extension_plan.md:

1. num_patches is now always derived from image_size and patch_size rather
   than hand-computed and copy-pasted into the config dict. The original
   forward pass silently depends on num_patches matching the actual
   unfolded patch count exactly (patches.view(..., self.cf["num_patches"],
   ...) will throw a RuntimeError if they disagree), so deriving it removes
   an entire class of copy-paste config bugs.

2. A single dropout_rate now drives the encoder, decoder, and ASPP unless
   explicitly overridden. The original source had three independently
   configured dropout rates that could silently diverge: cf["dropout_rate"]
   (encoder/patch_embed/pos_dropout), cf.get("block_dropout_rate", 0.1)
   (decoder ResBlocks, defaulting to 0.1 regardless of dropout_rate since
   "block_dropout_rate" was never actually present in the example config),
   and a hardcoded 0.1 inside LightweightASPP.output_conv not exposed via
   config at all. build_config() below sets block_dropout_rate and
   aspp_dropout_rate equal to dropout_rate by default, while still allowing
   an explicit override of either if that is ever wanted as a deliberate
   ablation (for example, testing a stronger decoder-only MC-Dropout signal).
"""

from typing import Optional


def build_config(
    image_size: int = 256,
    num_layers: int = 8,
    hidden_dim: int = 512,
    mlp_dim: int = 2048,
    num_heads: int = 8,
    dropout_rate: float = 0.1,
    patch_size: int = 16,
    num_channels: int = 1,
    block_dropout_rate: Optional[float] = None,
    aspp_dropout_rate: Optional[float] = None,
    aspp_rates: tuple = (6, 12, 18),
) -> dict:
    if image_size % patch_size != 0:
        raise ValueError(
            f"image_size ({image_size}) must be divisible by patch_size "
            f"({patch_size}); got remainder {image_size % patch_size}."
        )
    if num_layers not in (4, 6, 8, 12):
        raise ValueError(
            f"num_layers must be one of (4, 6, 8, 12) because the skip "
            f"connection index lookup in LightweightAuraViT.forward is "
            f"hardcoded for exactly these four depths; got {num_layers}. "
            f"Extend the skip_connection_index lookup first if a different "
            f"depth is required."
        )
    if hidden_dim % num_heads != 0:
        raise ValueError(
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads "
            f"({num_heads}) for nn.TransformerEncoderLayer; got remainder "
            f"{hidden_dim % num_heads}."
        )

    patches_per_side = image_size // patch_size
    num_patches = patches_per_side ** 2

    return {
        "image_size": image_size,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "mlp_dim": mlp_dim,
        "num_heads": num_heads,
        "dropout_rate": dropout_rate,
        "patch_size": patch_size,
        "num_channels": num_channels,
        "num_patches": num_patches,
        "block_dropout_rate": block_dropout_rate if block_dropout_rate is not None else dropout_rate,
        "aspp_dropout_rate": aspp_dropout_rate if aspp_dropout_rate is not None else dropout_rate,
        "aspp_rates": list(aspp_rates),
    }


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

# Base config exactly as provided for this project: image_size 256,
# num_channels 1 (grayscale MMOTU ultrasound, see segmentation_extension_plan.md
# Section 1, point 1). ~30M parameters, matching the "lightweight" name in
# the original repo despite still being fairly large; encoder alone is
# roughly 84% of that budget (see plan Section 2).
LAURA_BASE = build_config(
    image_size=256, num_layers=8, hidden_dim=512, mlp_dim=2048,
    num_heads=8, dropout_rate=0.1, patch_size=16, num_channels=1,
)

# First lightweighting tier: hidden_dim 512 -> 320, num_layers 8 -> 6.
# Targets the encoder directly, which is where ~84% of LAURA_BASE's
# parameters live (segmentation_extension_plan.md Section 2, options A+B).
LAURA_SMALL = build_config(
    image_size=256, num_layers=6, hidden_dim=320, mlp_dim=1280,
    num_heads=8, dropout_rate=0.1, patch_size=16, num_channels=1,
)

# Second lightweighting tier: further width/depth reduction. num_heads
# dropped to 4 to keep hidden_dim divisible by num_heads at this width.
LAURA_TINY = build_config(
    image_size=256, num_layers=4, hidden_dim=256, mlp_dim=1024,
    num_heads=4, dropout_rate=0.1, patch_size=16, num_channels=1,
)

PRESETS = {
    "laura_base": LAURA_BASE,
    "laura_small": LAURA_SMALL,
    "laura_tiny": LAURA_TINY,
}


def get_preset(name: str) -> dict:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(PRESETS.keys())}")
    return dict(PRESETS[name])  # return a copy, callers should not mutate the shared preset
