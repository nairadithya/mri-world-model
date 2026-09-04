"""BRAINIAC vision backbone: MONAI ViT-B/16 + SimCLR weights + LoRA.

Input:  (B, 1, 96, 96, 96) single-sequence MRI.
Output: (B, 768) latent (first patch token, per BRAINIAC's own forward).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn


def build_vit() -> nn.Module:
    from monai.networks.nets import ViT

    return ViT(
        in_channels=1,
        img_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        hidden_size=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        save_attn=False,
    )


def load_simclr_weights(vit: nn.Module, ckpt_path: str, strict: bool = False) -> None:
    """Load BRAINIAC SimCLR checkpoint (Lightning state_dict or safetensors)."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"BRAINIAC checkpoint not found: {ckpt_path}\n"
            "Download from https://www.dropbox.com/scl/fo/i51xt63roognvt7vuslbl/AG99uZljziHss5zJz4HiFis?rlkey=9w55le6tslwxlfz6c0viylmjb&st=b9cnvwh8&dl=0\n"
            "(see https://github.com/AIM-KannLab/BrainIAC) and place it at the "
            "configured path, or pass ckpt_path=None for random init (dev only)."
        )
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(ckpt_path)
    else:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
    # strip "backbone." prefix used by BRAINIAC's Lightning module
    stripped = {}
    for k, v in state.items():
        nk = k[9:] if k.startswith("backbone.") else k
        stripped[nk] = v
    own = set(vit.state_dict().keys())
    got = set(stripped.keys())
    missing, unexpected = own - got, got - own
    print(f"BRAINIAC weights: {len(own & got)}/{len(own)} keys loaded "
          f"({len(missing)} stay random: {sorted(missing)[:3]}{'...' if len(missing) > 3 else ''})")
    if unexpected:
        print(f"  ({len(unexpected)} checkpoint keys unused)")
    vit.load_state_dict(stripped, strict=strict)


def apply_lora(
    vit: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
) -> nn.Module:
    """Wrap attention/MLP linears with LoRA via peft.

    MONAI's SABlock uses `qkv`/`proj`, MLPBlock uses `linear1`/`linear2`
    (names vary by MONAI version, so we match several suffixes; peft applies
    to every Linear whose name ends with one of these).
    """
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["qkv", "proj", "linear1", "linear2", "fc1", "fc2",
                        "q_proj", "k_proj", "v_proj", "out_proj"],
    )
    return get_peft_model(vit, cfg)


class BrainiacEncoder(nn.Module):
    """Frozen ViT + trainable LoRA adapters. encode(x) -> (B, 768)."""

    def __init__(
        self,
        ckpt_path: str | None = None,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        vit = build_vit()
        if ckpt_path is not None:
            load_simclr_weights(vit, ckpt_path)
        if freeze_backbone:
            for p in vit.parameters():
                p.requires_grad = False
        self.vit = apply_lora(vit, lora_rank, lora_alpha, lora_dropout)

    def _pool(self, out) -> torch.Tensor:
        h = out[0] if isinstance(out, (tuple, list)) else out
        if h.dim() == 3:  # (B, tokens, 768) -> first patch token
            return h[:, 0]
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._pool(self.vit(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
