from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


class ConvNeXtFeatureAdapter(nn.Module):
    """Bottleneck adapter applied to a 4D feature map: y = x + up(GELU(down(x)))."""

    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        dim = int(dim)
        reduction = max(int(reduction), 1)
        hidden = max(1, dim // reduction)
        self.down = nn.Conv2d(dim, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.up = nn.Conv2d(hidden, dim, kernel_size=1)
        # Zero-init final projection so the adapter starts as identity.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


def build_convnext_feature_adapters(stage_dims: Sequence[int], reduction: int = 16) -> nn.ModuleList:
    return nn.ModuleList(
        [ConvNeXtFeatureAdapter(int(d), reduction=int(reduction)) for d in stage_dims]
    )


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0.")

        self.base = base_linear
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.zeros(self.r, base_linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, self.r))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (self.dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return base_out + lora_out


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = bool(requires_grad)


def _name_matches(name: str, target_keywords: Sequence[str] | None) -> bool:
    if not target_keywords:
        return True
    lname = name.lower()
    return any(keyword.lower() in lname for keyword in target_keywords)


def _inject_lora_recursively(
    module: nn.Module,
    target_keywords: Sequence[str] | None,
    r: int,
    alpha: float,
    dropout: float,
    prefix: str = "",
) -> int:
    replaced = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear) and _name_matches(full_name, target_keywords):
            setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            replaced += 1
            continue
        replaced += _inject_lora_recursively(
            child,
            target_keywords=target_keywords,
            r=r,
            alpha=alpha,
            dropout=dropout,
            prefix=full_name,
        )
    return replaced


def _enable_vpt_storage_tokens(model: nn.Module, num_tokens: int, init_std: float = 0.02) -> bool:
    if num_tokens <= 0:
        return False
    if not hasattr(model, "embed_dim"):
        raise ValueError("VPT requires encoder model to expose embed_dim.")
    if not hasattr(model, "n_storage_tokens"):
        raise ValueError("VPT requires encoder model to support storage tokens (n_storage_tokens).")

    embed_dim = int(getattr(model, "embed_dim"))
    dtype = model.cls_token.dtype if hasattr(model, "cls_token") else torch.float32
    device = model.cls_token.device if hasattr(model, "cls_token") else None

    storage_tokens = nn.Parameter(torch.zeros(1, int(num_tokens), embed_dim, dtype=dtype, device=device))
    nn.init.normal_(storage_tokens, std=float(init_std))

    model.storage_tokens = storage_tokens
    model.n_storage_tokens = int(num_tokens)
    model.storage_tokens.requires_grad = True
    return True


def _apply_partial_last_stage(
    encoder_model: nn.Module,
    trainable_stages: int = 1,
) -> dict:
    """Freeze all encoder params, then unfreeze the last *trainable_stages* ConvNeXt stages."""
    if not hasattr(encoder_model, "stages"):
        raise ValueError(
            "partial_last_stage strategy requires a ConvNeXt-like encoder with a `stages` attribute."
        )

    stages = encoder_model.stages
    num_stages = len(stages)
    trainable_stages = int(trainable_stages)
    if trainable_stages < 1 or trainable_stages > num_stages:
        raise ValueError(
            f"trainable_stages={trainable_stages} out of range [1, {num_stages}]."
        )

    # Freeze everything first
    _set_requires_grad(encoder_model, False)

    # Unfreeze the last N stages
    unfrozen_stage_indices = list(range(num_stages - trainable_stages, num_stages))
    for idx in unfrozen_stage_indices:
        _set_requires_grad(stages[idx], True)

    # Also unfreeze corresponding late downsample_layers if they exist
    unfrozen_ds = []
    if hasattr(encoder_model, "downsample_layers"):
        ds_layers = encoder_model.downsample_layers
        # downsample_layers[i] feeds into stages[i]; unfreeze those matching unfrozen stages
        for idx in unfrozen_stage_indices:
            if idx < len(ds_layers):
                _set_requires_grad(ds_layers[idx], True)
                unfrozen_ds.append(idx)

    trainable = sum(p.numel() for p in encoder_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in encoder_model.parameters())

    return {
        "strategy": "partial_last_stage",
        "lora_modules": 0,
        "vpt_tokens": 0,
        "trainable_stages": trainable_stages,
        "unfrozen_stage_indices": unfrozen_stage_indices,
        "unfrozen_downsample_indices": unfrozen_ds,
        "trainable_params": trainable,
        "total_params": total,
    }


def apply_encoder_peft(
    encoder_model: nn.Module,
    strategy: str = "full",
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    lora_target_modules: Sequence[str] | None = None,
    vpt_num_tokens: int = 8,
    vpt_init_std: float = 0.02,
    trainable_stages: int = 1,
) -> dict:
    strategy = str(strategy or "full").strip().lower()
    if strategy == "full":
        _set_requires_grad(encoder_model, True)
        return {"strategy": "full", "lora_modules": 0, "vpt_tokens": 0}

    if strategy == "lora":
        _set_requires_grad(encoder_model, False)
        replaced = _inject_lora_recursively(
            encoder_model,
            target_keywords=lora_target_modules,
            r=int(lora_r),
            alpha=float(lora_alpha),
            dropout=float(lora_dropout),
        )
        if replaced <= 0:
            raise ValueError("LoRA strategy selected but no Linear layers matched target modules.")
        return {"strategy": "lora", "lora_modules": replaced, "vpt_tokens": 0}

    if strategy == "vpt":
        _set_requires_grad(encoder_model, False)
        enabled = _enable_vpt_storage_tokens(
            encoder_model,
            num_tokens=int(vpt_num_tokens),
            init_std=float(vpt_init_std),
        )
        if not enabled:
            raise ValueError("VPT strategy selected but no prompt tokens were enabled.")
        return {"strategy": "vpt", "lora_modules": 0, "vpt_tokens": int(vpt_num_tokens)}

    if strategy == "partial_last_stage":
        return _apply_partial_last_stage(encoder_model, trainable_stages=trainable_stages)

    if strategy == "convnext_adapter":
        # Adapter modules themselves live outside the encoder model; here we only
        # freeze all base encoder parameters. DualDINOv3 builds the adapters.
        _set_requires_grad(encoder_model, False)
        return {"strategy": "convnext_adapter", "lora_modules": 0, "vpt_tokens": 0}

    raise ValueError(
        f"Unsupported encoder PEFT strategy: {strategy}. "
        "Expected one of: full, lora, vpt, partial_last_stage, convnext_adapter."
    )
