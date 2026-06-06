import inspect
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.logger import get_logger
from .peft_utils import apply_encoder_peft, build_convnext_feature_adapters
from ..net_utils import FeatureFusionModule as FFM
from ..net_utils import FeatureRectifyModule as FRM
from ..net_utils import FFTFusionModule, DwtFusionModule

logger = get_logger()


def _strip_prefix(state_dict, prefixes: Sequence[str]) -> dict:
    if not state_dict:
        return state_dict
    for prefix in prefixes:
        if all(k.startswith(prefix) for k in state_dict.keys()):
            return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _load_dinov3_model(repo_dir: str, model_name: str) -> nn.Module:
    """
    Load DINOv3 model from local hub without downloading from internet.
    Creates the model architecture without pretrained weights (weights are loaded separately).
    """
    try:
        # Map model names to their hub factory functions
        import sys
        sys.path.insert(0, repo_dir)
        from dinov3.hub import backbones
        
        # Get the factory function for the model
        if not hasattr(backbones, model_name):
            raise AttributeError(f"Model factory '{model_name}' not found in dinov3.hub.backbones")
        
        # Call the factory function with pretrained=False to create model without downloading
        model_factory = getattr(backbones, model_name)
        model = model_factory(pretrained=False)
        
        return model
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load DINOv3 model '{model_name}' from '{repo_dir}'. "
            "Ensure the DINOv3 repo is cloned locally and the model name matches a factory in that repo."
        ) from exc


def _get_attr_first(obj, names: Sequence[str], default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _normalize_patch_size(patch_size) -> Tuple[int, int]:
    if isinstance(patch_size, (list, tuple)):
        return int(patch_size[0]), int(patch_size[1])
    return int(patch_size), int(patch_size)


class DualDINOv3(nn.Module):
    def __init__(
        self,
        model_name: str,
        repo_dir: str,
        pretrained_path: Optional[str] = None,
        out_indices: Optional[Sequence[int]] = None,
        norm_fuse: nn.Module = nn.BatchNorm2d,
        use_frm: bool = True,
        use_adapter: bool = False,
        use_deform_attn: bool = False,
        deform_n_heads: int = 8,
        deform_n_points: int = 4,
        fusion_mode: str = "ffm",
        mamba_d_state: int = 4,
        fusion_gate_hidden_ratio: float = 0.25,
        fusion_gate_temperature: float = 1.0,
        ensemble_branch_drop_prob: float = 0.0,
        ensemble_use_residual_proj: bool = True,
        freeze_backbone: bool = True,
        peft_strategy: str = "full",
        peft_lora_r: int = 8,
        peft_lora_alpha: float = 16.0,
        peft_lora_dropout: float = 0.0,
        peft_lora_target_modules: Optional[Sequence[str]] = None,
        peft_vpt_num_tokens: int = 8,
        peft_vpt_init_std: float = 0.02,
        peft_trainable_stages: int = 1,
        peft_adapter_reduction: int = 16,
        peft_adapter_shared: bool = False,
        pseudo_siamese: bool = False,
    ):
        super().__init__()

        # Normalize early so every branch below sees the same canonical form
        # that apply_encoder_peft will use internally.
        peft_strategy = str(peft_strategy or "full").strip().lower()

        self.pseudo_siamese = bool(pseudo_siamese)
        self.model = _load_dinov3_model(repo_dir, model_name)
        if self.pseudo_siamese:
            self.model_post = _load_dinov3_model(repo_dir, model_name)

        self.embed_dim = _get_attr_first(self.model, ["embed_dim", "dim", "hidden_dim"], None)
        if self.embed_dim is None:
            raise ValueError("Could not infer embed_dim from DINOv3 model.")

        patch_size = _get_attr_first(self.model, ["patch_size"], None)
        if patch_size is None and hasattr(self.model, "patch_embed"):
            patch_size = _get_attr_first(self.model.patch_embed, ["patch_size"], None)
        self.patch_size = _normalize_patch_size(patch_size or 16)

        self.blocks = _get_attr_first(self.model, ["blocks", "layers"], None)
        if self.blocks is not None:
            self.num_blocks = len(self.blocks)
        else:
            self.num_blocks = _get_attr_first(self.model, ["n_blocks", "num_blocks", "depth"], None)
            if self.num_blocks is None and hasattr(self.model, "stages"):
                self.num_blocks = len(self.model.stages)
            if self.num_blocks is None and hasattr(self.model, "downsample_layers"):
                self.num_blocks = len(self.model.downsample_layers)
            if self.num_blocks is None and not hasattr(self.model, "get_intermediate_layers"):
                raise ValueError("Could not find transformer blocks on the DINOv3 model.")

        if out_indices is None:
            if self.num_blocks <= 4:
                out_indices = list(range(self.num_blocks))
            else:
                quarter = max(1, self.num_blocks // 4)
                out_indices = [quarter - 1, 2 * quarter - 1, 3 * quarter - 1, self.num_blocks - 1]
        self.out_indices = list(out_indices)

        self.use_frm = use_frm
        self.fusion_mode = str(fusion_mode).lower()
        if self.fusion_mode not in (
            "ffm",
            "mamba",
            "fft",
            "dwt",
            "ffm_fft",
            "ffm_dwt",
            "gated_ffm_fft_dwt",
            "gated_mamba_fft_dwt",
            "all_ensemble",
        ):
            raise ValueError(
                "Unsupported fusion_mode='{}'. Expected one of: 'ffm', 'mamba', 'fft', 'dwt', "
                "'ffm_fft', 'ffm_dwt', 'gated_ffm_fft_dwt', 'gated_mamba_fft_dwt', 'all_ensemble'.".format(fusion_mode)
            )
        self.fusion_gate_hidden_ratio = float(fusion_gate_hidden_ratio)
        self.fusion_gate_temperature = max(float(fusion_gate_temperature), 1e-6)
        self.ensemble_branch_drop_prob = float(ensemble_branch_drop_prob)
        self.ensemble_use_residual_proj = bool(ensemble_use_residual_proj)

        self.is_convnext = hasattr(self.model, "stages") or hasattr(self.model, "downsample_layers")
        self.use_adapter = bool(use_adapter) and (not self.is_convnext)
        stage_dims = None
        if hasattr(self.model, "embed_dims"):
            stage_dims = list(self.model.embed_dims)
        elif hasattr(self.model, "dims"):
            stage_dims = list(self.model.dims)

        if self.use_adapter:
            if len(self.out_indices) != 4:
                raise ValueError("DINOv3 adapter expects 4 interaction indexes for adapter-style decoding.")
            self.stage_dims = [self.embed_dim] * 4
        else:
            if stage_dims is not None and len(stage_dims) > 0:
                self.stage_dims = [stage_dims[i] for i in self.out_indices]
            else:
                self.stage_dims = [self.embed_dim] * len(self.out_indices)

        if self.fusion_mode == "ffm":
            self.FRMs = nn.ModuleList([FRM(dim=dim, reduction=1) for dim in self.stage_dims])
            self.FFMs = nn.ModuleList(
                [
                    FFM(
                        dim=dim,
                        reduction=1,
                        norm_layer=norm_fuse,
                        use_deform_attn=use_deform_attn,
                        deform_n_heads=deform_n_heads,
                        deform_n_points=deform_n_points,
                    )
                    for dim in self.stage_dims
                ]
            )
        elif self.fusion_mode == "ffm_fft":
            self.FRMs = nn.ModuleList([FRM(dim=dim, reduction=1) for dim in self.stage_dims])
            self.FFMs = nn.ModuleList(
                [
                    FFM(
                        dim=dim,
                        reduction=1,
                        norm_layer=norm_fuse,
                        use_deform_attn=use_deform_attn,
                        deform_n_heads=deform_n_heads,
                        deform_n_points=deform_n_points,
                    )
                    for dim in self.stage_dims
                ]
            )
            self.fft_fusions = nn.ModuleList([FFTFusionModule(dim=dim, norm_layer=norm_fuse) for dim in self.stage_dims])
            self.hybrid_projs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
                        norm_fuse(dim),
                        nn.ReLU(inplace=True),
                    )
                    for dim in self.stage_dims
                ]
            )
        elif self.fusion_mode == "ffm_dwt":
            self.FRMs = nn.ModuleList([FRM(dim=dim, reduction=1) for dim in self.stage_dims])
            self.FFMs = nn.ModuleList(
                [
                    FFM(
                        dim=dim,
                        reduction=1,
                        norm_layer=norm_fuse,
                        use_deform_attn=use_deform_attn,
                        deform_n_heads=deform_n_heads,
                        deform_n_points=deform_n_points,
                    )
                    for dim in self.stage_dims
                ]
            )
            self.dwt_fusions = nn.ModuleList([DwtFusionModule(dim=dim, norm_layer=norm_fuse) for dim in self.stage_dims])
            self.hybrid_projs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
                        norm_fuse(dim),
                        nn.ReLU(inplace=True),
                    )
                    for dim in self.stage_dims
                ]
            )
        elif self.fusion_mode == "gated_ffm_fft_dwt":
            self.FRMs = nn.ModuleList([FRM(dim=dim, reduction=1) for dim in self.stage_dims])
            self.FFMs = nn.ModuleList(
                [
                    FFM(
                        dim=dim,
                        reduction=1,
                        norm_layer=norm_fuse,
                        use_deform_attn=use_deform_attn,
                        deform_n_heads=deform_n_heads,
                        deform_n_points=deform_n_points,
                    )
                    for dim in self.stage_dims
                ]
            )
            self.fft_fusions = nn.ModuleList([FFTFusionModule(dim=dim, norm_layer=norm_fuse) for dim in self.stage_dims])
            self.dwt_fusions = nn.ModuleList([DwtFusionModule(dim=dim, norm_layer=norm_fuse) for dim in self.stage_dims])
            self.fusion_gates = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Conv2d(dim * 3, max(16, int(dim * self.fusion_gate_hidden_ratio)), kernel_size=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(max(16, int(dim * self.fusion_gate_hidden_ratio)), 3, kernel_size=1, bias=True),
                    )
                    for dim in self.stage_dims
                ]
            )
        elif self.fusion_mode == "gated_mamba_fft_dwt":
            from .vmamba import ConcatMambaFusionBlock
            self.FRMs = nn.ModuleList([FRM(dim=dim, reduction=1) for dim in self.stage_dims])
            self.fft_fusions = nn.ModuleList([FFTFusionModule(dim=dim, norm_layer=norm_fuse) for dim in self.stage_dims])
            self.dwt_fusions = nn.ModuleList([DwtFusionModule(dim=dim, norm_layer=norm_fuse) for dim in self.stage_dims])
            self.mamba_fusions = nn.ModuleList(
                [
                    ConcatMambaFusionBlock(
                        hidden_dim=dim,
                        mlp_ratio=0.0,
                        d_state=mamba_d_state,
                    )
                    for dim in self.stage_dims
                ]
            )
            self.fusion_gates = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Conv2d(dim * 3, max(16, int(dim * self.fusion_gate_hidden_ratio)), kernel_size=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(max(16, int(dim * self.fusion_gate_hidden_ratio)), 3, kernel_size=1, bias=True),
                    )
                    for dim in self.stage_dims
                ]
            )
            self.ensemble_residual_projs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False),
                        norm_fuse(dim),
                    )
                    for dim in self.stage_dims
                ]
            )
        elif self.fusion_mode == "all_ensemble":
            from .vmamba import ConcatMambaFusionBlock
            self.FRMs = nn.ModuleList([FRM(dim=dim, reduction=1) for dim in self.stage_dims])
            self.FFMs = nn.ModuleList(
                [
                    FFM(
                        dim=dim,
                        reduction=1,
                        norm_layer=norm_fuse,
                        use_deform_attn=use_deform_attn,
                        deform_n_heads=deform_n_heads,
                        deform_n_points=deform_n_points,
                    )
                    for dim in self.stage_dims
                ]
            )
            self.fft_fusions = nn.ModuleList([FFTFusionModule(dim=dim, norm_layer=norm_fuse) for dim in self.stage_dims])
            self.dwt_fusions = nn.ModuleList([DwtFusionModule(dim=dim, norm_layer=norm_fuse) for dim in self.stage_dims])
            self.mamba_fusions = nn.ModuleList(
                [
                    ConcatMambaFusionBlock(
                        hidden_dim=dim,
                        mlp_ratio=0.0,
                        d_state=mamba_d_state,
                    )
                    for dim in self.stage_dims
                ]
            )
            self.fusion_gates = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Conv2d(dim * 4, max(16, int(dim * self.fusion_gate_hidden_ratio)), kernel_size=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(max(16, int(dim * self.fusion_gate_hidden_ratio)), 4, kernel_size=1, bias=True),
                    )
                    for dim in self.stage_dims
                ]
            )
            self.ensemble_residual_projs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(dim * 4, dim, kernel_size=1, bias=False),
                        norm_fuse(dim),
                    )
                    for dim in self.stage_dims
                ]
            )
        elif self.fusion_mode == "fft":
            self.FRMs = nn.ModuleList([FRM(dim=dim, reduction=1) for dim in self.stage_dims])
            self.freq_fusions = nn.ModuleList([FFTFusionModule(dim=dim, norm_layer=norm_fuse) for dim in self.stage_dims])
        elif self.fusion_mode == "dwt":
            self.FRMs = nn.ModuleList([FRM(dim=dim, reduction=1) for dim in self.stage_dims])
            self.freq_fusions = nn.ModuleList([DwtFusionModule(dim=dim, norm_layer=norm_fuse) for dim in self.stage_dims])
        else:
            from .vmamba import ConcatMambaFusionBlock
            self.mamba_fusions = nn.ModuleList(
                [
                    ConcatMambaFusionBlock(
                        hidden_dim=dim,
                        mlp_ratio=0.0,
                        d_state=mamba_d_state,
                    )
                    for dim in self.stage_dims
                ]
            )

        self.out_channels = list(self.stage_dims)

        if self.use_adapter:
            from dinov3.eval.segmentation.models.backbone.dinov3_adapter import DINOv3_Adapter
            self.adapter = DINOv3_Adapter(
                self.model,
                interaction_indexes=self.out_indices,
                freeze_backbone=freeze_backbone,
            )
            if self.pseudo_siamese:
                self.adapter_post = DINOv3_Adapter(
                    self.model_post,
                    interaction_indexes=self.out_indices,
                    freeze_backbone=freeze_backbone,
                )

        if pretrained_path:
            self._load_pretrained(pretrained_path)

        if peft_strategy == "partial_last_stage" and not self.is_convnext:
            raise ValueError(
                "partial_last_stage PEFT strategy requires a ConvNeXt-like DINOv3 encoder "
                "with `stages` / `downsample_layers`. Current model does not qualify."
            )

        if peft_strategy == "convnext_adapter" and not self.is_convnext:
            raise ValueError(
                "convnext_adapter PEFT strategy requires a ConvNeXt-like DINOv3 encoder "
                "with `stages` / `downsample_layers`. Current model does not qualify."
            )

        self.peft_summary = apply_encoder_peft(
            self.model,
            strategy=peft_strategy,
            lora_r=peft_lora_r,
            lora_alpha=peft_lora_alpha,
            lora_dropout=peft_lora_dropout,
            lora_target_modules=peft_lora_target_modules,
            vpt_num_tokens=peft_vpt_num_tokens,
            vpt_init_std=peft_vpt_init_std,
            trainable_stages=peft_trainable_stages,
        )
        logger.info(
            "Encoder PEFT strategy=%s lora_modules=%s vpt_tokens=%s",
            self.peft_summary.get("strategy"),
            self.peft_summary.get("lora_modules"),
            self.peft_summary.get("vpt_tokens"),
        )
        if self.peft_summary.get("strategy") == "partial_last_stage":
            logger.info(
                "partial_last_stage: trainable_stages=%d unfrozen_stages=%s "
                "unfrozen_downsample=%s trainable_params=%s/%s",
                self.peft_summary["trainable_stages"],
                self.peft_summary["unfrozen_stage_indices"],
                self.peft_summary["unfrozen_downsample_indices"],
                self.peft_summary["trainable_params"],
                self.peft_summary["total_params"],
            )
        if self.pseudo_siamese:
            self.peft_summary_post = apply_encoder_peft(
                self.model_post,
                strategy=peft_strategy,
                lora_r=peft_lora_r,
                lora_alpha=peft_lora_alpha,
                lora_dropout=peft_lora_dropout,
                lora_target_modules=peft_lora_target_modules,
                vpt_num_tokens=peft_vpt_num_tokens,
                vpt_init_std=peft_vpt_init_std,
                trainable_stages=peft_trainable_stages,
            )
            logger.info("Pseudo-siamese post-encoder PEFT applied.")

        self.stage_adapter_shared = bool(peft_adapter_shared)
        self.stage_adapter_reduction = int(peft_adapter_reduction)
        self.stage_adapters = None
        self.stage_adapters_post = None
        if peft_strategy == "convnext_adapter":
            self.stage_adapters = build_convnext_feature_adapters(
                self.stage_dims, reduction=self.stage_adapter_reduction
            )
            if not self.stage_adapter_shared:
                self.stage_adapters_post = build_convnext_feature_adapters(
                    self.stage_dims, reduction=self.stage_adapter_reduction
                )
            adapter_params = sum(p.numel() for p in self.stage_adapters.parameters())
            if self.stage_adapters_post is not None:
                adapter_params += sum(p.numel() for p in self.stage_adapters_post.parameters())
            logger.info(
                "ConvNeXt feature-adapter PEFT: reduction=%d shared=%s adapter_params=%d",
                self.stage_adapter_reduction,
                self.stage_adapter_shared,
                adapter_params,
            )

    def _apply_branch_dropout(self, weights: torch.Tensor) -> torch.Tensor:
        if not self.training or self.ensemble_branch_drop_prob <= 0.0:
            return weights
        B, K, _, _ = weights.shape
        if K <= 1:
            return weights
        drop_samples = (torch.rand(B, device=weights.device) < self.ensemble_branch_drop_prob)
        if not drop_samples.any():
            return weights
        drop_idx = torch.randint(low=0, high=K, size=(B,), device=weights.device)
        keep_mask = torch.ones_like(weights)
        keep_mask[drop_samples, drop_idx[drop_samples], :, :] = 0.0
        weights = weights * keep_mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return weights

    def _fuse_stage(self, i: int, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode == "ffm":
            if self.use_frm:
                fa, fb = self.FRMs[i](fa, fb)
            return self.FFMs[i](fa, fb)
        if self.fusion_mode in ("fft", "dwt"):
            if self.use_frm:
                fa, fb = self.FRMs[i](fa, fb)
            return self.freq_fusions[i](fa, fb)
        if self.fusion_mode == "ffm_fft":
            if self.use_frm:
                fa, fb = self.FRMs[i](fa, fb)
            out_ffm = self.FFMs[i](fa, fb)
            out_fft = self.fft_fusions[i](fa, fb)
            return self.hybrid_projs[i](torch.cat([out_ffm, out_fft], dim=1))
        if self.fusion_mode == "ffm_dwt":
            if self.use_frm:
                fa, fb = self.FRMs[i](fa, fb)
            out_ffm = self.FFMs[i](fa, fb)
            out_dwt = self.dwt_fusions[i](fa, fb)
            return self.hybrid_projs[i](torch.cat([out_ffm, out_dwt], dim=1))
        if self.fusion_mode == "gated_ffm_fft_dwt":
            if self.use_frm:
                fa, fb = self.FRMs[i](fa, fb)
            out_ffm = self.FFMs[i](fa, fb)
            out_fft = self.fft_fusions[i](fa, fb)
            out_dwt = self.dwt_fusions[i](fa, fb)
            gate_in = torch.cat([out_ffm, out_fft, out_dwt], dim=1)
            gate_logits = self.fusion_gates[i](gate_in) / self.fusion_gate_temperature
            gate = torch.softmax(gate_logits, dim=1)
            return gate[:, 0:1] * out_ffm + gate[:, 1:2] * out_fft + gate[:, 2:3] * out_dwt
        if self.fusion_mode == "gated_mamba_fft_dwt":
            if self.use_frm:
                fa, fb = self.FRMs[i](fa, fb)
            out_fft = self.fft_fusions[i](fa, fb)
            out_dwt = self.dwt_fusions[i](fa, fb)
            out_mamba = self.mamba_fusions[i](
                fa.permute(0, 2, 3, 1).contiguous(),
                fb.permute(0, 2, 3, 1).contiguous(),
            ).permute(0, 3, 1, 2).contiguous()
            gate_in = torch.cat([out_mamba, out_fft, out_dwt], dim=1)
            gate_logits = self.fusion_gates[i](gate_in) / self.fusion_gate_temperature
            gate = torch.softmax(gate_logits, dim=1)
            gate = self._apply_branch_dropout(gate)
            fused = gate[:, 0:1] * out_mamba + gate[:, 1:2] * out_fft + gate[:, 2:3] * out_dwt
            if self.ensemble_use_residual_proj:
                fused = fused + self.ensemble_residual_projs[i](gate_in)
            return fused
        if self.fusion_mode == "all_ensemble":
            if self.use_frm:
                fa, fb = self.FRMs[i](fa, fb)
            out_ffm = self.FFMs[i](fa, fb)
            out_fft = self.fft_fusions[i](fa, fb)
            out_dwt = self.dwt_fusions[i](fa, fb)
            out_mamba = self.mamba_fusions[i](
                fa.permute(0, 2, 3, 1).contiguous(),
                fb.permute(0, 2, 3, 1).contiguous(),
            ).permute(0, 3, 1, 2).contiguous()
            gate_in = torch.cat([out_ffm, out_fft, out_dwt, out_mamba], dim=1)
            gate_logits = self.fusion_gates[i](gate_in) / self.fusion_gate_temperature
            gate = torch.softmax(gate_logits, dim=1)
            gate = self._apply_branch_dropout(gate)
            fused = (
                gate[:, 0:1] * out_ffm
                + gate[:, 1:2] * out_fft
                + gate[:, 2:3] * out_dwt
                + gate[:, 3:4] * out_mamba
            )
            if self.ensemble_use_residual_proj:
                fused = fused + self.ensemble_residual_projs[i](gate_in)
            return fused
        fused = self.mamba_fusions[i](
            fa.permute(0, 2, 3, 1).contiguous(),
            fb.permute(0, 2, 3, 1).contiguous(),
        )
        return fused.permute(0, 3, 1, 2).contiguous()

    def _load_pretrained(self, pretrained_path: str) -> None:
        logger.info(f"Loading DINOv3 weights from {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location="cpu")
        if isinstance(ckpt, dict):
            for key in ["model", "state_dict", "teacher", "student"]:
                if key in ckpt:
                    ckpt = ckpt[key]
                    break
        if isinstance(ckpt, dict):
            ckpt = _strip_prefix(ckpt, ["model.", "backbone.", "student.", "student.backbone."])
        missing, unexpected = self.model.load_state_dict(ckpt, strict=False)
        if missing:
            logger.info(f"Missing keys when loading DINOv3: {len(missing)}")
        if unexpected:
            logger.info(f"Unexpected keys when loading DINOv3: {len(unexpected)}")
        if self.pseudo_siamese:
            missing_post, unexpected_post = self.model_post.load_state_dict(ckpt, strict=False)
            if missing_post:
                logger.info(f"Missing keys when loading DINOv3 post-encoder: {len(missing_post)}")
            if unexpected_post:
                logger.info(f"Unexpected keys when loading DINOv3 post-encoder: {len(unexpected_post)}")

    def _reshape_tokens(self, tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # tokens: B, N, C
        if tokens.dim() == 4:
            return tokens
        if tokens.shape[1] == H * W + 1:
            tokens = tokens[:, 1:, :]
        return tokens.transpose(1, 2).reshape(tokens.shape[0], self.embed_dim, H, W).contiguous()

    def _get_intermediate_layers(self, x: torch.Tensor, model: Optional[nn.Module] = None) -> List[torch.Tensor]:
        if model is None:
            model = self.model

        # Prefer built-in helper if available
        if hasattr(model, "get_intermediate_layers"):
            getter = model.get_intermediate_layers
            sig = inspect.signature(getter)
            kwargs = {}
            if "reshape" in sig.parameters:
                kwargs["reshape"] = True
            if "return_class_token" in sig.parameters:
                kwargs["return_class_token"] = False
            try:
                return getter(x, n=self.out_indices, **kwargs)
            except Exception:
                try:
                    return getter(x, n=len(self.out_indices), **kwargs)
                except Exception:
                    return getter(x)

        # Fallback: use hooks on transformer blocks
        blocks = _get_attr_first(model, ["blocks", "layers"], None)
        if blocks is None:
            raise ValueError("DINOv3 model does not expose blocks and get_intermediate_layers failed.")
        feats = {}
        hooks = []

        def _make_hook(idx):
            def _hook(_, __, output):
                feats[idx] = output
            return _hook

        for idx in self.out_indices:
            hooks.append(blocks[idx].register_forward_hook(_make_hook(idx)))

        if hasattr(model, "forward_features"):
            _ = model.forward_features(x)
        else:
            _ = model(x)

        for h in hooks:
            h.remove()

        return [feats[i] for i in self.out_indices]

    def forward(self, x: torch.Tensor, x_d: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if self.use_adapter:
            feats_a = self.adapter(x)
            if self.pseudo_siamese:
                feats_b = self.adapter_post(x_d)
            else:
                feats_b = self.adapter(x_d)
            outs: List[torch.Tensor] = []
            for i, key in enumerate(["1", "2", "3", "4"]):
                fa, fb = feats_a[key], feats_b[key]
                outs.append(self._fuse_stage(i, fa, fb))
            return tuple(outs)

        H = x.shape[2] // self.patch_size[0]
        W = x.shape[3] // self.patch_size[1]

        if self.pseudo_siamese:
            feats_a = self._get_intermediate_layers(x, model=self.model)
            feats_b = self._get_intermediate_layers(x_d, model=self.model_post)
        else:
            feats_a = self._get_intermediate_layers(x)
            feats_b = self._get_intermediate_layers(x_d)

        outs: List[torch.Tensor] = []
        adapters_b = (
            self.stage_adapters_post if self.stage_adapters_post is not None else self.stage_adapters
        )
        for i, (fa, fb) in enumerate(zip(feats_a, feats_b)):
            fa = self._reshape_tokens(fa, H, W)
            fb = self._reshape_tokens(fb, H, W)
            if self.stage_adapters is not None:
                fa = self.stage_adapters[i](fa)
                fb = adapters_b[i](fb)
            outs.append(self._fuse_stage(i, fa, fb))

        return tuple(outs)
