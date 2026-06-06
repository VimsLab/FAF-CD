import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.init_func import init_weight
from engine.logger import get_logger

logger = get_logger()


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        cfg=None,
        criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255),
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        if cfg is None:
            raise ValueError('A FAF-CD config object is required.')
        if cfg.backbone != 'dinov3':
            raise ValueError("FAF-CD release supports only cfg.backbone='dinov3'.")
        if cfg.decoder != 'MambaDecoder':
            raise ValueError("FAF-CD release supports only cfg.decoder='MambaDecoder'.")

        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        self.deep_supervision = False
        self.aux_head = None
        self.gate = nn.Identity()
        self.replicate_modal_x_channels = bool(getattr(cfg, 'replicate_modal_x_channels', False))

        logger.info('Using backbone: DINOv3')
        from .encoders.dual_dinov3 import DualDINOv3 as backbone

        self.backbone = backbone(
            model_name=getattr(cfg, 'dinov3_model_name', ''),
            repo_dir=getattr(cfg, 'dinov3_repo_dir', ''),
            pretrained_path=getattr(cfg, 'dinov3_pretrained', None),
            out_indices=getattr(cfg, 'dinov3_out_indices', None),
            norm_fuse=norm_layer,
            use_frm=getattr(cfg, 'dinov3_use_frm', True),
            use_adapter=bool(getattr(cfg, 'dinov3_use_adapter', False)),
            use_deform_attn=getattr(cfg, 'dinov3_use_deform_attn', False),
            deform_n_heads=getattr(cfg, 'dinov3_deform_n_heads', 8),
            deform_n_points=getattr(cfg, 'dinov3_deform_n_points', 4),
            fusion_mode=getattr(cfg, 'dinov3_fusion', 'ffm'),
            mamba_d_state=getattr(cfg, 'dinov3_mamba_d_state', 4),
            fusion_gate_hidden_ratio=getattr(cfg, 'dinov3_fusion_gate_hidden_ratio', 0.25),
            fusion_gate_temperature=getattr(cfg, 'dinov3_fusion_gate_temperature', 1.0),
            ensemble_branch_drop_prob=getattr(cfg, 'dinov3_ensemble_branch_drop_prob', 0.0),
            ensemble_use_residual_proj=getattr(cfg, 'dinov3_ensemble_use_residual_proj', True),
            freeze_backbone=getattr(cfg, 'dinov3_freeze_backbone', True),
            peft_strategy=getattr(cfg, 'dinov3_peft_strategy', 'full'),
            peft_lora_r=getattr(cfg, 'dinov3_peft_lora_r', 8),
            peft_lora_alpha=getattr(cfg, 'dinov3_peft_lora_alpha', 16.0),
            peft_lora_dropout=getattr(cfg, 'dinov3_peft_lora_dropout', 0.0),
            peft_lora_target_modules=getattr(cfg, 'dinov3_peft_lora_target_modules', None),
            peft_vpt_num_tokens=getattr(cfg, 'dinov3_peft_vpt_num_tokens', 8),
            peft_vpt_init_std=getattr(cfg, 'dinov3_peft_vpt_init_std', 0.02),
            peft_trainable_stages=getattr(cfg, 'dinov3_trainable_stages', 1),
            peft_adapter_reduction=getattr(cfg, 'dinov3_adapter_reduction', 16),
            peft_adapter_shared=getattr(cfg, 'dinov3_adapter_shared', False),
            pseudo_siamese=getattr(cfg, 'pseudo_siamese', False),
        )
        self.channels = self.backbone.out_channels

        logger.info('Using Mamba Decoder')
        from .decoders.MambaDecoder import MambaDecoder

        self.decode_head = MambaDecoder(
            img_size=[cfg.image_height, cfg.image_width],
            in_channels=self.channels,
            num_classes=cfg.num_classes,
            embed_dim=self.channels[0],
            deep_supervision=self.deep_supervision,
            use_checkpoint=bool(getattr(cfg, 'mamba_decoder_use_checkpoint', False)),
        )

        self.criterion = criterion
        if self.criterion:
            self.init_weights(cfg, pretrained=getattr(cfg, 'pretrained_model', None))

    def init_weights(self, cfg, pretrained=None):
        if pretrained and hasattr(self.backbone, 'init_weights'):
            logger.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        logger.info('Initing weights ...')
        init_weight(
            self.decode_head,
            nn.init.kaiming_normal_,
            self.norm_layer,
            cfg.bn_eps,
            cfg.bn_momentum,
            mode='fan_in',
            nonlinearity='relu',
        )
        init_weight(
            self.gate,
            nn.init.kaiming_normal_,
            self.norm_layer,
            cfg.bn_eps,
            cfg.bn_momentum,
            mode='fan_in',
            nonlinearity='relu',
        )

    def encode_decode(self, rgb, modal_x):
        orisize = rgb.shape
        x = self.backbone(rgb, modal_x)
        out = self.decode_head.forward(x)
        if isinstance(out, dict):
            return out
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
        return out

    def forward(self, rgb, modal_x, label=None):
        if self.replicate_modal_x_channels and modal_x.shape[1] == 1:
            modal_x = modal_x.repeat(1, 3, 1, 1)
        rgb = self.gate(rgb)
        modal_x = self.gate(modal_x)

        out = self.encode_decode(rgb, modal_x)
        if isinstance(out, dict):
            if label is not None:
                return self.criterion(out, label.long())
            if hasattr(self.decode_head, 'semantic_logits_from_raw'):
                out = self.decode_head.semantic_logits_from_raw(out, logits=True)
                out = F.interpolate(out, size=rgb.shape[2:], mode='bilinear', align_corners=False)
            return out

        if label is not None:
            return self.criterion(out, label.long())
        return out

    def flops(self, shape=(3, 256, 256)):
        from fvcore.nn import flop_count, parameter_count
        import copy

        supported_ops = {
            'aten::silu': None,
            'aten::neg': None,
            'aten::exp': None,
            'aten::flip': None,
            'prim::PythonOp.SelectiveScanMamba': selective_scan_flop_jit,
            'prim::PythonOp.SelectiveScanOflex': selective_scan_flop_jit,
            'prim::PythonOp.SelectiveScanCore': selective_scan_flop_jit,
            'prim::PythonOp.SelectiveScanNRow': selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda().eval()
        inputs = (
            torch.randn((1, *shape), device=next(model.parameters()).device),
            torch.randn((1, *shape), device=next(model.parameters()).device),
        )
        params = parameter_count(model)['']
        gflops, _ = flop_count(model=model, inputs=inputs, supported_ops=supported_ops)
        del model, inputs, params
        return sum(gflops.values()) * 1e9


def print_jit_input_names(inputs):
    print('input params: ', end=' ', flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=' ', flush=True)
    except Exception:
        pass
    print('', flush=True)


def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    assert not with_complex
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    return flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
