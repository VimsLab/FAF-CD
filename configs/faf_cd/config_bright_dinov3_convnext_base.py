import os
import os.path as osp
import sys
import time

import numpy as np
from dotenv import load_dotenv
from easydict import EasyDict as edict

_repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
load_dotenv(dotenv_path=osp.join(_repo_root, '.env'))

C = edict()
config = C
cfg = C

C.seed = 3407
C.root_dir = _repo_root
C.abs_dir = _repo_root

# Dataset config
C.dataset_name = 'BRIGHT'
C.root_folder = osp.join(os.environ.get('DATASETS_ROOT', './datasets'), 'BRIGHT-1024')
C.A_format = '_pre_disaster.tif'
C.B_format = '_post_disaster.tif'
C.gt_format = '.png'
C.A_dir = 'pre-event'
C.B_dir = 'post-event'
C.gt_dir = 'gt'
C.B_grayscale = True
C.gt_is_binary = False
C.is_test = False
C.num_train_imgs = 2332
C.num_eval_imgs = 697
C.num_classes = 4
C.class_names = ['background', 'intact', 'damaged', 'destroyed']
C.auto_test_enable = True

# Image config
C.background = 255
C.image_height = 1024
C.image_width = 1024
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])
C.norm_mean_B = np.array([0.5])
C.norm_std_B = np.array([0.5])
C.pseudo_siamese = True
C.replicate_modal_x_channels = True

# Model config
C.backbone = 'dinov3'
C.pretrained_model = None
C.decoder = 'MambaDecoder'
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'
C.log_backend = 'tensorboard'

# Loss config: FAF-CD uses weighted CrossEntropy with MambaDecoder.
C.use_dice = False
C.dice_weight = 1
C.use_CrossE = True
C.CrossE_weight = 1
C.CrossE_class_weights = [0.1, 1.0, 2.0, 3.0]

# DINOv3 options
C.dinov3_repo_dir = os.environ.get('DINOV3_REPO', './dinov3')
C.dinov3_model_name = 'dinov3_convnext_base'
C.dinov3_pretrained = os.path.join('pretrained', 'DINOv3', 'dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth')
C.dinov3_out_indices = None
C.dinov3_use_frm = True
C.dinov3_use_deform_attn = True
C.dinov3_deform_n_heads = 8
C.dinov3_deform_n_points = 6
C.dinov3_freeze_epochs = 5
C.dinov3_backbone_lr_mult = 0.1
C.dinov3_fusion = 'gated_ffm_fft_dwt'
C.dinov3_fusion_gate_hidden_ratio = 0.25
C.dinov3_fusion_gate_temperature = 1.0

# Train config
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 8
C.nepochs = 150
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 16
C.train_scale_array = None
C.warm_up_epoch = 10
C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

# Eval config
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]
C.eval_flip = False
C.eval_crop_size = [1024, 1024]
C.ap_num_bins = 512

# Store config
C.checkpoint_start_epoch = 5
C.checkpoint_step = 5


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path(C.root_dir)

backbone_str = C.dinov3_model_name
filename = os.path.basename(C.dinov3_pretrained)
if 'pretrain_' in filename:
    start = filename.find('pretrain_') + len('pretrain_')
    end = filename.find('-', start)
    if end != -1:
        backbone_str = backbone_str + '_' + filename[start:end]

C.log_dir = osp.join(C.root_dir, 'logs', C.dataset_name, backbone_str + '_' + C.decoder)
C.tb_dir = osp.join(C.log_dir, 'tb')
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.join(C.log_dir, 'checkpoint')

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = osp.join(C.log_dir, 'log_' + exp_time + '.log')
C.link_log_file = osp.join(C.log_dir, 'log_last.log')
C.val_log_file = osp.join(C.log_dir, 'val_' + exp_time + '.log')
C.link_val_log_file = osp.join(C.log_dir, 'val_last.log')
C.test_log_file = osp.join(C.log_dir, 'test_' + exp_time + '.log')
C.link_test_log_file = osp.join(C.log_dir, 'test_last.log')

if __name__ == '__main__':
    print(config.nepochs)
