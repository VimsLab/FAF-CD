import cv2
import torch
import numpy as np
from torch.utils import data
import random
from utils.transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize


def random_mirror(A, B, gt):
    if random.random() >= 0.5:
        A = cv2.flip(A, 1)
        B = cv2.flip(B, 1)
        gt = cv2.flip(gt, 1)

    return A, B, gt

def random_scale(A, B, gt, scales):
    scale = random.choice(scales)
    sh = int(A.shape[0] * scale)
    sw = int(A.shape[1] * scale)
    A = cv2.resize(A, (sw, sh), interpolation=cv2.INTER_LINEAR)
    B = cv2.resize(B, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)

    return A, B, gt, scale

class TrainPre(object):
    def __init__(self, norm_mean, norm_std, config, gt_is_binary=True,
                 norm_mean_B=None, norm_std_B=None):
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.norm_mean_B = norm_mean_B
        self.norm_std_B = norm_std_B
        self.config = config
        self.gt_is_binary = gt_is_binary

    def __call__(self, A, B, gt):
        base_crop_size = A.shape[:2]
        A, B, gt = random_mirror(A, B, gt)
        if self.config.train_scale_array is not None:
            A, B, gt, scale = random_scale(A, B, gt, self.config.train_scale_array)

        if B.ndim == 2:
            B = B[:, :, np.newaxis]

        A = normalize(A, self.norm_mean, self.norm_std)
        if B.shape[2] == 1:
            mean_B = self.norm_mean_B
            std_B = self.norm_std_B
            if mean_B is None or std_B is None:
                mean_B, std_B = np.array([0.5]), np.array([0.5])
            else:
                mean_B = np.array([np.asarray(mean_B).reshape(-1)[0]])
                std_B = np.array([np.asarray(std_B).reshape(-1)[0]])
            B = normalize(B, mean_B, std_B)
        else:
            mean_B = self.norm_mean_B if self.norm_mean_B is not None else self.norm_mean
            std_B = self.norm_std_B if self.norm_std_B is not None else self.norm_std
            B = normalize(B, mean_B, std_B)
        if self.gt_is_binary:
            gt = (gt>124)+0

        crop_size = getattr(self.config, "train_crop_size", None)
        if crop_size is None and self.config.train_scale_array is not None:
            crop_size = base_crop_size
        if crop_size is not None:
            if isinstance(crop_size, int):
                crop_size = (crop_size, crop_size)
            else:
                crop_size = tuple(crop_size)
                if len(crop_size) != 2:
                    raise ValueError(f"train_crop_size must be int or [h, w], got: {crop_size}")
            crop_pos = generate_random_crop_pos(A.shape[:2], crop_size)
            p_A, _ = random_crop_pad_to_shape(A, crop_pos, crop_size, 0)
            p_B, _ = random_crop_pad_to_shape(B, crop_pos, crop_size, 0)
            p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        else:
            p_A, p_B, p_gt = A, B, gt

        p_A = p_A.transpose(2, 0, 1)
        p_B = p_B.transpose(2, 0, 1)
        
        return p_A, p_B, p_gt

class ValPre(object):
    def __init__(self, gt_is_binary=True, perturbation=None):
        self.gt_is_binary = gt_is_binary
        self.perturbation = perturbation

    def set_sample_id(self, sample_id):
        if self.perturbation is not None and hasattr(self.perturbation, "set_sample_id"):
            self.perturbation.set_sample_id(sample_id)

    def __call__(self, A, B, gt):
        if self.perturbation is not None:
            A, B = self.perturbation(A, B)
        if self.gt_is_binary:
            gt = (gt > 124) + 0
        return A, B, gt

def get_train_loader(engine, dataset, config):
    data_setting = {
        'root': config.root_folder,
        'A_format': config.A_format,
        'B_format': config.B_format,
        'gt_format': config.gt_format,
        'class_names': config.class_names,
        'A_dir': getattr(config, 'A_dir', 'A'),
        'B_dir': getattr(config, 'B_dir', 'B'),
        'gt_dir': getattr(config, 'gt_dir', 'gt'),
        'B_grayscale': getattr(config, 'B_grayscale', False),
    }

    # print("-"*20)
    # print("config.norm_mean: ", config.norm_mean)
    # print("config.norm_std: ", config.norm_std)
    # print("-"*20)

    gt_is_binary = getattr(config, 'gt_is_binary', True)
    norm_mean_B = getattr(config, 'norm_mean_B', None)
    norm_std_B = getattr(config, 'norm_std_B', None)

    train_preprocess = TrainPre(config.norm_mean, config.norm_std, config,
                                gt_is_binary=gt_is_binary,
                                norm_mean_B=norm_mean_B,
                                norm_std_B=norm_std_B)

    # train_dataset = dataset(data_setting, "train", train_preprocess, config.batch_size * config.niters_per_epoch)
    train_dataset = dataset(data_setting, "train", train_preprocess)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler
