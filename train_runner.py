import argparse
import collections.abc
import json
import math
import glob
import os
import os.path as osp
import re
import shutil
import sys
import time
import traceback
from contextlib import nullcontext

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from dataloader.changeDataset import ChangeDataset
from dataloader.dataloader import ValPre, get_train_loader
from engine.engine import Engine
from engine.logger import get_logger
from eval import SegEvaluator
from eval import run_eval
from models.builder import EncoderDecoder as segmodel
from utils.config_utils import load_config_by_name
from utils.init_func import group_weight
from utils.lr_policy import WarmUpPolyLR
from utils.muon_optimizer import MuonWithAuxAdam
from utils.pyt_utils import all_reduce_tensor
from utils.wandb_tracking import collect_git_metadata, inject_git_metadata_to_wandb_run


logger = get_logger()
# Keep a legacy fallback for non-torchrun launches, but do not override
# launcher-provided rendezvous settings.
os.environ.setdefault("MASTER_PORT", "16005")


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _sanitize_path_component(value, fallback="unknown"):
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._")
    return text or fallback


def _backbone_name_for_run(config):
    backbone = str(getattr(config, "backbone", "model") or "model").strip()
    if backbone.lower() != "dinov3":
        return backbone

    parts = ["dinov3"]

    model_name = str(getattr(config, "dinov3_model_name", "") or "").strip().lower()
    if model_name:
        if model_name.startswith("dinov3_"):
            model_name = model_name[len("dinov3_") :]
        model_alias = {
            "convnext_tiny": "convnextt",
            "convnext_small": "convnexts",
            "convnext_base": "convnextb",
            "convnext_large": "convnextl",
        }
        model_token = model_alias.get(model_name, model_name.replace("_", ""))
        if model_token:
            parts.append(model_token)

    pretrained = str(getattr(config, "dinov3_pretrained", "") or "").strip()
    if pretrained:
        filename = osp.basename(pretrained)
        match = re.search(r"pretrain_([A-Za-z0-9]+)", filename)
        if match:
            parts.append(match.group(1).lower())

    return "_".join(parts)


def _prepare_wandb_aware_log_paths(config):
    backend = str(getattr(config, "log_backend", "tensorboard") or "tensorboard").strip().lower()
    if backend not in {"wandb", "wb"}:
        return

    dataset_name = _sanitize_path_component(getattr(config, "dataset_name", "dataset"), fallback="dataset")
    run_id = _sanitize_path_component(
        getattr(config, "wandb_run_id", None) or time.strftime("%Y%m%d_%H%M%S", time.localtime()),
        fallback="run",
    )
    backbone_name = _backbone_name_for_run(config)

    default_run_name = (
        f"{getattr(config, 'dataset_name', 'dataset')}_{backbone_name}_{getattr(config, 'decoder', 'decoder')}_{run_id}"
    )
    run_name_raw = getattr(config, "wandb_run_name", None) or default_run_name
    run_name_safe = _sanitize_path_component(run_name_raw, fallback="run")
    config.wandb_run_name = run_name_raw
    config.wandb_run_id = run_id

    project_root = osp.abspath(str(getattr(config, "root_dir", os.getcwd()) or os.getcwd()))
    log_root = osp.join(project_root, "logs")

    run_dir_name = run_name_safe if run_name_safe.endswith(f"_{run_id}") else f"{run_name_safe}_{run_id}"
    config.log_dir = osp.join(log_root, dataset_name, run_dir_name)
    config.tb_dir = osp.join(config.log_dir, "tb")
    config.log_dir_link = config.log_dir
    config.checkpoint_dir = osp.join(config.log_dir, "checkpoint")

    exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    config.log_file = osp.join(config.log_dir, f"log_{exp_time}.log")
    config.link_log_file = osp.join(config.log_dir, "log_last.log")
    config.val_log_file = osp.join(config.log_dir, f"val_{exp_time}.log")
    config.link_val_log_file = osp.join(config.log_dir, "val_last.log")
    config.test_log_file = osp.join(config.log_dir, "test_during_train.log")
    config.link_test_log_file = osp.join(config.log_dir, "test_last.log")


def _ensure_shared_run_id(config, engine):
    configured_run_id = str(getattr(config, "wandb_run_id", "") or "").strip()

    if not engine.distributed:
        if configured_run_id:
            config.wandb_run_id = _sanitize_path_component(configured_run_id, fallback="run")
        return

    if not dist.is_available() or not dist.is_initialized():
        fallback_run_id = configured_run_id or time.strftime("%Y%m%d_%H%M%S", time.localtime())
        config.wandb_run_id = _sanitize_path_component(fallback_run_id, fallback="run")
        return

    rank = dist.get_rank()
    shared_run_id = [
        _sanitize_path_component(
            configured_run_id or time.strftime("%Y%m%d_%H%M%S", time.localtime()),
            fallback="run",
        )
        if rank == 0
        else ""
    ]
    dist.broadcast_object_list(shared_run_id, src=0)
    config.wandb_run_id = _sanitize_path_component(shared_run_id[0], fallback="run")


def _resolve_auto_test_checkpoint(config, preferred_checkpoint_path=None):
    explicit_checkpoint = getattr(config, "auto_test_checkpoint_path", None)
    if explicit_checkpoint:
        explicit_checkpoint = osp.abspath(str(explicit_checkpoint))
        if osp.isfile(explicit_checkpoint):
            return explicit_checkpoint
        logger.warning("auto_test_checkpoint_path does not exist: %s", explicit_checkpoint)

    if preferred_checkpoint_path:
        preferred_checkpoint_path = osp.abspath(str(preferred_checkpoint_path))
        if osp.isfile(preferred_checkpoint_path):
            return preferred_checkpoint_path
        logger.warning("preferred auto-test checkpoint does not exist: %s", preferred_checkpoint_path)

    checkpoint_dir = getattr(config, "auto_test_checkpoint_dir", None) or getattr(config, "checkpoint_dir", None)
    if not checkpoint_dir:
        return None
    checkpoint_dir = osp.abspath(str(checkpoint_dir))

    best_val_checkpoint = osp.join(checkpoint_dir, "best_val.pth")
    if osp.isfile(best_val_checkpoint):
        return best_val_checkpoint

    logger.warning("best_val checkpoint not found for auto test: %s", best_val_checkpoint)
    return None


def _run_auto_test_after_training(
    config,
    engine,
    source_train_run_name=None,
    source_train_run_id=None,
    preferred_checkpoint_path=None,
):
    auto_test_enable = bool(getattr(config, "auto_test_enable", False))
    if not auto_test_enable:
        return

    checkpoint_path = _resolve_auto_test_checkpoint(config, preferred_checkpoint_path=preferred_checkpoint_path)
    if checkpoint_path is None:
        logger.warning("auto_test_enable=true but no best-val checkpoint was found. Skipping post-training test run.")
        return

    checkpoint_dir_raw = getattr(config, "auto_test_checkpoint_dir", None) or getattr(config, "checkpoint_dir", None)
    checkpoint_dir = osp.abspath(str(checkpoint_dir_raw)) if checkpoint_dir_raw else None

    eval_args = argparse.Namespace(
        epochs=str(getattr(config, "auto_test_epochs", "last") or "last"),
        devices=str(getattr(engine.args, "devices", "0") or "0"),
        verbose=bool(getattr(config, "auto_test_verbose", False)),
        show_image=bool(getattr(config, "auto_test_show_image", False)),
        save_path=getattr(config, "auto_test_save_path", None),
        log_saved_every=int(getattr(config, "auto_test_log_saved_every", 0) or 0),
        save_visualizations=bool(getattr(config, "auto_test_save_visualizations", False)),
        config_name=str(getattr(config, "dataset_name", "levir") or "levir"),
        config_path=None,
        dataset_name=None,
        split=str(getattr(config, "auto_test_split", "test") or "test"),
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=checkpoint_path,
        legacy_eval_compat=bool(getattr(config, "auto_test_legacy_eval_compat", False)),
        legacy_bgr_input=bool(getattr(config, "auto_test_legacy_bgr_input", False)),
        legacy_score_exp=bool(getattr(config, "auto_test_legacy_score_exp", False)),
        time_warmup=int(getattr(config, "auto_test_time_warmup", 0) or 0),
        local_rank=None,
        wandb_enable=bool(getattr(config, "auto_test_wandb_enable", False)) and str(getattr(config, "log_backend", "tensorboard") or "").lower() in {"wandb", "wb"},
        wandb_project=getattr(config, "wandb_project", None),
        wandb_entity=getattr(config, "wandb_entity", None),
        wandb_group=getattr(config, "wandb_group", None) or getattr(config, "dataset_name", None),
        wandb_run_name=getattr(config, "wandb_run_name", None),
        wandb_test_run_name=None,
        wandb_source_train_run_name=source_train_run_name or getattr(config, "wandb_run_name", None),
        wandb_source_train_run_id=source_train_run_id,
        wandb_job_type=str(getattr(config, "auto_test_wandb_job_type", "test") or "test"),
        wandb_tags=getattr(config, "auto_test_wandb_tags", "auto-test"),
        wandb_notes="Auto test after training",
        wandb_mode=str(getattr(config, "auto_test_wandb_mode", "online") or "online"),
        wandb_dir=getattr(config, "wandb_dir", None),
        wandb_save_code=bool(getattr(config, "wandb_save_code", True)),
    )

    logger.info("Auto test run started with checkpoint: %s", checkpoint_path)
    run_eval(args=eval_args, config=config)
    logger.info("Auto test run finished.")


def _config_to_dict(config):
    def _to_wandb_serializable(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): _to_wandb_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_to_wandb_serializable(v) for v in value]
        return value

    config_dict = {}

    # Prefer mapping-style extraction for EasyDict / dict-like config objects.
    if isinstance(config, collections.abc.Mapping):
        for key, value in config.items():
            if str(key).startswith("_") or callable(value):
                continue
            config_dict[str(key)] = _to_wandb_serializable(value)
        return config_dict

    # Fallback for object-style config.
    if hasattr(config, "__dict__"):
        for key, value in vars(config).items():
            if key.startswith("_") or callable(value):
                continue
            config_dict[key] = _to_wandb_serializable(value)

    for key in dir(config):
        if key.startswith("_") or key in config_dict:
            continue
        try:
            value = getattr(config, key)
            if not callable(value):
                config_dict[key] = _to_wandb_serializable(value)
        except Exception:
            continue

    return config_dict


class _NoOpExperimentLogger:
    def add_scalar(self, tag, value, step):
        return None

    def close(self):
        return None


class _TensorBoardLogger:
    def __init__(self, config, engine):
        from tensorboardX import SummaryWriter

        tb_dir = config.tb_dir + "/{}".format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + "/tb"
        self.writer = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)
        logger.info("TensorBoard logging enabled. tb_dir=%s", tb_dir)

    def add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, float(value), int(step))

    def close(self):
        self.writer.close()


class _WandbLogger:
    def __init__(self, config):
        import wandb

        self.wandb = wandb
        self._owns_run = False
        active_run = getattr(self.wandb, "run", None)
        if active_run is not None:
            config_dict = _config_to_dict(config)
            try:
                active_run.config.update(config_dict, allow_val_change=True)
                active_run.config.update({"runtime": config_dict}, allow_val_change=True)
            except Exception:
                pass
            logger.info("W&B logging enabled. Reusing active run: %s", getattr(active_run, "name", "<unnamed>"))
            return

        config_dict = _config_to_dict(config)
        required_fields = ["dataset_name", "batch_size", "backbone", "decoder", "lr", "nepochs"]
        missing_fields = [k for k in required_fields if k not in config_dict]
        if missing_fields:
            logger.warning("W&B config is missing expected field(s): %s", ", ".join(missing_fields))
        run_id_default = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        backbone_name = _backbone_name_for_run(config)
        default_run_name = f"{getattr(config, 'dataset_name', 'dataset')}_{backbone_name}_{getattr(config, 'decoder', 'decoder')}_{run_id_default}"
        mode = str(getattr(config, "wandb_mode", "online") or "online")
        project = getattr(config, "wandb_project", None) or "FAF-CD"
        entity = getattr(config, "wandb_entity", None)
        run_name = getattr(config, "wandb_run_name", None) or default_run_name
        group = getattr(config, "wandb_group", None)
        notes = getattr(config, "wandb_notes", None)
        run_dir = getattr(config, "wandb_dir", None) or getattr(config, "log_dir", None)
        resume = getattr(config, "wandb_resume", None)
        run_id = getattr(config, "wandb_run_id", None)
        tags = getattr(config, "wandb_tags", None)
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",") if tag.strip()]

        init_kwargs = {
            "project": project,
            "name": run_name,
            "config": config_dict,
            "mode": mode,
            "dir": run_dir,
            "save_code": bool(getattr(config, "wandb_save_code", True)),
        }
        if entity:
            init_kwargs["entity"] = entity
        if group:
            init_kwargs["group"] = group
        if notes:
            init_kwargs["notes"] = notes
        if tags:
            init_kwargs["tags"] = tags
        if resume:
            init_kwargs["resume"] = resume
        if run_id:
            init_kwargs["id"] = run_id

        git_meta = collect_git_metadata()
        # Keep flat top-level config keys so W&B table columns (e.g. dataset_name, batch_size)
        # are directly available, and store nested runtime/git metadata as extras.
        init_kwargs["config"] = dict(config_dict)
        init_kwargs["config"]["runtime"] = config_dict
        init_kwargs["config"]["git"] = git_meta
        run = self.wandb.init(**init_kwargs)
        inject_git_metadata_to_wandb_run(run, git_meta)
        self._owns_run = True
        logger.info("W&B logging enabled. project=%s run=%s mode=%s", project, run_name, mode)

    def add_scalar(self, tag, value, step):
        self.wandb.log({tag: float(value)}, step=int(step))

    def close(self):
        if self._owns_run:
            self.wandb.finish()


def _build_experiment_logger(config, engine):
    is_main_process = (engine.distributed and engine.local_rank == 0) or (not engine.distributed)
    if not is_main_process:
        return _NoOpExperimentLogger()

    backend = str(getattr(config, "log_backend", "tensorboard") or "tensorboard").strip().lower()
    if backend in {"none", "off", "disable", "disabled"}:
        logger.info("Experiment logging disabled (log_backend=%s).", backend)
        return _NoOpExperimentLogger()

    if backend in {"wandb", "wb"}:
        try:
            return _WandbLogger(config)
        except Exception as exc:
            logger.warning("Failed to initialize W&B logger (%s). Falling back to TensorBoard.", exc)
            backend = "tensorboard"

    if backend in {"tensorboard", "tb"}:
        try:
            return _TensorBoardLogger(config, engine)
        except Exception as exc:
            logger.warning("Failed to initialize TensorBoard logger (%s). Logging disabled.", exc)
            return _NoOpExperimentLogger()

    logger.warning("Unknown log_backend=%s, logging disabled.", backend)
    return _NoOpExperimentLogger()


def _set_optimizer_lrs(optimizer, lr):
    for param_group in optimizer.param_groups:
        lr_mult = param_group.get("lr_mult", 1.0)
        param_group["lr"] = lr * lr_mult


def _compute_grad_norm(module):
    grad_sq_sum = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        grad_norm = param.grad.data.norm(2)
        grad_sq_sum += float(grad_norm.item()) ** 2
    return grad_sq_sum ** 0.5


def _configure_train_schedule(config, engine, train_loader):
    grad_accum_steps = int(getattr(config, "grad_accum_steps", 1) or 1)
    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got: {grad_accum_steps}")

    dataloader_micro_steps = len(train_loader)
    if dataloader_micro_steps <= 0:
        raise ValueError(
            "Training dataloader produced zero batches. "
            "Check batch_size/world_size versus dataset size when drop_last=True."
        )

    batch_size_per_gpu = int(getattr(train_loader, "batch_size", 0) or 0)
    global_micro_batch_size = batch_size_per_gpu * engine.world_size if engine.distributed else batch_size_per_gpu
    requested_batch_size = int(getattr(config, "batch_size", global_micro_batch_size) or global_micro_batch_size)
    niters_explicit = getattr(config, "_niters_per_epoch_explicit", None)
    if niters_explicit is None:
        num_train_imgs = getattr(config, "num_train_imgs", None)
        legacy_niters = getattr(config, "niters_per_epoch", None)
        if num_train_imgs is not None and requested_batch_size and legacy_niters is not None:
            legacy_auto_niters = int(num_train_imgs) // int(requested_batch_size) + 1
            niters_explicit = int(legacy_niters) != legacy_auto_niters
        else:
            niters_explicit = False
    niters_explicit = bool(niters_explicit)
    requested_optimizer_steps = None

    if requested_batch_size != global_micro_batch_size:
        logger.warning(
            "Requested global batch_size=%d but DataLoader yields %d samples per micro-step "
            "(world_size=%d, batch_size_per_gpu=%d).",
            requested_batch_size,
            global_micro_batch_size,
            getattr(engine, "world_size", 1),
            batch_size_per_gpu,
        )

    if niters_explicit:
        requested_optimizer_steps = int(getattr(config, "niters_per_epoch", 0) or 0)
        if requested_optimizer_steps < 1:
            raise ValueError(f"niters_per_epoch must be >= 1 when explicitly set, got: {requested_optimizer_steps}")
        target_micro_steps = requested_optimizer_steps * grad_accum_steps
        if target_micro_steps > dataloader_micro_steps:
            logger.warning(
                "Requested niters_per_epoch=%d with grad_accum_steps=%d needs %d micro-steps, "
                "but the dataloader only provides %d. Clamping to available batches.",
                requested_optimizer_steps,
                grad_accum_steps,
                target_micro_steps,
                dataloader_micro_steps,
            )
            target_micro_steps = dataloader_micro_steps
    else:
        target_micro_steps = dataloader_micro_steps

    optimizer_steps_per_epoch = int(math.ceil(target_micro_steps / float(grad_accum_steps)))

    config.grad_accum_steps = grad_accum_steps
    config.batch_size_per_gpu = batch_size_per_gpu
    config.global_micro_batch_size = global_micro_batch_size
    config.effective_batch_size = global_micro_batch_size * grad_accum_steps
    config.dataloader_niters_per_epoch = dataloader_micro_steps
    config.micro_niters_per_epoch = target_micro_steps
    config.optimizer_steps_per_epoch = optimizer_steps_per_epoch
    if requested_optimizer_steps is not None:
        config.requested_niters_per_epoch = requested_optimizer_steps
    config.niters_per_epoch = optimizer_steps_per_epoch

    logger.info(
        "Training schedule: global_micro_batch=%d batch_size_per_gpu=%d grad_accum_steps=%d "
        "effective_batch_size=%d dataloader_micro_steps=%d micro_steps=%d optimizer_steps=%d",
        global_micro_batch_size,
        batch_size_per_gpu,
        grad_accum_steps,
        config.effective_batch_size,
        dataloader_micro_steps,
        target_micro_steps,
        optimizer_steps_per_epoch,
    )


def _log_training_config(config):
    logger.info("=" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 80)

    config_dict = _config_to_dict(config)

    logger.info("\n--- Dataset Configuration ---")
    dataset_keys = [
        "dataset_name", "root_folder", "A_format", "B_format", "gt_format",
        "num_train_imgs", "num_eval_imgs", "num_classes", "class_names",
    ]
    for key in dataset_keys:
        if key in config_dict:
            logger.info("  %s: %s", key, config_dict[key])

    logger.info("\n--- Image Configuration ---")
    image_keys = ["image_height", "image_width", "background", "norm_mean", "norm_std"]
    for key in image_keys:
        if key in config_dict:
            logger.info("  %s: %s", key, config_dict[key])

    logger.info("\n--- Model Configuration ---")
    model_keys = [
        "backbone", "decoder", "decoder_embed_dim", "pretrained_model", "pretrained_path",
        "dinov3_repo_dir", "dinov3_model_name", "dinov3_pretrained",
        "dinov3_out_indices", "dinov3_use_frm",
    ]
    for key in model_keys:
        if key in config_dict:
            logger.info("  %s: %s", key, config_dict[key])

    logger.info("\n--- Training Configuration ---")
    train_keys = [
        "optimizer", "lr", "lr_power", "momentum", "weight_decay", "batch_size",
        "batch_size_per_gpu", "global_micro_batch_size", "grad_accum_steps",
        "effective_batch_size", "nepochs", "niters_per_epoch", "requested_niters_per_epoch",
        "optimizer_steps_per_epoch", "micro_niters_per_epoch", "dataloader_niters_per_epoch",
        "num_workers", "train_scale_array",
        "warm_up_epoch", "fix_bias", "bn_eps", "bn_momentum", "seed",
        "use_dice", "dice_weight", "use_CrossE", "CrossE_weight", "CrossE_class_weights",
        "mask2former_train_mode", "mask2former_class_weights", "mask2former_set_class_weights",
        "mask2former_class_weight", "mask2former_dice_weight",
        "mask2former_mask_weight", "mask2former_no_object_weight",
        "mask2former_num_points", "mask2former_oversample_ratio",
        "mask2former_importance_sample_ratio", "mask2former_focal_gamma",
        "mask2former_focal_loss_weight",
    ]
    for key in train_keys:
        if key in config_dict:
            logger.info("  %s: %s", key, config_dict[key])

    logger.info("\n--- Evaluation Configuration ---")
    eval_keys = ["eval_stride_rate", "eval_scale_array", "eval_flip", "eval_crop_size"]
    for key in eval_keys:
        if key in config_dict:
            logger.info("  %s: %s", key, config_dict[key])

    logger.info("\n--- Checkpoint Configuration ---")
    checkpoint_keys = ["checkpoint_start_epoch", "checkpoint_step", "checkpoint_dir"]
    for key in checkpoint_keys:
        if key in config_dict:
            logger.info("  %s: %s", key, config_dict[key])

    logger.info("\n--- Path Configuration ---")
    path_keys = [
        "root_dir", "abs_dir", "log_dir", "tb_dir", "log_file", "val_log_file",
        "log_backend", "wandb_project", "wandb_entity", "wandb_run_name", "wandb_mode",
    ]
    for key in path_keys:
        if key in config_dict:
            logger.info("  %s: %s", key, config_dict[key])

    all_logged_keys = set(dataset_keys + image_keys + model_keys + train_keys + eval_keys + checkpoint_keys + path_keys)
    remaining_keys = sorted(k for k in config_dict.keys() if k not in all_logged_keys)
    if remaining_keys:
        logger.info("\n--- Other Configuration ---")
        for key in remaining_keys:
            logger.info("  %s: %s", key, config_dict[key])

    logger.info("=" * 80)
    logger.info("")


def run_training(config=None, engine_args=None):
    parser = argparse.ArgumentParser()
    with Engine(custom_parser=parser, args_override=engine_args) as engine:
        args = engine.args
        is_main_process = (not engine.distributed) or engine.local_rank == 0
        ddp_debug = _env_flag("FAF_CD_DDP_DEBUG", default=False)

        if is_main_process:
            print(args)

        if config is None:
            config_name = args.config_name or args.dataset_name
            if is_main_process:
                print("CONFIG NAME::  ", config_name)
            config = load_config_by_name(config_name)
        _ensure_shared_run_id(config, engine)

        if is_main_process:
            print("DINOv3 adapter:", getattr(config, "dinov3_use_adapter", None))
            print("DINOv3 freeze backbone:", getattr(config, "dinov3_freeze_backbone", None))
        logger.info("DINOv3 adapter: %s", getattr(config, "dinov3_use_adapter", None))
        logger.info("DINOv3 freeze backbone: %s", getattr(config, "dinov3_freeze_backbone", None))

        _prepare_wandb_aware_log_paths(config)
        dataset_name = str(getattr(config, "dataset_name", "")).strip().lower()
        use_map_for_model_selection = dataset_name == "bright"
        best_model_selection_metric = "mAP" if use_map_for_model_selection else "mean_IoU"
        ap_num_bins = int(getattr(config, "ap_num_bins", 512))
        config.best_model_selection_metric = best_model_selection_metric
        config.best_model_selection_metric_reason = (
            "BRIGHT dataset uses mAP primary metric" if use_map_for_model_selection else "default metric policy"
        )
        config.ap_metric_method = f"histogram AP with {ap_num_bins} bins"
        logger.info(
            "Best model selection metric: %s (dataset=%s)",
            best_model_selection_metric,
            getattr(config, "dataset_name", None),
        )
        logger.info("AP metric method: histogram AP with %d bins", ap_num_bins)

        checkpoint_root = getattr(config, "checkpoint_dir", None)
        backend = str(getattr(config, "log_backend", "tensorboard") or "tensorboard").strip().lower()
        if backend not in {"wandb", "wb"} and checkpoint_root and osp.basename(checkpoint_root) == "checkpoint":
            run_id = _sanitize_path_component(
                getattr(config, "wandb_run_id", None) or time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                fallback="run",
            )
            config.checkpoint_dir = osp.join(checkpoint_root, run_id)

        if is_main_process:
            print("=======================================")
            print(config.tb_dir)
            print("=======================================")

        cudnn.benchmark = True
        seed = config.seed
        if engine.distributed:
            seed = engine.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        train_loader, train_sampler = get_train_loader(engine, ChangeDataset, config)
        _configure_train_schedule(config, engine, train_loader)

        if is_main_process:
            _log_training_config(config)

        experiment_logger = _build_experiment_logger(config, engine)
        train_run_name = None
        train_run_id = None
        if str(getattr(config, "log_backend", "tensorboard") or "").lower() in {"wandb", "wb"}:
            try:
                import wandb

                if getattr(wandb, "run", None) is not None:
                    train_run_name = getattr(wandb.run, "name", None)
                    train_run_id = getattr(wandb.run, "id", None)
                    wandb.run.summary["best_model_selection_metric"] = best_model_selection_metric
                    wandb.run.summary["best_model_selection_dataset"] = str(getattr(config, "dataset_name", ""))
                    wandb.run.summary["ap_metric_method"] = "histogram"
                    wandb.run.summary["ap_num_bins"] = int(getattr(config, "ap_num_bins", 512))
            except Exception:
                pass

        if config.backbone != "dinov3" or config.decoder != "MambaDecoder":
            raise ValueError("FAF-CD release training supports only DINOv3 + MambaDecoder configs.")

        class_weights = getattr(config, "CrossE_class_weights", None)
        ignore_index = getattr(config, "ignore_index", 255)
        logger.info("========================================")
        logger.info("Using CrossEntropyLoss as criterion")
        logger.info("CrossE_class_weights: %s", class_weights)
        logger.info("ignore_index: %s", ignore_index)
        logger.info("========================================")
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None,
            reduction="mean",
            ignore_index=ignore_index,
        )

        BatchNorm2d = nn.SyncBatchNorm if engine.distributed else nn.BatchNorm2d
        model = segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)

        peft_strategy = str(getattr(config, "dinov3_peft_strategy", "full") or "full").strip().lower()
        use_encoder_peft = peft_strategy in {"lora", "vpt", "partial_last_stage", "convnext_adapter"}
        if config.backbone == "dinov3":
            logger.info("Encoder PEFT strategy: %s", peft_strategy)

        def _set_backbone_requires_grad(target_model, requires_grad):
            base_model = target_model.module if isinstance(target_model, DistributedDataParallel) else target_model
            if hasattr(base_model, "backbone"):
                for param in base_model.backbone.parameters():
                    param.requires_grad = requires_grad

        freeze_epochs = int(getattr(config, "dinov3_freeze_epochs", 0) or 0)
        if config.backbone == "dinov3" and freeze_epochs > 0 and not use_encoder_peft:
            _set_backbone_requires_grad(model, False)
            logger.info("Freezing DINOv3 backbone for first %d epochs", freeze_epochs)

        if hasattr(config, "pretrained_path") and config.pretrained_path and os.path.isfile(config.pretrained_path):
            logger.info("Loading pretrained weights from: %s", config.pretrained_path)
            checkpoint = torch.load(config.pretrained_path, map_location="cpu")
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            new_state_dict = {}
            for key, value in state_dict.items():
                new_state_dict[key[7:] if key.startswith("module.") else key] = value
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            logger.info("Loaded pretrained model with missing keys: %s, unexpected keys: %s", missing_keys, unexpected_keys)
        else:
            logger.info("No pretrained model loaded.")

        base_lr = config.lr
        params_list = []
        backbone_lr_mult = float(getattr(config, "dinov3_backbone_lr_mult", 1.0) or 1.0)

        def _group_weight_excluding_modules(weight_group, module, norm_layer, lr, skip_modules=None):
            skip_modules = set(skip_modules or [])
            group_decay = []
            group_no_decay = []
            for module_item in module.modules():
                if module_item in skip_modules:
                    continue
                if isinstance(module_item, nn.Linear):
                    group_decay.append(module_item.weight)
                    if module_item.bias is not None:
                        group_no_decay.append(module_item.bias)
                elif isinstance(module_item, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                    group_decay.append(module_item.weight)
                    if module_item.bias is not None:
                        group_no_decay.append(module_item.bias)
                elif isinstance(module_item, norm_layer) or isinstance(module_item, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
                    if module_item.weight is not None:
                        group_no_decay.append(module_item.weight)
                    if module_item.bias is not None:
                        group_no_decay.append(module_item.bias)
                elif isinstance(module_item, nn.Parameter):
                    group_decay.append(module_item)

            weight_group.append(dict(params=group_decay, lr=lr, lr_mult=1.0))
            weight_group.append(dict(params=group_no_decay, weight_decay=0.0, lr=lr, lr_mult=1.0))
            return weight_group

        base_model = model.module if isinstance(model, DistributedDataParallel) else model
        if config.backbone == "dinov3" and backbone_lr_mult != 1.0 and hasattr(base_model, "backbone"):
            backbone_modules = set(base_model.backbone.modules())
            params_list = _group_weight_excluding_modules(params_list, base_model, BatchNorm2d, base_lr, backbone_modules)
            params_list = group_weight(params_list, base_model.backbone, BatchNorm2d, base_lr * backbone_lr_mult)
            for group in params_list[-2:]:
                group["lr_mult"] = backbone_lr_mult
            for group in params_list[:-2]:
                group.setdefault("lr_mult", 1.0)
        else:
            params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
            for group in params_list:
                group.setdefault("lr_mult", 1.0)

        grouped_param_ids = {
            id(param)
            for group in params_list
            for param in group.get("params", [])
        }
        backbone_param_ids = set()
        if config.backbone == "dinov3" and hasattr(base_model, "backbone"):
            backbone_param_ids = {id(param) for param in base_model.backbone.parameters()}

        extra_decay_default = []
        extra_no_decay_default = []
        extra_decay_backbone = []
        extra_no_decay_backbone = []
        extra_param_names = []

        for param_name, param in base_model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in grouped_param_ids:
                continue

            param_name_lower = param_name.lower()
            # Keep adapter-style parameters stable by default.
            no_decay = (
                param.ndim <= 1
                or param_name.endswith(".bias")
                or "norm" in param_name_lower
                or "bn" in param_name_lower
                or "lora_" in param_name_lower
                or "storage_tokens" in param_name_lower
            )
            use_backbone_mult = id(param) in backbone_param_ids and backbone_lr_mult != 1.0

            if use_backbone_mult and no_decay:
                extra_no_decay_backbone.append(param)
            elif use_backbone_mult:
                extra_decay_backbone.append(param)
            elif no_decay:
                extra_no_decay_default.append(param)
            else:
                extra_decay_default.append(param)

            extra_param_names.append(param_name)

        if extra_decay_default:
            params_list.append(dict(params=extra_decay_default, lr=base_lr, lr_mult=1.0))
        if extra_no_decay_default:
            params_list.append(dict(params=extra_no_decay_default, weight_decay=0.0, lr=base_lr, lr_mult=1.0))
        if extra_decay_backbone:
            params_list.append(dict(params=extra_decay_backbone, lr=base_lr * backbone_lr_mult, lr_mult=backbone_lr_mult))
        if extra_no_decay_backbone:
            params_list.append(
                dict(
                    params=extra_no_decay_backbone,
                    weight_decay=0.0,
                    lr=base_lr * backbone_lr_mult,
                    lr_mult=backbone_lr_mult,
                )
            )
        if extra_param_names:
            logger.info(
                "Added %d trainable parameters missing from default grouping (examples: %s)",
                len(extra_param_names),
                extra_param_names[:10],
            )

        optimizer_name = str(getattr(config, "optimizer", "") or "").strip()
        optimizer_name_norm = optimizer_name.lower()
        if optimizer_name_norm == "adamw":
            optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
        elif optimizer_name_norm in {"sgdm", "sgd"}:
            optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
        elif optimizer_name_norm == "muon":
            muon_variant = str(getattr(config, "muon_variant", "muon_32_adamw32") or "muon_32_adamw32")
            muon_quantization = str(getattr(config, "muon_quantization", "none") or "none")
            muon_adamw_state_precision = str(getattr(config, "muon_adamw_state_precision", "fp32") or "fp32")
            muon_scope = str(getattr(config, "muon_scope", "full_model") or "full_model").strip().lower()
            raw_muon_targets = getattr(config, "muon_target_modules", None)
            if raw_muon_targets is None:
                muon_target_modules = []
            elif isinstance(raw_muon_targets, str):
                muon_target_modules = [raw_muon_targets]
            else:
                muon_target_modules = list(raw_muon_targets)
            muon_target_modules = [str(name).strip() for name in muon_target_modules if str(name).strip()]

            if muon_variant not in {"muon_32_adamw32", "muon_8d_adamw32"}:
                raise ValueError(
                    f"Unsupported muon_variant='{muon_variant}'. "
                    "Supported: muon_32_adamw32, muon_8d_adamw32"
                )
            if muon_quantization.lower() not in {"none", "dynamic"}:
                raise ValueError(
                    f"Unsupported muon_quantization='{muon_quantization}'. "
                    "Supported: none, dynamic"
                )
            if muon_variant == "muon_8d_adamw32" and muon_quantization.lower() != "dynamic":
                raise ValueError(
                    "muon_variant='muon_8d_adamw32' requires muon_quantization='dynamic'."
                )
            if muon_variant == "muon_32_adamw32" and muon_quantization.lower() != "none":
                raise ValueError(
                    "muon_variant='muon_32_adamw32' requires muon_quantization='none'."
                )
            if muon_adamw_state_precision.lower() not in {"fp32", "fp16", "bf16"}:
                raise ValueError(
                    f"Unsupported muon_adamw_state_precision='{muon_adamw_state_precision}'. "
                    "Supported: fp32, fp16, bf16"
                )
            if muon_scope not in {"full_model", "encoder_only", "target_modules"}:
                raise ValueError(
                    f"Unsupported muon_scope='{muon_scope}'. "
                    "Supported: full_model, encoder_only, target_modules"
                )
            if muon_scope == "target_modules" and not muon_target_modules:
                raise ValueError(
                    "muon_scope='target_modules' requires non-empty muon_target_modules "
                    "(module name prefixes from model.named_parameters())."
                )

            muon_ns_steps = int(getattr(config, "muon_ns_steps", 5) or 5)
            muon_nesterov = bool(getattr(config, "muon_nesterov", True))
            muon_momentum_cfg = getattr(config, "muon_momentum", None)
            muon_momentum = 0.95 if muon_momentum_cfg is None else float(muon_momentum_cfg)
            adam_betas = (0.9, 0.999)
            adam_eps = 1e-8

            named_params = dict(base_model.named_parameters())
            param_name_by_id = {id(param): name for name, param in named_params.items()}
            backbone_param_ids = set()
            if hasattr(base_model, "backbone"):
                backbone_param_ids = {id(param) for param in base_model.backbone.parameters()}

            def _match_target_module(param_name):
                return any(
                    param_name == target_prefix or param_name.startswith(f"{target_prefix}.")
                    for target_prefix in muon_target_modules
                )

            def _is_in_muon_scope(param):
                param_id = id(param)
                if muon_scope == "full_model":
                    return True
                if muon_scope == "encoder_only":
                    return param_id in backbone_param_ids
                param_name = param_name_by_id.get(param_id, "")
                return _match_target_module(param_name)

            muon_param_groups = []
            muon_param_count = 0
            adam_param_count = 0
            for group in params_list:
                group_params = list(group.get("params", []))
                if not group_params:
                    continue
                group_lr = float(group.get("lr", base_lr))
                group_lr_mult = float(group.get("lr_mult", 1.0))
                group_weight_decay = float(group.get("weight_decay", config.weight_decay))

                muon_params = [
                    p for p in group_params
                    if getattr(p, "ndim", 0) >= 2 and _is_in_muon_scope(p)
                ]
                muon_param_ids = {id(p) for p in muon_params}
                adam_params = [p for p in group_params if id(p) not in muon_param_ids]

                if muon_params:
                    muon_param_groups.append(
                        dict(
                            params=muon_params,
                            use_muon=True,
                            lr=group_lr,
                            lr_mult=group_lr_mult,
                            momentum=muon_momentum,
                            weight_decay=group_weight_decay,
                        )
                    )
                    muon_param_count += len(muon_params)
                if adam_params:
                    muon_param_groups.append(
                        dict(
                            params=adam_params,
                            use_muon=False,
                            lr=group_lr,
                            lr_mult=group_lr_mult,
                            betas=adam_betas,
                            eps=adam_eps,
                            weight_decay=group_weight_decay,
                        )
                    )
                    adam_param_count += len(adam_params)

            if not muon_param_groups:
                raise RuntimeError("Muon optimizer requested but no parameters were assigned to optimizer groups.")

            logger.info(
                "Using MuonWithAuxAdam (variant=%s, quantization=%s, scope=%s, targets=%s, "
                "state_precision=%s, ns_steps=%d, nesterov=%s, muon_momentum=%.4f). "
                "Assigned params: muon=%d, adam=%d",
                muon_variant,
                muon_quantization,
                muon_scope,
                muon_target_modules if muon_target_modules else "[]",
                muon_adamw_state_precision,
                muon_ns_steps,
                muon_nesterov,
                muon_momentum,
                muon_param_count,
                adam_param_count,
            )
            optimizer = MuonWithAuxAdam(
                muon_param_groups,
                ns_steps=muon_ns_steps,
                nesterov=muon_nesterov,
                muon_quantization=muon_quantization,
                adam_state_precision=muon_adamw_state_precision,
            )
        else:
            raise NotImplementedError(f"Unsupported optimizer: {optimizer_name}")

        total_iteration = config.nepochs * config.niters_per_epoch
        lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

        if engine.distributed:
            logger.info(".............distributed training.............")
            if torch.cuda.is_available():
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
                logger.info(
                    "DDP wrap start: rank=%d local_rank=%d world_size=%d",
                    rank,
                    engine.local_rank,
                    getattr(engine, "world_size", -1),
                )
                if ddp_debug:
                    print(
                        f"[ddp-debug] rank={rank} local_rank={engine.local_rank} stage=before_model_cuda",
                        flush=True,
                    )
                try:
                    model.cuda()
                    torch.cuda.synchronize(device=engine.local_rank)
                except Exception:
                    logger.exception(
                        "model.cuda() failed: rank=%d local_rank=%d world_size=%d",
                        rank,
                        engine.local_rank,
                        getattr(engine, "world_size", -1),
                    )
                    traceback.print_exc()
                    raise
                if ddp_debug:
                    print(
                        f"[ddp-debug] rank={rank} local_rank={engine.local_rank} stage=after_model_cuda",
                        flush=True,
                    )
                    print(
                        f"[ddp-debug] rank={rank} local_rank={engine.local_rank} stage=before_ddp_ctor",
                        flush=True,
                    )
                try:
                    model = DistributedDataParallel(
                        model,
                        device_ids=[engine.local_rank],
                        output_device=engine.local_rank,
                        find_unused_parameters=True,
                    )
                except Exception:
                    logger.exception(
                        "DistributedDataParallel(...) failed: rank=%d local_rank=%d world_size=%d",
                        rank,
                        engine.local_rank,
                        getattr(engine, "world_size", -1),
                    )
                    traceback.print_exc()
                    raise
                if hasattr(model, "_set_static_graph") and int(getattr(config, "grad_accum_steps", 1) or 1) == 1:
                    model._set_static_graph()
                elif hasattr(model, "_set_static_graph"):
                    logger.info("Skipping DDP static graph optimization because grad_accum_steps=%d", int(getattr(config, "grad_accum_steps", 1) or 1))
                if ddp_debug:
                    print(
                        f"[ddp-debug] rank={rank} local_rank={engine.local_rank} stage=after_ddp_ctor",
                        flush=True,
                    )
                logger.info(
                    "DDP wrap done: rank=%d local_rank=%d world_size=%d",
                    rank,
                    engine.local_rank,
                    getattr(engine, "world_size", -1),
                )
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

        engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
        if engine.continue_state_object:
            engine.restore_checkpoint()

        optimizer.zero_grad()
        model.train()
        logger.info("begin trainning:")

        val_setting = {
            "root": config.root_folder,
            "A_format": config.A_format,
            "B_format": config.B_format,
            "gt_format": config.gt_format,
            "class_names": config.class_names,
            "A_dir": getattr(config, "A_dir", "A"),
            "B_dir": getattr(config, "B_dir", "B"),
            "gt_dir": getattr(config, "gt_dir", "gt"),
            "B_grayscale": getattr(config, "B_grayscale", False),
        }

        val_pre = ValPre(gt_is_binary=getattr(config, 'gt_is_binary', True))
        val_dataset = ChangeDataset(val_setting, "val", val_pre)
        enable_val_test_dual_tracking = bool(getattr(config, "enable_val_test_dual_tracking", False))
        test_dataset = ChangeDataset(val_setting, "test", val_pre) if enable_val_test_dual_tracking else None
        change_class_index = 1 if config.num_classes > 1 else 0
        for idx, class_name in enumerate(getattr(config, "class_names", [])):
            if str(class_name).strip().lower() == "change":
                change_class_index = idx
                break

        best_mean_iou = -1.0
        best_epoch = None
        best_test_change_iou = -1.0
        best_test_map = -1.0
        best_test_epoch = None
        best_test_metrics = None
        best_val_checkpoint_path = os.path.join(config.checkpoint_dir, "best_val.pth")
        best_val_metadata_path = os.path.join(config.checkpoint_dir, "best_val.json")
        best_test_checkpoint_path = os.path.join(config.checkpoint_dir, "best_test.pth")
        auto_test_best_val_checkpoint_path = None
        keep_epoch_last_checkpoint = bool(getattr(config, "keep_epoch_last_checkpoint", True))
        epoch_last_checkpoint_path = os.path.join(config.checkpoint_dir, "epoch-last.pth")
        test_log_file = getattr(config, "test_log_file", os.path.join(config.log_dir, "test_during_train.log"))
        link_test_log_file = getattr(config, "link_test_log_file", os.path.join(config.log_dir, "test_last.log"))
        dist_eval_sync_tag = _sanitize_path_component(
            getattr(config, "wandb_run_id", None) or osp.basename(str(config.log_dir or "")) or "run",
            fallback="run",
        )
        try:
            dist_timeout_sec = int(os.environ.get("FAF_CD_DIST_TIMEOUT_SEC", "1800"))
        except (TypeError, ValueError):
            dist_timeout_sec = 1800
        try:
            dist_eval_wait_timeout_sec = int(
                os.environ.get(
                    "FAF_CD_EVAL_SYNC_TIMEOUT_SEC",
                    str(max(3600, dist_timeout_sec * 4)),
                )
            )
        except (TypeError, ValueError):
            dist_eval_wait_timeout_sec = max(3600, dist_timeout_sec * 4)

        def _safe_float(value, default=np.nan):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _extract_eval_metrics(mean_iou, dice, metrics):
            metrics = metrics if isinstance(metrics, dict) else {}
            dice = list(dice) if dice is not None else []
            iou_per_class = metrics.get("iou_per_class", [])
            change_iou = _safe_float(iou_per_class[change_class_index], np.nan) if len(iou_per_class) > change_class_index else np.nan
            change_f1 = _safe_float(dice[change_class_index], np.nan) if len(dice) > change_class_index else np.nan
            return {
                "mean_iou": _safe_float(mean_iou, np.nan),
                "mAP": _safe_float(metrics.get("mAP"), np.nan),
                "change_iou": change_iou,
                "change_f1": change_f1,
                "f1_mean": _safe_float(metrics.get("f1_mean", np.nanmean(dice) if len(dice) > 0 else np.nan), np.nan),
                "precision": _safe_float(metrics.get("precision"), np.nan),
                "recall": _safe_float(metrics.get("recall"), np.nan),
                "f1": _safe_float(metrics.get("f1"), np.nan),
                "pixel_acc": _safe_float(metrics.get("pixel_acc"), np.nan),
                "mean_pixel_acc": _safe_float(metrics.get("mean_pixel_acc"), np.nan),
            }

        def _log_eval_metrics(prefix, epoch, mean_iou, dice, metrics):
            def _add_metric_if_finite(tag, value):
                if value is None:
                    return
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    return
                if np.isnan(value):
                    return
                experiment_logger.add_scalar(tag, value, epoch)

            experiment_logger.add_scalar(f"{prefix}/mean_IoU", mean_iou, epoch)
            for class_idx, dice_score in enumerate(dice):
                if class_idx < len(config.class_names):
                    class_name = config.class_names[class_idx]
                    experiment_logger.add_scalar(f"{prefix}/f1/{class_name}", dice_score, epoch)
            iou_per_class = metrics.get("iou_per_class", []) if isinstance(metrics, dict) else []
            for class_idx, class_iou in enumerate(iou_per_class):
                if class_idx < len(config.class_names):
                    class_name = config.class_names[class_idx]
                    experiment_logger.add_scalar(f"{prefix}/iou/{class_name}", class_iou, epoch)
            if isinstance(metrics, dict) and "pixel_acc" in metrics:
                experiment_logger.add_scalar(f"{prefix}/pixel_acc", metrics["pixel_acc"], epoch)
            if isinstance(metrics, dict):
                _add_metric_if_finite(f"{prefix}/precision", metrics.get("precision"))
                _add_metric_if_finite(f"{prefix}/recall", metrics.get("recall"))
                _add_metric_if_finite(f"{prefix}/f1", metrics.get("f1"))
                _add_metric_if_finite(f"{prefix}/mAP", metrics.get("mAP"))
                ap_fg_classes = metrics.get("ap_fg_classes", []) or []
                ap_per_class = metrics.get("ap_per_class", []) or []
                for fg_idx, ap_val in zip(ap_fg_classes, ap_per_class):
                    fg_idx = int(fg_idx)
                    if fg_idx < len(config.class_names):
                        class_name = config.class_names[fg_idx]
                    else:
                        class_name = f"class_{fg_idx}"
                    _add_metric_if_finite(f"{prefix}/AP/{class_name}", ap_val)

        def _run_split_eval(split_name, dataset, epoch, log_file, link_log_file):
            devices_eval = [engine.local_rank] if engine.distributed else [0]
            segmentor = SegEvaluator(
                dataset=dataset,
                class_num=config.num_classes,
                norm_mean=config.norm_mean,
                norm_std=config.norm_std,
                network=model,
                multi_scales=config.eval_scale_array,
                is_flip=config.eval_flip,
                devices=devices_eval,
                verbose=False,
                config=config,
            )
            eval_output = segmentor.run(config.checkpoint_dir, str(epoch), log_file, link_log_file)
            mean_iou = eval_output[1]
            dice = eval_output[2] if len(eval_output) > 2 and eval_output[2] is not None else []
            metrics = eval_output[3] if len(eval_output) > 3 else {}
            logger.info("%s epoch %d mean_IoU: %.6f", split_name.upper(), epoch, mean_iou)
            return eval_output[0], mean_iou, dice, metrics

        def _cleanup_epoch_checkpoints(exclude_paths=None):
            exclude_paths = set(os.path.abspath(path) for path in (exclude_paths or []))
            for ckpt in glob.glob(os.path.join(config.checkpoint_dir, "epoch-*.pth")):
                ckpt_abs = os.path.abspath(ckpt)
                if ckpt_abs in exclude_paths:
                    continue
                if os.path.exists(ckpt):
                    os.remove(ckpt)

        def _materialize_epoch_last_from(checkpoint_path):
            if not keep_epoch_last_checkpoint:
                return
            if os.path.lexists(epoch_last_checkpoint_path):
                os.remove(epoch_last_checkpoint_path)
            shutil.copy2(checkpoint_path, epoch_last_checkpoint_path)

        def _write_best_val_metadata(epoch, metric):
            metadata = {
                "epoch": int(epoch),
                "metric": float(metric),
                "metric_name": best_model_selection_metric,
            }
            with open(best_val_metadata_path, "w", encoding="utf-8") as metadata_fp:
                json.dump(metadata, metadata_fp, sort_keys=True)
                metadata_fp.write("\n")

        def _load_best_val_metadata():
            if not os.path.isfile(best_val_checkpoint_path) or not os.path.isfile(best_val_metadata_path):
                return None, -1.0
            try:
                with open(best_val_metadata_path, "r", encoding="utf-8") as metadata_fp:
                    metadata = json.load(metadata_fp)
                return int(metadata["epoch"]), float(metadata["metric"])
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("Ignoring invalid best VAL metadata %s: %s", best_val_metadata_path, exc)
                return None, -1.0

        def _update_best_val_checkpoint(checkpoint_path, epoch, metric):
            shutil.copy2(checkpoint_path, best_val_checkpoint_path)
            _write_best_val_metadata(epoch, metric)
            return best_val_checkpoint_path

        def _dist_eval_sync_paths(epoch):
            filename_prefix = f".dist_eval_sync_{dist_eval_sync_tag}_epoch{int(epoch)}"
            return (
                os.path.join(config.checkpoint_dir, f"{filename_prefix}.done"),
                os.path.join(config.checkpoint_dir, f"{filename_prefix}.error"),
            )

        best_epoch, best_mean_iou = _load_best_val_metadata()
        if best_epoch is not None and np.isfinite(best_mean_iou):
            auto_test_best_val_checkpoint_path = best_val_checkpoint_path
            logger.info(
                "Loaded existing best VAL model metadata: epoch=%d %s=%.6f",
                best_epoch,
                best_model_selection_metric,
                best_mean_iou,
            )

        for epoch in range(engine.state.epoch, config.nepochs + 1):
            epoch_start_time = time.time()
            if torch.cuda.is_available():
                device_idx = engine.local_rank if engine.distributed else torch.cuda.current_device()
                torch.cuda.reset_peak_memory_stats(device=device_idx)
            if config.backbone == "dinov3" and freeze_epochs > 0 and (not use_encoder_peft) and epoch == freeze_epochs + 1:
                _set_backbone_requires_grad(model, True)
                logger.info("Unfroze DINOv3 backbone at epoch %d", epoch)
            if engine.distributed:
                train_sampler.set_epoch(epoch)
            bar_format = "{desc}[{elapsed}<{remaining},{rate_fmt}]"
            pbar = tqdm(
                total=config.niters_per_epoch,
                file=sys.stdout,
                bar_format=bar_format,
                disable=engine.distributed and dist.get_rank() != 0,
            )
            dataloader = iter(train_loader)
            sum_loss = 0.0
            micro_loss_sum = 0.0
            grad_norm_sum = 0.0
            grad_norm_count = 0
            optimizer_step_count = 0
            micro_step_count = 0
            last_step_lr = float(optimizer.param_groups[0].get("lr", base_lr))

            for micro_idx in range(config.micro_niters_per_epoch):
                try:
                    minibatch = next(dataloader)
                except StopIteration:
                    break
                accumulation_group_start = micro_idx - (micro_idx % config.grad_accum_steps)
                accumulation_group_size = min(
                    config.grad_accum_steps,
                    config.micro_niters_per_epoch - accumulation_group_start,
                )
                accumulation_group_offset = micro_idx - accumulation_group_start
                should_step = (accumulation_group_offset + 1) == accumulation_group_size

                if accumulation_group_offset == 0:
                    optimizer.zero_grad(set_to_none=True)
                    pending_loss_sum = 0.0
                    last_step_lr = float(optimizer.param_groups[0].get("lr", base_lr))

                As = minibatch["A"].cuda(non_blocking=True)
                Bs = minibatch["B"].cuda(non_blocking=True)
                gts = minibatch["gt"].cuda(non_blocking=True)
                sync_context = nullcontext() if (not engine.distributed or should_step) else model.no_sync()
                with sync_context:
                    loss = model(As, Bs, gts)
                    if engine.distributed:
                        reduce_loss = all_reduce_tensor(loss.detach(), world_size=engine.world_size)
                        loss_value = float(reduce_loss.item())
                    else:
                        loss_value = float(loss.detach().item())
                    pending_loss_sum += loss_value
                    micro_loss_sum += loss_value
                    (loss / float(accumulation_group_size)).backward()

                micro_step_count += 1
                if not should_step:
                    del loss
                    continue

                grad_norm_sum += _compute_grad_norm(model)
                grad_norm_count += 1
                optimizer.step()
                optimizer_step_count += 1
                engine.update_iteration(epoch, optimizer_step_count - 1)

                current_idx = (epoch - 1) * config.niters_per_epoch + (optimizer_step_count - 1)
                next_lr = lr_policy.get_lr(current_idx)
                _set_optimizer_lrs(optimizer, next_lr)

                step_loss = pending_loss_sum / float(accumulation_group_size)
                sum_loss += step_loss
                print_str = (
                    "Epoch {}/{} Step {}/{} (micro {}/{}): lr={:.4e} loss={:.4f} total_loss={:.4f}".format(
                        epoch,
                        config.nepochs,
                        optimizer_step_count,
                        config.niters_per_epoch,
                        micro_idx + 1,
                        config.micro_niters_per_epoch,
                        last_step_lr,
                        step_loss,
                        sum_loss / optimizer_step_count,
                    )
                )
                pbar.set_description(print_str, refresh=False)
                pbar.update(1)
                del loss

            pbar.close()

            experiment_logger.add_scalar("train/loss", sum_loss / max(optimizer_step_count, 1), epoch)
            experiment_logger.add_scalar("train/micro_loss", micro_loss_sum / max(micro_step_count, 1), epoch)
            experiment_logger.add_scalar("train/lr", last_step_lr, epoch)
            experiment_logger.add_scalar("train/optimizer_steps", optimizer_step_count, epoch)
            experiment_logger.add_scalar("train/micro_steps", micro_step_count, epoch)
            if grad_norm_count > 0:
                experiment_logger.add_scalar("train/grad_norm", grad_norm_sum / grad_norm_count, epoch)
            experiment_logger.add_scalar("time/epoch_sec", time.time() - epoch_start_time, epoch)
            if torch.cuda.is_available():
                device_idx = engine.local_rank if engine.distributed else torch.cuda.current_device()
                peak_mem_mb = torch.cuda.max_memory_allocated(device=device_idx) / (1024.0 * 1024.0)
                experiment_logger.add_scalar("system/gpu_mem_mb", peak_mem_mb, epoch)

            should_save_checkpoint = ((epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0)) or (epoch == config.nepochs)
            if should_save_checkpoint:
                if engine.distributed and engine.local_rank == 0:
                    engine.save_and_link_checkpoint(config.checkpoint_dir, config.log_dir, config.log_dir_link)
                elif not engine.distributed:
                    engine.save_and_link_checkpoint(config.checkpoint_dir, config.log_dir, config.log_dir_link)

            torch.cuda.empty_cache()
            is_main_process = (engine.distributed and dist.get_rank() == 0) or (not engine.distributed)
            should_eval = (epoch >= config.checkpoint_start_epoch) and ((epoch - config.checkpoint_start_epoch) % config.checkpoint_step == 0)
            eval_done_marker = None
            eval_error_marker = None
            if engine.distributed and should_eval:
                eval_done_marker, eval_error_marker = _dist_eval_sync_paths(epoch)
                if is_main_process:
                    for sync_path in (eval_done_marker, eval_error_marker):
                        if os.path.exists(sync_path):
                            os.remove(sync_path)
            if is_main_process and should_eval:
                try:
                    model.eval()
                    with torch.no_grad():
                        _, val_mean_iou, val_dice, val_metrics = _run_split_eval(
                            split_name="val",
                            dataset=val_dataset,
                            epoch=epoch,
                            log_file=config.val_log_file,
                            link_log_file=config.link_val_log_file,
                        )
                        _log_eval_metrics("val", epoch, val_mean_iou, val_dice, val_metrics)
                        checkpoint_path = os.path.join(config.checkpoint_dir, f"epoch-{epoch}.pth")

                        if enable_val_test_dual_tracking:
                            val_selection_metric = _safe_float(
                                (val_metrics or {}).get("mAP") if use_map_for_model_selection else val_mean_iou,
                                np.nan,
                            )
                            if np.isfinite(val_selection_metric) and val_selection_metric > best_mean_iou:
                                best_mean_iou = val_selection_metric
                                best_epoch = epoch
                                if os.path.exists(checkpoint_path):
                                    auto_test_best_val_checkpoint_path = _update_best_val_checkpoint(
                                        checkpoint_path,
                                        epoch,
                                        val_selection_metric,
                                    )
                                if use_map_for_model_selection:
                                    logger.info("Updated best VAL model at epoch %d (mAP=%.6f)", epoch, val_selection_metric)
                                else:
                                    logger.info("Updated best VAL model at epoch %d (mean_IoU=%.6f)", epoch, val_selection_metric)

                            _, test_mean_iou, test_dice, test_metrics = _run_split_eval(
                                split_name="test",
                                dataset=test_dataset,
                                epoch=epoch,
                                log_file=test_log_file,
                                link_log_file=link_test_log_file,
                            )
                            _log_eval_metrics("test", epoch, test_mean_iou, test_dice, test_metrics)
                            test_summary = _extract_eval_metrics(test_mean_iou, test_dice, test_metrics)
                            test_selection_metric = test_summary["mAP"] if use_map_for_model_selection else test_summary["change_iou"]
                            should_update_best_test = np.isfinite(test_selection_metric) and (
                                test_selection_metric > (best_test_map if use_map_for_model_selection else best_test_change_iou)
                            )
                            if should_update_best_test:
                                if use_map_for_model_selection:
                                    best_test_map = test_selection_metric
                                else:
                                    best_test_change_iou = test_selection_metric
                                best_test_epoch = epoch
                                best_test_metrics = test_summary
                                if os.path.exists(checkpoint_path):
                                    shutil.copy2(checkpoint_path, best_test_checkpoint_path)
                                if use_map_for_model_selection:
                                    logger.info(
                                        "Updated best TEST model at epoch %d (mAP=%.6f)",
                                        epoch,
                                        test_summary["mAP"],
                                    )
                                else:
                                    logger.info(
                                        "Updated best TEST model at epoch %d (change_IoU=%.6f, change_F1=%.6f)",
                                        epoch,
                                        test_summary["change_iou"],
                                        test_summary["change_f1"],
                                    )
                            if os.path.exists(checkpoint_path):
                                if keep_epoch_last_checkpoint:
                                    _materialize_epoch_last_from(checkpoint_path)
                                os.remove(checkpoint_path)
                            if keep_epoch_last_checkpoint:
                                _cleanup_epoch_checkpoints(exclude_paths=[epoch_last_checkpoint_path])
                            else:
                                _cleanup_epoch_checkpoints()
                        else:
                            val_selection_metric = _safe_float(
                                (val_metrics or {}).get("mAP") if use_map_for_model_selection else val_mean_iou,
                                np.nan,
                            )
                            if np.isfinite(val_selection_metric) and val_selection_metric > best_mean_iou:
                                old_best_epoch = best_epoch
                                best_epoch = epoch
                                best_mean_iou = val_selection_metric
                                if os.path.exists(checkpoint_path):
                                    auto_test_best_val_checkpoint_path = _update_best_val_checkpoint(
                                        checkpoint_path,
                                        epoch,
                                        val_selection_metric,
                                    )
                                    if use_map_for_model_selection:
                                        logger.info("Updated best VAL model at epoch %d (mAP=%.6f)", epoch, val_selection_metric)
                                    else:
                                        logger.info("Updated best VAL model at epoch %d (mean_IoU=%.6f)", epoch, val_selection_metric)
                                if old_best_epoch is not None:
                                    old_checkpoint_path = os.path.join(config.checkpoint_dir, f"epoch-{old_best_epoch}.pth")
                                    if os.path.exists(old_checkpoint_path):
                                        os.remove(old_checkpoint_path)
                            if os.path.exists(checkpoint_path):
                                if keep_epoch_last_checkpoint:
                                    _materialize_epoch_last_from(checkpoint_path)
                                if checkpoint_path != auto_test_best_val_checkpoint_path:
                                    os.remove(checkpoint_path)
                except Exception:
                    if eval_error_marker is not None:
                        with open(eval_error_marker, "w", encoding="utf-8") as error_fp:
                            error_fp.write(traceback.format_exc())
                    raise
                finally:
                    model.train()
                if eval_done_marker is not None:
                    with open(eval_done_marker, "w", encoding="utf-8") as done_fp:
                        done_fp.write("done\n")
            elif engine.distributed and should_eval:
                wait_start_time = time.time()
                while True:
                    if eval_error_marker is not None and os.path.exists(eval_error_marker):
                        error_message = "unknown rank0 evaluation error"
                        try:
                            with open(eval_error_marker, "r", encoding="utf-8") as error_fp:
                                error_message = error_fp.read().strip() or error_message
                        except OSError:
                            pass
                        raise RuntimeError(
                            f"Rank0 evaluation failed at epoch {epoch}. "
                            f"See marker file: {eval_error_marker}\n{error_message}"
                        )
                    if eval_done_marker is not None and os.path.exists(eval_done_marker):
                        break
                    if (time.time() - wait_start_time) > dist_eval_wait_timeout_sec:
                        raise TimeoutError(
                            f"Timed out waiting for rank0 evaluation sync marker at epoch {epoch}: {eval_done_marker}. "
                            f"Increase FAF_CD_EVAL_SYNC_TIMEOUT_SEC if validation/test is expected to run longer."
                        )
                    time.sleep(1.0)

        if ((engine.distributed and dist.get_rank() == 0) or (not engine.distributed)) and enable_val_test_dual_tracking:
            if keep_epoch_last_checkpoint:
                _cleanup_epoch_checkpoints(exclude_paths=[epoch_last_checkpoint_path])
            else:
                _cleanup_epoch_checkpoints()
            logger.info("=" * 80)
            logger.info("TRAINING SUMMARY (BEST TEST CHECKPOINT)")
            logger.info("=" * 80)
            if use_map_for_model_selection:
                logger.info("Best VAL mAP: %.6f at epoch %s", best_mean_iou, str(best_epoch))
            else:
                logger.info("Best VAL mean_IoU: %.6f at epoch %s", best_mean_iou, str(best_epoch))
            if best_test_metrics is not None:
                logger.info("Best TEST epoch: %d", best_test_epoch)
                if use_map_for_model_selection:
                    logger.info("Best TEST mAP: %.6f", best_test_metrics["mAP"])
                else:
                    logger.info("Best TEST change_IoU: %.6f", best_test_metrics["change_iou"])
                logger.info("Best TEST change_F1: %.6f", best_test_metrics["change_f1"])
                logger.info("Best TEST mean_IoU: %.6f", best_test_metrics["mean_iou"])
                logger.info("Best TEST F1(mean): %.6f", best_test_metrics["f1_mean"])
                logger.info("Best TEST precision: %.6f", best_test_metrics["precision"])
                logger.info("Best TEST recall: %.6f", best_test_metrics["recall"])
                logger.info("Best TEST F1(global): %.6f", best_test_metrics["f1"])
                logger.info("Best TEST pixel_acc: %.6f", best_test_metrics["pixel_acc"])
                logger.info("Best TEST mean_pixel_acc: %.6f", best_test_metrics["mean_pixel_acc"])
                logger.info("Best TEST checkpoint: %s", best_test_checkpoint_path)
                logger.info("Best VAL checkpoint: %s", best_val_checkpoint_path)
            else:
                logger.info("No TEST evaluation was recorded. Check checkpoint_start_epoch/checkpoint_step.")
            logger.info("=" * 80)

        experiment_logger.close()

        is_main_process = (engine.distributed and dist.get_rank() == 0) or (not engine.distributed)
        if is_main_process:
            _run_auto_test_after_training(
                config=config,
                engine=engine,
                source_train_run_name=train_run_name,
                source_train_run_id=train_run_id,
                preferred_checkpoint_path=auto_test_best_val_checkpoint_path,
            )


def main():
    run_training()


if __name__ == "__main__":
    main()
