import os
import cv2
import argparse
import json
import numpy as np
import time
import subprocess
import zipfile

import torch
import torch.nn as nn

from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.config_utils import load_config_by_name, load_config_by_path
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import (
    hist_info,
    compute_score,
    empty_ap_hists,
    update_ap_hists,
    merge_ap_hists,
    compute_ap_from_hists,
)
from dataloader.changeDataset import ChangeDataset
from models.builder import EncoderDecoder as segmodel
from dataloader.dataloader import ValPre
from dataloader.perturbations import (
    APPLY_TO_CHOICES,
    PERTURBATION_KINDS,
    TestPerturbation,
)
from PIL import Image

logger = get_logger()


def _save_palette_prediction(pred, output_path, class_colors):
    result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
    palette_list = list(np.array(class_colors, dtype=np.uint8).flatten())
    if len(palette_list) < 768:
        palette_list += [0] * (768 - len(palette_list))
    else:
        palette_list = palette_list[:768]
    result_img.putpalette(palette_list)
    result_img.save(output_path)


def _parse_bool_flag(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value '{value}'. Use T/F, True/False, or 1/0."
    )


def _parse_int_list(value):
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(v.strip()) for v in str(value).split(",") if v.strip()]


def _normalize_bright_sample_id(file_name, suffix):
    file_name = os.path.basename(file_name)
    stem = os.path.splitext(file_name)[0]
    suffix_stem = os.path.splitext(suffix)[0] if suffix else ""
    if suffix_stem and stem.endswith(suffix_stem):
        stem = stem[:-len(suffix_stem)]
    return stem


def _load_bright_submission_image_ids(manifest_path, b_format):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    image_id_map = {}
    for image in manifest.get("images", []):
        file_name = image.get("file_name")
        image_id = image.get("id")
        if file_name is None or image_id is None:
            continue
        sample_id = _normalize_bright_sample_id(file_name, b_format)
        image_id_map[sample_id] = int(image_id)
    if not image_id_map:
        raise ValueError(f"No image ids found in BRIGHT submission manifest: {manifest_path}")
    return image_id_map


def _coco_uncompressed_rle(binary_mask):
    pixels = np.asarray(binary_mask, dtype=np.uint8).reshape(-1, order="F")
    counts = []
    last = 0
    run_len = 0
    for pix in pixels:
        pix = 1 if pix else 0
        if pix == last:
            run_len += 1
        else:
            counts.append(run_len)
            run_len = 1
            last = pix
    counts.append(run_len)
    return counts


def _coco_rle_counts_to_string(counts):
    chars = []
    for idx, count in enumerate(counts):
        x = int(count)
        if idx > 2:
            x -= int(counts[idx - 2])
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = x != -1 if (c & 0x10) else x != 0
            if more:
                c |= 0x20
            chars.append(chr(c + 48))
    return "".join(chars)


def _encode_binary_mask_to_coco_rle(binary_mask):
    binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    try:
        from pycocotools import mask as mask_util

        rle = mask_util.encode(binary_mask)
        counts = rle.get("counts")
        if isinstance(counts, bytes):
            rle["counts"] = counts.decode("utf-8")
        return rle
    except ImportError:
        counts = _coco_uncompressed_rle(binary_mask)
        return {
            "size": [int(binary_mask.shape[0]), int(binary_mask.shape[1])],
            "counts": _coco_rle_counts_to_string(counts),
        }


def _prediction_to_bright_coco_results(
    pred,
    scores,
    sample_name,
    image_id_map,
    class_ids,
    min_area,
    score_thr,
    connectivity,
):
    if sample_name not in image_id_map:
        raise KeyError(f"Sample '{sample_name}' not found in BRIGHT submission manifest.")

    image_id = image_id_map[sample_name]
    results = []
    for class_id in class_ids:
        if class_id <= 0:
            continue
        class_mask = (pred == class_id).astype(np.uint8)
        if not np.any(class_mask):
            continue

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            class_mask,
            connectivity=connectivity,
        )
        for label_idx in range(1, num_labels):
            x, y, width, height, area = stats[label_idx].tolist()
            if area < min_area:
                continue
            component = labels == label_idx
            if scores is not None and scores.ndim == 3 and class_id < scores.shape[2]:
                score = float(np.mean(scores[:, :, class_id][component]))
            else:
                score = 1.0
            if score < score_thr:
                continue
            results.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(class_id),
                    "bbox": [float(x), float(y), float(width), float(height)],
                    "score": score,
                    "segmentation": _encode_binary_mask_to_coco_rle(component),
                }
            )
    return results


def _zip_json_member_name(output_path):
    archive_name = os.path.basename(output_path)
    if archive_name.endswith(".json.zip"):
        return archive_name[:-4]
    if archive_name.endswith(".zip"):
        return archive_name[:-4] + ".json"
    return archive_name + ".json"


def _write_bright_submission_results(results, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        ensure_dir(output_dir)

    if output_path.endswith(".zip"):
        member_name = _zip_json_member_name(output_path)
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(member_name, json.dumps(results, separators=(",", ":")))
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, separators=(",", ":"))



def _resolve_bright_submission_output(args, config, run_name):
    if args.bright_submission_output:
        output_path = os.path.abspath(args.bright_submission_output)
    else:
        base_dir = os.path.abspath(args.save_path) if args.save_path else os.path.join(config.log_dir, "bright_submission")
        output_path = os.path.join(base_dir, run_name, "predictions.zip")

    if not output_path.endswith((".json", ".zip", ".json.zip")):
        output_path += ".zip"
    return output_path



def _safe_git_output(args, cwd):
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def _collect_git_metadata(cwd=None):
    cwd = cwd or os.getcwd()
    meta = {
        "available": False,
        "commit": None,
        "commit_short": None,
        "branch": None,
        "remote": None,
        "dirty": None,
    }

    commit = _safe_git_output(["rev-parse", "HEAD"], cwd)
    if not commit:
        return meta

    meta["available"] = True
    meta["commit"] = commit
    meta["commit_short"] = _safe_git_output(["rev-parse", "--short", "HEAD"], cwd)
    meta["branch"] = _safe_git_output(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    meta["remote"] = _safe_git_output(["config", "--get", "remote.origin.url"], cwd)
    status = _safe_git_output(["status", "--porcelain"], cwd)
    if status is not None:
        meta["dirty"] = bool(status)
    return meta


def _normalize_tags(tags):
    if tags is None:
        return None
    if isinstance(tags, str):
        tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
    if isinstance(tags, (list, tuple, set)):
        cleaned = [str(tag).strip() for tag in tags if str(tag).strip()]
        return cleaned or None
    return [str(tags)]


def _build_checkpoint_tag(args):
    if args.checkpoint_path:
        return os.path.splitext(os.path.basename(os.path.abspath(args.checkpoint_path)))[0]
    return f"epoch_{args.epochs}"


def _init_wandb_test_run(args, config, config_tag):
    if not bool(getattr(args, "wandb_enable", False)):
        return None

    import wandb

    checkpoint_tag = _build_checkpoint_tag(args)
    split = getattr(args, "split", "test")
    source_train_run_name = getattr(args, "wandb_source_train_run_name", None)
    source_train_run_id = getattr(args, "wandb_source_train_run_id", None)

    base_run_name = source_train_run_name or getattr(args, "wandb_run_name", None) or config_tag
    test_run_name = getattr(args, "wandb_test_run_name", None) or f"{base_run_name}__test__{split}__{checkpoint_tag}"

    tags = _normalize_tags(getattr(args, "wandb_tags", None)) or []
    tags.extend(["test", split])
    tags = list(dict.fromkeys(tags))

    git_meta = _collect_git_metadata()
    wandb_cfg = {
        "task": "evaluation",
        "split": split,
        "config_tag": config_tag,
        "checkpoint_tag": checkpoint_tag,
        "source_train_run_name": source_train_run_name,
        "source_train_run_id": source_train_run_id,
        "source_checkpoint_path": getattr(args, "checkpoint_path", None),
        "source_checkpoint_dir": getattr(args, "checkpoint_dir", None),
        "git": git_meta,
    }

    run = wandb.init(
        project=getattr(args, "wandb_project", None) or "FAF-CD",
        entity=getattr(args, "wandb_entity", None),
        group=getattr(args, "wandb_group", None),
        name=test_run_name,
        id=getattr(args, "wandb_run_id", None),
        resume=getattr(args, "wandb_resume", None),
        job_type=getattr(args, "wandb_job_type", None) or "test",
        tags=tags,
        notes=getattr(args, "wandb_notes", None),
        mode=getattr(args, "wandb_mode", None) or "online",
        dir=getattr(args, "wandb_dir", None) or getattr(config, "log_dir", None),
        config=wandb_cfg,
        save_code=bool(getattr(args, "wandb_save_code", True)),
    )
    try:
        run.config.update({"git": git_meta}, allow_val_change=True)
    except Exception:
        pass
    return run


def _log_test_metrics_to_wandb(run, mean_iou, dice_per_class, metrics, class_names=None):
    if run is None:
        return
    import wandb

    if isinstance(metrics, dict) and metrics.get("has_gt") is False:
        payload = {
            "test/has_gt": 0,
            "test/num_samples": int(metrics.get("num_samples", 0)),
            "test/num_labeled_samples": 0,
        }
        wandb.log(payload)
        summary = getattr(run, "summary", None)
        if summary is not None:
            summary["test/has_gt"] = 0
            summary["test/num_samples"] = payload["test/num_samples"]
        return

    class_names = list(class_names or [])

    def _class_tag(index):
        if index < len(class_names) and class_names[index] is not None:
            return str(class_names[index])
        return f"class_{index}"

    payload = {
        "test/mean_IoU": float(mean_iou),
    }
    if isinstance(metrics, dict):
        for key in ["precision", "recall", "f1", "pixel_acc"]:
            if key in metrics and metrics[key] is not None:
                payload[f"test/{key}"] = float(metrics[key])

        if "mean_pixel_acc" in metrics and metrics["mean_pixel_acc"] is not None:
            payload["test/mean_pixel_acc"] = float(metrics["mean_pixel_acc"])
        if "freq_iou" in metrics and metrics["freq_iou"] is not None:
            payload["test/freq_iou"] = float(metrics["freq_iou"])

        iou_per_class = metrics.get("iou_per_class", [])
        for idx, score in enumerate(iou_per_class):
            payload[f"test/iou/{_class_tag(idx)}"] = float(score)

        if "mAP" in metrics and metrics["mAP"] is not None:
            payload["test/mAP"] = float(metrics["mAP"])
        if "ap_num_bins" in metrics and metrics["ap_num_bins"] is not None:
            payload["test/ap_num_bins"] = int(metrics["ap_num_bins"])
            payload["test/ap_method_histogram"] = 1
        ap_fg_classes = metrics.get("ap_fg_classes", []) or []
        ap_per_class = metrics.get("ap_per_class", []) or []
        for fg_idx, ap_val in zip(ap_fg_classes, ap_per_class):
            payload[f"test/AP/{_class_tag(int(fg_idx))}"] = float(ap_val)

    if dice_per_class is not None:
        for idx, score in enumerate(dice_per_class):
            payload[f"test/f1/{_class_tag(idx)}"] = float(score)

    wandb.log(payload)
    summary = getattr(run, "summary", None)
    if summary is not None:
        summary["best_test/mean_IoU"] = float(mean_iou)

class SegEvaluator(Evaluator):
    def __init__(
        self,
        *args,
        log_saved_every=0,
        time_warmup=0,
        bright_submission=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.log_saved_every = int(log_saved_every)
        self._num_saved = 0
        self.time_warmup = max(0, int(time_warmup))
        self._timed_samples = 0
        self.bright_submission = bright_submission

    def func_per_iteration(self, data, device, config):
        As = data['A']
        Bs = data['B']
        label = data['gt']
        name = data['fn']
        has_gt = bool(data.get('has_gt', True))
        fn = name + '.png'
        raw_output_path = os.path.join(self.save_path, "raw", fn) if self.save_path is not None else None

        compute_map = int(getattr(config, 'num_classes', 0)) >= 2

        infer_start = time.perf_counter()
        if compute_map:
            pred, scores = self.sliding_eval_rgbX(
                As, Bs, config.eval_crop_size, config.eval_stride_rate, device,
                return_scores=True,
            )
        else:
            pred = self.sliding_eval_rgbX(As, Bs, config.eval_crop_size, config.eval_stride_rate, device)
            scores = None
        inference_time_ms = (time.perf_counter() - infer_start) * 1000.0
        self._timed_samples += 1
        should_count_time = self._timed_samples > self.time_warmup
        results_dict = {'has_gt': has_gt}
        if self.bright_submission is not None:
            results_dict['bright_submission_results'] = _prediction_to_bright_coco_results(
                pred=pred,
                scores=scores,
                sample_name=name,
                image_id_map=self.bright_submission['image_id_map'],
                class_ids=self.bright_submission['class_ids'],
                min_area=self.bright_submission['min_area'],
                score_thr=self.bright_submission['score_thr'],
                connectivity=self.bright_submission['connectivity'],
            )
        if has_gt:
            hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
            results_dict.update({'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp})
            if compute_map and scores is not None:
                ap_num_bins = int(getattr(config, "ap_num_bins", 512))
                ap_hists = empty_ap_hists(int(config.num_classes), num_bins=ap_num_bins)
                update_ap_hists(ap_hists, scores, label, ignore_index=255)
                results_dict['ap_hists'] = ap_hists
        if should_count_time:
            results_dict['inference_time_ms'] = inference_time_ms

        if self.save_path is not None:
            raw_dir = os.path.join(self.save_path, "raw")
            color_dir = os.path.join(self.save_path, "color")
            paper_qual_dir = os.path.join(self.save_path, "paper_qualitative")
            ensure_dir(raw_dir)
            ensure_dir(color_dir)
            ensure_dir(paper_qual_dir)

            # save colored result
            class_colors = self.dataset.get_class_colors()
            _save_palette_prediction(pred, os.path.join(color_dir, fn), class_colors)

            # save raw result
            cv2.imwrite(raw_output_path, pred)

            # paper-style TP/TN/FP/FN qualitative map is only meaningful for binary CD.
            if has_gt and self.config.num_classes == 2:
                pred_pos = pred == 1
                gt_pos = label == 1
                tp = pred_pos & gt_pos
                tn = (~pred_pos) & (~gt_pos)
                fp = pred_pos & (~gt_pos)
                fn_mask = (~pred_pos) & gt_pos

                qual = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
                qual[tn] = [0, 0, 0]
                qual[tp] = [255, 255, 255]
                qual[fp] = [0, 255, 0]
                qual[fn_mask] = [255, 0, 0]
                cv2.imwrite(os.path.join(paper_qual_dir, fn), cv2.cvtColor(qual, cv2.COLOR_RGB2BGR))

            self._num_saved += 1
            if self.log_saved_every > 0 and self._num_saved % self.log_saved_every == 0:
                logger.info(f"Saved {self._num_saved} predictions")

        if self.show_image:
            colors = self.dataset.get_class_colors()
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((self.config.num_classes, self.config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        unlabeled_count = 0
        inference_times_ms = []
        ap_hists_total = None
        bright_submission_results = []
        for d in results:
            if not d.get('has_gt', True):
                unlabeled_count += 1
            elif 'hist' in d:
                hist += d['hist']
                correct += d['correct']
                labeled += d['labeled']
                count += 1
            if 'inference_time_ms' in d:
                inference_times_ms.append(d['inference_time_ms'])
            if 'ap_hists' in d and d['ap_hists'] is not None:
                if ap_hists_total is None:
                    ap_hists_total = d['ap_hists']
                else:
                    merge_ap_hists(ap_hists_total, d['ap_hists'])
            if 'bright_submission_results' in d:
                bright_submission_results.extend(d['bright_submission_results'])

        print("correct: ", correct, " labeled: ", labeled, " count: ", count)

        def _append_bright_submission(result_line):
            if self.bright_submission is None:
                return result_line
            if len(results) != self.ndata:
                return result_line

            bright_submission_results.sort(
                key=lambda item: (
                    item["image_id"],
                    item["category_id"],
                    -item["score"],
                    item["bbox"][1],
                    item["bbox"][0],
                )
            )
            output_path = self.bright_submission['output_path']
            _write_bright_submission_results(bright_submission_results, output_path)
            logger.info(
                "Saved BRIGHT challenge submission: %s (%d detections)",
                output_path,
                len(bright_submission_results),
            )
            result_line += (
                f"\nBRIGHT submission output: {output_path}"
                f"\nBRIGHT submission detections: {len(bright_submission_results)}\n"
            )
            return result_line


        def _append_timing(result_line):
            if len(inference_times_ms) > 0:
                avg_ms = float(np.mean(inference_times_ms))
                std_ms = float(np.std(inference_times_ms))
                logger.info(
                    "Average Inference Time Per Image Pair (ms): %.3f | Std (ms): %.3f | Timed Pairs: %d | Warmup Skipped: %d",
                    avg_ms,
                    std_ms,
                    len(inference_times_ms),
                    self.time_warmup,
                )
                result_line += (
                    f"\nAverage Inference Time Per Image Pair (ms): {avg_ms:.3f}"
                    f"\nInference Time Std (ms): {std_ms:.3f}"
                    f"\nTimed Pairs: {len(inference_times_ms)}"
                    f"\nTiming Warmup Skipped: {self.time_warmup}\n"
                )
            else:
                logger.info(
                    "Average Inference Time Per Image Pair (ms): N/A | Timed Pairs: 0 | Warmup Skipped: %d",
                    self.time_warmup,
                )
                result_line += (
                    "\nAverage Inference Time Per Image Pair (ms): N/A"
                    f"\nTimed Pairs: 0"
                    f"\nTiming Warmup Skipped: {self.time_warmup}\n"
                )
            return result_line

        if count == 0:
            result_line = (
                "No ground-truth labels available for this split; "
                f"saved predictions for {unlabeled_count} image pairs and skipped IoU/mAP metrics.\n"
            )
            result_line = _append_bright_submission(result_line)
            result_line = _append_timing(result_line)
            metrics = {
                "has_gt": False,
                "num_samples": int(unlabeled_count),
                "num_labeled_samples": 0,
            }
            return result_line, float('nan'), None, metrics

        iou, recall, precision, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc, dice_scalar, dice_per_class = compute_score(hist, correct, labeled)
        result_line = print_iou(iou, recall, precision, freq_IoU, mean_pixel_acc, pixel_acc, dice_scalar,
                                self.dataset.class_names, show_no_back=False)
        f1_global = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        if len(dice_per_class) > 2:
            f1_mean = float(np.nanmean(dice_per_class[1:]))
        else:
            f1_mean = float(np.nanmean(dice_per_class))

        metrics = {
            "iou_per_class": [float(x) for x in iou.tolist()],
            "dice_per_class": [float(x) for x in dice_per_class.tolist()],
            "mean_iou": float(mean_IoU),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1_global),
            "freq_iou": float(freq_IoU),
            "mean_pixel_acc": float(mean_pixel_acc),
            "pixel_acc": float(pixel_acc),
            "dice_change": float(dice_scalar),
            "f1_mean": f1_mean,
        }

        if ap_hists_total is not None:
            mean_ap, ap_per_class = compute_ap_from_hists(ap_hists_total)
            metrics["mAP"] = float(mean_ap)
            metrics["ap_fg_classes"] = list(ap_hists_total['fg_classes'])
            metrics["ap_per_class"] = [float(v) for v in ap_per_class]
            metrics["ap_num_bins"] = int(ap_hists_total.get("num_bins", 512))
            class_names = getattr(self.dataset, 'class_names', None) or []
            ap_lines = []
            for fg_idx, ap_val in zip(ap_hists_total['fg_classes'], ap_per_class):
                tag = (
                    class_names[fg_idx]
                    if class_names and fg_idx < len(class_names) and class_names[fg_idx] is not None
                    else f"class_{fg_idx}"
                )
                ap_lines.append(f"AP({tag})={ap_val * 100:.3f}%")
            ap_summary = (
                "mAP: {:.3f}%  |  ".format(mean_ap * 100)
                + "  ".join(ap_lines)
                + f"  |  AP method: histogram AP with {int(ap_hists_total.get('num_bins', 512))} bins"
            )
            logger.info(ap_summary)
            result_line += "\n" + ap_summary + "\n"
        if unlabeled_count > 0:
            result_line += f"\nSkipped unlabeled samples: {unlabeled_count}\n"
            metrics["num_unlabeled_samples"] = int(unlabeled_count)
        result_line = _append_bright_submission(result_line)
        result_line = _append_timing(result_line)
        return result_line, mean_IoU, dice_per_class, metrics

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)
    parser.add_argument(
        '--log_saved_every',
        type=int,
        default=0,
        help='log every N saved visualization files; 0 disables per-save logging',
    )
    parser.add_argument(
        '--save_visualizations',
        default=None,
        type=_parse_bool_flag,
        help='set True/False to enable or disable saving predicted masks',
    )
    parser.add_argument(
        '--config_name', '-n', default='faf_cd.levir_dinov3_convnext_large', type=str,
        help='config name, e.g. faf_cd.levir_dinov3_convnext_large'
    )
    parser.add_argument(
        '--config_path', default=None, type=str,
        help='path to config python file that defines `config`'
    )
    parser.add_argument(
        '--dataset_name', default=None, type=str,
        help='deprecated: use --config_name instead'
    )
    parser.add_argument('--split', '-c', default='val', type=str)
    parser.add_argument(
        '--checkpoint_dir', '-k', default=None, type=str,
        help='path to checkpoint directory (overrides config.checkpoint_dir)'
    )
    parser.add_argument(
        '--checkpoint_path', default=None, type=str,
        help='path to a specific checkpoint file (.pth), overrides --checkpoint_dir/--epochs'
    )
    parser.add_argument(
        '--load_backbone_pretrain',
        action='store_true',
        help='load the DINOv3 pretrain file before loading the evaluation checkpoint; usually unnecessary for full FAF-CD checkpoints',
    )
    parser.add_argument(
        '--legacy_eval_compat',
        action='store_true',
        help='enable compatibility mode for checkpoints trained with BGR input and exp(score) aggregation',
    )
    parser.add_argument(
        '--legacy_bgr_input',
        action='store_true',
        help='read RGB image pairs in OpenCV BGR channel order for checkpoints trained with the old loader',
    )
    parser.add_argument(
        '--legacy_score_exp',
        action='store_true',
        help='apply exp(score) before evaluator aggregation, matching the old eval path',
    )
    parser.add_argument(
        '--time_warmup',
        default=0,
        type=int,
        help='number of initial image pairs to skip when averaging inference time',
    )
    parser.add_argument(
        '--save_bright_submission',
        action='store_true',
        help='also export BRIGHT challenge COCO instance-segmentation predictions',
    )
    parser.add_argument(
        '--bright_submission_output',
        default=None,
        type=str,
        help='output .zip or .json path for BRIGHT challenge predictions; default is predictions.zip under the run output dir',
    )
    parser.add_argument(
        '--bright_submission_manifest',
        default=None,
        type=str,
        help='BRIGHT test_manifest/holdout COCO image manifest; defaults to <root_folder>/test_manifest.json',
    )
    parser.add_argument(
        '--bright_submission_class_ids',
        default='1,2,3',
        type=str,
        help='comma-separated foreground class/category ids to export for BRIGHT, default 1,2,3',
    )
    parser.add_argument(
        '--bright_submission_min_area',
        default=1,
        type=int,
        help='minimum connected-component area in pixels for exported BRIGHT instances',
    )
    parser.add_argument(
        '--bright_submission_score_thr',
        default=0.0,
        type=float,
        help='minimum instance score for exported BRIGHT instances',
    )
    parser.add_argument(
        '--bright_submission_connectivity',
        default=8,
        type=int,
        choices=[4, 8],
        help='connected-component connectivity used to convert semantic masks to BRIGHT instances',
    )
    # torch.distributed.launch passes --local-rank; accept and ignore for single-GPU eval
    parser.add_argument('--local-rank', type=int, default=None)

    parser.add_argument('--wandb_enable', type=_parse_bool_flag, default=False)
    parser.add_argument('--wandb_project', type=str, default='FAF-CD')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_test_run_name', type=str, default=None)
    parser.add_argument('--wandb_source_train_run_name', type=str, default=None)
    parser.add_argument('--wandb_source_train_run_id', type=str, default=None)
    parser.add_argument('--wandb_job_type', type=str, default='test')
    parser.add_argument('--wandb_tags', type=str, default=None)
    parser.add_argument('--wandb_notes', type=str, default=None)
    parser.add_argument('--wandb_mode', type=str, default='online')
    parser.add_argument('--wandb_dir', type=str, default=None)
    parser.add_argument('--wandb_save_code', type=_parse_bool_flag, default=True)
    parser.add_argument('--wandb_run_id', type=str, default=None)
    parser.add_argument('--wandb_resume', type=str, default=None)

    parser.add_argument(
        '--perturbation_kind', type=str, default=None,
        choices=list(PERTURBATION_KINDS),
        help='If set, apply a pseudo-change perturbation to test images (robustness eval).'
    )
    parser.add_argument(
        '--perturbation_severity', type=int, default=3,
        help='Severity 1-5 for the chosen perturbation (ImageNet-C style).'
    )
    parser.add_argument(
        '--perturbation_apply_to', type=str, default='both',
        choices=list(APPLY_TO_CHOICES),
        help="Which branch to perturb: A, B, or both (default; A and B perturbed independently)."
    )
    parser.add_argument(
        '--perturbation_seed', type=int, default=0,
        help='Global seed for per-sample deterministic perturbations.'
    )
    return parser


def run_eval(args=None, config=None):
    parser = build_parser()
    args = parser.parse_args() if args is None else parser.parse_args([], namespace=args)
    all_dev = parse_devices(args.devices)
    
    if config is not None:
        config_tag = getattr(config, "dataset_name", "config")
    elif args.config_path:
        config = load_config_by_path(args.config_path)
        config_tag = os.path.splitext(os.path.basename(args.config_path))[0]
    else:
        config_name = args.config_name or args.dataset_name
        config = load_config_by_name(config_name)
        config_tag = (config_name or "config").replace("/", "_")

    if args.legacy_eval_compat or args.legacy_bgr_input:
        setattr(config, 'legacy_bgr_input', True)
    if args.legacy_eval_compat or args.legacy_score_exp:
        setattr(config, 'eval_legacy_exp_scores', True)

    checkpoint_dir = config.checkpoint_dir
    if args.checkpoint_dir:
        checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    checkpoint_path = None
    if args.checkpoint_path:
        checkpoint_path = os.path.abspath(args.checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        checkpoint_tag = os.path.splitext(os.path.basename(checkpoint_path))[0]
    else:
        checkpoint_tag = f"epoch_{args.epochs}"

    if checkpoint_path is not None and not args.load_backbone_pretrain and hasattr(config, 'dinov3_pretrained'):
        logger.info("Skipping DINOv3 pretrain load because a full evaluation checkpoint was provided.")
        config.dinov3_pretrained = None

    run_name = f"{config_tag}__{checkpoint_tag}__{args.split}"

    save_visualizations = args.save_visualizations
    if save_visualizations is None:
        save_visualizations = args.save_path is not None
    save_path = None
    if save_visualizations:
        if not args.save_path:
            raise ValueError("--save_visualizations is True but --save_path is not set.")
        save_path = os.path.join(os.path.abspath(args.save_path), run_name)
        logger.info(f"Visualization outputs will be saved under: {save_path}")

    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    flops = network.flops()
    print("Gflops of the network: ", flops/(10**9))
    print("number of paramters: ", sum(p.numel() if p.requires_grad==True else 0 for p in network.parameters()))
    # 1/0
    data_setting = {'root': config.root_folder,
                    'A_format': config.A_format,
                    'B_format': config.B_format,
                    'gt_format': config.gt_format,
                    'class_names': config.class_names,
                    'A_dir': getattr(config, 'A_dir', 'A'),
                    'B_dir': getattr(config, 'B_dir', 'B'),
                    'gt_dir': getattr(config, 'gt_dir', 'gt'),
                    'B_grayscale': getattr(config, 'B_grayscale', False),
                    'legacy_bgr_input': getattr(config, 'legacy_bgr_input', False)}
    perturbation = None
    if getattr(args, 'perturbation_kind', None):
        perturbation = TestPerturbation(
            kind=args.perturbation_kind,
            severity=args.perturbation_severity,
            apply_to=args.perturbation_apply_to,
            seed=args.perturbation_seed,
        )
        logger.info(
            "Test-time perturbation enabled: kind=%s severity=%d apply_to=%s seed=%d",
            args.perturbation_kind,
            args.perturbation_severity,
            args.perturbation_apply_to,
            args.perturbation_seed,
        )
    val_pre = ValPre(
        gt_is_binary=getattr(config, 'gt_is_binary', True),
        perturbation=perturbation,
    )
    dataset = ChangeDataset(data_setting, args.split, val_pre)
    bright_submission = None
    if args.save_bright_submission:
        if int(getattr(config, "num_classes", 0)) < 4:
            raise ValueError("BRIGHT submission export expects a 4-class BRIGHT config.")
        if args.split != "test":
            logger.warning(
                "BRIGHT submission export is usually intended for split=test, got split=%s",
                args.split,
            )
        manifest_path = args.bright_submission_manifest or os.path.join(config.root_folder, "test_manifest.json")
        manifest_path = os.path.abspath(manifest_path)
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                f"BRIGHT submission manifest not found: {manifest_path}. "
                "Pass --bright_submission_manifest to point at the official image manifest."
            )
        image_id_map = _load_bright_submission_image_ids(manifest_path, getattr(config, 'B_format', ''))
        dataset_names = list(getattr(dataset, "_file_names", []))
        missing_names = [name for name in dataset_names if name not in image_id_map]
        if missing_names:
            raise ValueError(
                "BRIGHT submission manifest is missing image ids for "
                f"{len(missing_names)} dataset samples, e.g. {missing_names[:5]}"
            )
        bright_submission = {
            "manifest_path": manifest_path,
            "output_path": _resolve_bright_submission_output(args, config, run_name),
            "image_id_map": image_id_map,
            "class_ids": _parse_int_list(args.bright_submission_class_ids),
            "min_area": max(1, int(args.bright_submission_min_area)),
            "score_thr": float(args.bright_submission_score_thr),
            "connectivity": int(args.bright_submission_connectivity),
        }
        logger.info(
            "BRIGHT submission export enabled: manifest=%s output=%s class_ids=%s min_area=%d score_thr=%.4f",
            bright_submission["manifest_path"],
            bright_submission["output_path"],
            bright_submission["class_ids"],
            bright_submission["min_area"],
            bright_submission["score_thr"],
        )
 
    wandb_run = _init_wandb_test_run(args=args, config=config, config_tag=config_tag)
    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
                                 config.norm_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, save_path,
                                 args.show_image, config,
                                 log_saved_every=args.log_saved_every,
                                 time_warmup=args.time_warmup,
                                 bright_submission=bright_submission)
        model_indice = checkpoint_path if checkpoint_path is not None else args.epochs
        try:
            _, mean_IoU, dice_per_class, metrics = segmentor.run_eval(
                checkpoint_dir,
                model_indice,
                config.val_log_file,
                config.link_val_log_file,
            )
            _log_test_metrics_to_wandb(
                wandb_run,
                mean_IoU,
                dice_per_class,
                metrics,
                class_names=getattr(config, "class_names", None),
            )
        finally:
            if wandb_run is not None:
                import wandb

                wandb.finish()

    #visualize erf

    # with torch.enable_grad():
    #     segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
    #                                 config.norm_std, network,
    #                                 config.eval_scale_array, config.eval_flip,
    #                                 all_dev, args.verbose, args.save_path,
    #                                 args.show_image, config)
        
    #     segmentor.get_erf(config.checkpoint_dir, args.epochs)


def main():
    run_eval()


if __name__ == "__main__":
    main()
