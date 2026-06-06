# encoding: utf-8

import numpy as np

np.seterr(divide='ignore', invalid='ignore')


def hist_info(n_cl, pred, gt):
    assert (pred.shape == gt.shape)
    k = (gt >= 0) & (gt < n_cl)
    labeled = np.sum(k)
    correct = np.sum((pred[k] == gt[k]))
    confusionMatrix = np.bincount(n_cl * gt[k].astype(int) + pred[k].astype(int),
                        minlength=n_cl ** 2).reshape(n_cl, n_cl)
    return confusionMatrix, labeled, correct

def compute_score(hist, correct, labeled):
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IoU = np.nanmean(iou)
    mean_IoU_no_back = np.nanmean(iou[1:]) # useless for NYUDv2
    tp = np.diag(hist).astype(float)
    gt_count = hist.sum(axis=1).astype(float)
    pred_count = hist.sum(axis=0).astype(float)

    recall_per_class = np.divide(tp, gt_count, out=np.full_like(tp, np.nan, dtype=float), where=gt_count > 0)
    precision_per_class = np.divide(tp, pred_count, out=np.full_like(tp, np.nan, dtype=float), where=pred_count > 0)

    # Backward compatibility for binary CD:
    # keep class-1 foreground precision/recall exactly as before.
    if hist.shape[0] <= 2:
        recall = recall_per_class[1] if recall_per_class.size > 1 else recall_per_class[0]
        precision = precision_per_class[1] if precision_per_class.size > 1 else precision_per_class[0]
    else:
        # Multi-class: report macro foreground precision/recall (exclude background class 0).
        recall = np.nanmean(recall_per_class[1:])
        precision = np.nanmean(precision_per_class[1:])

    # Per-class dice
    fp = hist.sum(axis=0) - tp
    fn = hist.sum(axis=1) - tp
    denom = (2 * tp + fp + fn)
    dice_per_class = np.divide(2 * tp, denom, out=np.full_like(tp, np.nan, dtype=float), where=denom > 0)
    if dice_per_class.size <= 1:
        dice_scalar = dice_per_class[0]
    elif dice_per_class.size == 2:
        dice_scalar = dice_per_class[1]
    else:
        # Multi-class: summarize with mean foreground dice.
        dice_scalar = np.nanmean(dice_per_class[1:])

    freq = hist.sum(1) / hist.sum()
    freq_IoU = (iou[freq > 0] * freq[freq > 0]).sum()

    classAcc = np.diag(hist) / hist.sum(axis=1)
    mean_pixel_acc = np.nanmean(classAcc)

    pixel_acc = correct / labeled

    return iou, recall, precision, mean_IoU, mean_IoU_no_back, freq_IoU, mean_pixel_acc, pixel_acc, dice_scalar, dice_per_class


# ---------------------------------------------------------------------------
# Average Precision (binned, streaming) — used for multi-class semantic
# segmentation datasets such as BRIGHT where the primary metric is mAP over
# the foreground (damage) classes.
# ---------------------------------------------------------------------------

def _default_fg_classes(num_classes):
    if num_classes <= 1:
        return [0]
    return list(range(1, num_classes))


def empty_ap_hists(num_classes, num_bins=512, fg_classes=None):
    """Allocate empty positive/negative score histograms for per-class AP."""
    if fg_classes is None:
        fg_classes = _default_fg_classes(num_classes)
    fg_classes = list(fg_classes)
    n = len(fg_classes)
    return {
        'fg_classes': fg_classes,
        'num_bins': int(num_bins),
        'pos_hist': np.zeros((n, num_bins), dtype=np.int64),
        'neg_hist': np.zeros((n, num_bins), dtype=np.int64),
    }


def update_ap_hists(hists, scores, gt, ignore_index=255):
    """Accumulate one image's per-pixel class scores into the histograms.

    Args:
        hists: dict from `empty_ap_hists`.
        scores: float array of shape (H, W, C) — per-class confidence in [0, 1]
            (typically a softmax over the accumulated sliding-window logits).
        gt: int array of shape (H, W) — semantic label per pixel.
        ignore_index: pixels with this gt value are excluded.
    """
    num_bins = hists['num_bins']
    fg_classes = hists['fg_classes']
    pos_hist = hists['pos_hist']
    neg_hist = hists['neg_hist']

    valid = gt != ignore_index
    if not np.any(valid):
        return hists
    gt_v = gt[valid]
    scores_v = scores[valid]  # shape (N, C)

    for i, c in enumerate(fg_classes):
        sc = scores_v[:, c]
        bins = np.clip((sc * num_bins).astype(np.int64), 0, num_bins - 1)
        pos_mask = gt_v == c
        if pos_mask.any():
            pos_hist[i] += np.bincount(bins[pos_mask], minlength=num_bins)
        if (~pos_mask).any():
            neg_hist[i] += np.bincount(bins[~pos_mask], minlength=num_bins)
    return hists


def merge_ap_hists(target, other):
    """In-place merge of two histogram dicts (must share fg_classes/num_bins)."""
    if target is None:
        return other
    if other is None:
        return target
    target['pos_hist'] += other['pos_hist']
    target['neg_hist'] += other['neg_hist']
    return target


def compute_ap_from_hists(hists):
    """Compute per-class AP and mean AP from accumulated score histograms.

    Returns:
        mean_ap: float (NaN if no valid class)
        ap_per_class: list[float] aligned with hists['fg_classes']
    """
    fg_classes = hists['fg_classes']
    pos_hist = hists['pos_hist']
    neg_hist = hists['neg_hist']

    ap_per_class = []
    for i, _c in enumerate(fg_classes):
        pos = pos_hist[i]
        neg = neg_hist[i]
        total_pos = int(pos.sum())
        if total_pos == 0:
            ap_per_class.append(float('nan'))
            continue
        # Walk thresholds from highest score bin to lowest.
        cum_tp = np.cumsum(pos[::-1]).astype(np.float64)
        cum_fp = np.cumsum(neg[::-1]).astype(np.float64)
        recall = cum_tp / float(total_pos)
        denom = cum_tp + cum_fp
        precision = np.divide(
            cum_tp, denom,
            out=np.zeros_like(cum_tp, dtype=np.float64),
            where=denom > 0,
        )
        recall_prev = np.concatenate(([0.0], recall[:-1]))
        ap = float(np.sum((recall - recall_prev) * precision))
        ap_per_class.append(ap)

    valid_aps = [v for v in ap_per_class if not np.isnan(v)]
    mean_ap = float(np.mean(valid_aps)) if valid_aps else float('nan')
    return mean_ap, ap_per_class


def softmax_scores(processed_pred):
    """Numerically-stable softmax along the last axis (class dim)."""
    x = np.asarray(processed_pred, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    np.exp(x, out=x)
    s = np.sum(x, axis=-1, keepdims=True)
    s[s == 0] = 1.0
    return (x / s).astype(np.float32)
