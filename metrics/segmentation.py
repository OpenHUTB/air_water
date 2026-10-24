"""语义分割评测：IoU / mIoU / Pixel Accuracy。"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def confusion_matrix(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """返回 C x C 混淆矩阵，cm[i, j] = 真实为 i、预测为 j 的像素数。"""
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape} gt={gt.shape}")
    p = pred.astype(np.int64).ravel()
    g = gt.astype(np.int64).ravel()
    if p.min() < 0 or g.min() < 0 or p.max() >= num_classes or g.max() >= num_classes:
        raise ValueError("label out of range")
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (g, p), 1)
    return cm


def iou(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    num_classes: int | None = None,
    ignore_index: int | None = None,
    class_id: int,
) -> float:
    """单类 IoU = TP / (TP + FP + FN)。"""
    p = pred
    g = gt
    if ignore_index is not None:
        mask = g != ignore_index
        p = p[mask]
        g = g[mask]
    tp = int(np.sum((g == class_id) & (p == class_id)))
    fp = int(np.sum((g != class_id) & (p == class_id)))
    fn = int(np.sum((g == class_id) & (p != class_id)))
    denom = tp + fp + fn
    if denom == 0:
        return float("nan")
    return tp / denom


def miou(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    classes: Sequence[int],
    ignore_index: int | None = None,
) -> float:
    """各类 IoU 的平均（忽略 NaN 类）。"""
    vals = [
        iou(pred, gt, ignore_index=ignore_index, class_id=c) for c in classes
    ]
    vals = [v for v in vals if v == v]  # drop NaN
    if not vals:
        raise ValueError("no valid class IoU")
    return float(sum(vals) / len(vals))


def pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    """预测像素等于真值的比例。"""
    if pred.shape != gt.shape:
        raise ValueError("shape mismatch")
    total = gt.size
    if total == 0:
        raise ValueError("empty")
    return float(np.sum(pred == gt) / total)
