"""目标检测评测：Precision / Recall / F1（混淆计数）。

不依赖框 IoU 的版本：调用方先决定每条预测是否命中
（例如按 IoU 阈值或类别+位置匹配），再统计 TP/FP/FN。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Counts:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def detection_counts(
    matched: Sequence[bool],
    n_groundtruth: int,
) -> Counts:
    """matched：每条预测是否命中（True=TP，False=FP）；n_groundtruth：真值数。"""
    tp = sum(1 for m in matched if m)
    fp = sum(1 for m in matched if not m)
    fn = max(0, n_groundtruth - tp)
    return Counts(tp=tp, fp=fp, fn=fn)


def precision_recall_f1(
    matched: Sequence[bool],
    n_groundtruth: int,
) -> dict[str, float]:
    c = detection_counts(matched, n_groundtruth)
    return {
        "precision": c.precision,
        "recall": c.recall,
        "f1": c.f1,
        "tp": float(c.tp),
        "fp": float(c.fp),
        "fn": float(c.fn),
    }
