"""metrics 自测：合成数据验证 ATE/RPE、IoU、检测指标。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from metrics import ate_rmse, iou, miou, pixel_accuracy, precision_recall_f1, rpe  # noqa: E402


def test_trajectory() -> None:
    gt = [(float(i), 0.0, 0.0) for i in range(10)]
    # 估计沿 x 偏移 0.1
    est = [(float(i) + 0.1, 0.0, 0.0) for i in range(10)]
    ate = ate_rmse(gt, est)
    assert abs(ate - 0.1) < 1e-9, ate
    # 对齐后 ATE 应接近 0
    ate_al = ate_rmse(gt, est, align=True)
    assert ate_al < 1e-9, ate_al
    # RPE：相对位移相同 → 0
    r = rpe(gt, est, delta=1)
    assert abs(r) < 1e-9, r
    # 整体平移不影响相对位移
    gt2 = [(float(i) + 5.0, 1.0, 2.0) for i in range(10)]
    est2 = [(float(i) + 5.1, 1.0, 2.0) for i in range(10)]
    assert rpe(gt2, est2, delta=1) < 1e-9


def test_segmentation() -> None:
    gt = np.zeros((4, 4), dtype=np.int64)
    gt[0:2, :] = 1
    pred = gt.copy()
    assert abs(pixel_accuracy(pred, gt) - 1.0) < 1e-12
    assert abs(iou(pred, gt, class_id=1) - 1.0) < 1e-12

    # 改 1 个像素：类 1 TP=7, FN=1（若 8 个类1像素）
    pred = gt.copy()
    pred[0, 0] = 0
    v1 = iou(pred, gt, class_id=1)
    assert 0.0 < v1 < 1.0
    m = miou(pred, gt, classes=[0, 1])
    assert 0.0 < m <= 1.0

    # 形状不匹配应报错
    try:
        pixel_accuracy(pred, gt[:2, :2])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_detection() -> None:
    # 4 个预测，3 命中；5 个真值 → TP=3 FP=1 FN=2
    out = precision_recall_f1([True, True, False, True], n_groundtruth=5)
    assert out["tp"] == 3 and out["fp"] == 1 and out["fn"] == 2
    assert abs(out["precision"] - 0.75) < 1e-12
    assert abs(out["recall"] - 0.6) < 1e-12
    assert abs(out["f1"] - 2 * 0.75 * 0.6 / 1.35) < 1e-12


def main() -> int:
    test_trajectory()
    test_segmentation()
    test_detection()
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
