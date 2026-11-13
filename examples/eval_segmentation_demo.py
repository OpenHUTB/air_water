"""分割精度示例：IoU / mIoU / 像素准确率

不依赖 HoloOcean。可用合成掩码演示，或读取目录下的
pred.npy / gt.npy（形状相同，整型标签）。

用法：
    python examples/eval_segmentation_demo.py --demo
    python examples/eval_segmentation_demo.py --dir path/to/masks
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from metrics import iou, miou, pixel_accuracy  # noqa: E402


def make_demo_masks() -> tuple[np.ndarray, np.ndarray, list[int]]:
    """2 类合成掩码：gt 为分块，pred 故意错一块。"""
    gt = np.zeros((64, 64), dtype=np.int64)
    gt[:, 32:] = 1
    pred = gt.copy()
    pred[0:16, 32:48] = 0  # 第 1 类右上角漏检
    return pred, gt, [0, 1]


def load_masks(dir_path: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    pred_p = os.path.join(dir_path, "pred.npy")
    gt_p = os.path.join(dir_path, "gt.npy")
    if not (os.path.isfile(pred_p) and os.path.isfile(gt_p)):
        raise FileNotFoundError(f"need pred.npy and gt.npy in {dir_path}")
    pred = np.load(pred_p)
    gt = np.load(gt_p)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch pred={pred.shape} gt={gt.shape}")
    classes = sorted(set(int(x) for x in np.unique(gt)) | set(int(x) for x in np.unique(pred)))
    return pred, gt, classes


def main() -> int:
    ap = argparse.ArgumentParser(description="Segmentation IoU/mIoU demo")
    ap.add_argument("--demo", action="store_true", help="合成掩码演示")
    ap.add_argument("--dir", default=None, help="含 pred.npy 与 gt.npy 的目录")
    ap.add_argument("--class-id", type=int, default=None, help="仅打印单类 IoU")
    args = ap.parse_args()

    if args.dir:
        pred, gt, classes = load_masks(args.dir)
        print(f"loaded: {args.dir}  shape={gt.shape}  classes={classes}")
    elif args.demo or not args.dir:
        pred, gt, classes = make_demo_masks()
        print("demo mode: synthetic 64x64 binary masks")
    else:
        print("error: provide --demo or --dir")
        return 2

    acc = pixel_accuracy(pred, gt)
    m = miou(pred, gt, classes=classes)
    print(f"pixel_accuracy = {acc:.4f}")
    print(f"mIoU           = {m:.4f}")
    for c in classes:
        print(f"IoU[{c}]        = {iou(pred, gt, class_id=c):.4f}")
    if args.class_id is not None:
        print(f"requested IoU[{args.class_id}] = {iou(pred, gt, class_id=args.class_id):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
