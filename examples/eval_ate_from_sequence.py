"""读取已采集序列，计算轨迹 ATE / RPE。

不启动 HoloOcean。真值来自 dataset_norm 目录的 pose_groundtruth/data.csv；
估计轨迹需用 --est 指向另一序列（或 --selfcheck 做固定偏移自检）。

用法：
    python examples/eval_ate_from_sequence.py <gt_seq> --est <est_seq>
    python examples/eval_ate_from_sequence.py <gt_seq> --selfcheck
    python examples/eval_ate_from_sequence.py --demo
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from metrics import ate_rmse, rpe  # noqa: E402


def load_pose_xyz(seq_dir: str) -> np.ndarray:
    """读 pose_groundtruth/data.csv，返回 (N, 3) 位置。"""
    path = os.path.join(seq_dir, "pose_groundtruth", "data.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing {path}")
    xs, ys, zs = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError(f"empty csv: {path}")
        for row in reader:
            if len(row) < 4:
                continue
            xs.append(float(row[1]))
            ys.append(float(row[2]))
            zs.append(float(row[3]))
    if not xs:
        raise ValueError(f"no pose rows: {path}")
    return np.column_stack([xs, ys, zs])


def as_list(xyz: np.ndarray) -> list[tuple[float, float, float]]:
    return [(float(a), float(b), float(c)) for a, b, c in xyz]


def main() -> int:
    ap = argparse.ArgumentParser(description="ATE/RPE on a dataset_norm sequence")
    ap.add_argument("sequence", nargs="?", help="真值序列目录（含 pose_groundtruth/）")
    ap.add_argument(
        "--est",
        default=None,
        help="估计轨迹序列目录（同样含 pose_groundtruth/data.csv）",
    )
    ap.add_argument(
        "--selfcheck",
        action="store_true",
        help="真值加固定偏移做自检，不读取估计轨迹",
    )
    ap.add_argument("--demo", action="store_true", help="用合成轨迹演示")
    ap.add_argument("--align", action="store_true", help="ATE 先做平移对齐")
    args = ap.parse_args()

    if args.demo or not args.sequence:
        t = np.linspace(0, 10, 200)
        gt = np.column_stack([t, np.zeros_like(t), -2 * np.ones_like(t)])
        rng = np.random.default_rng(0)
        est = gt + np.array([0.15, 0.02, 0.0]) + rng.normal(0, 0.01, gt.shape)
        print("demo mode: synthetic straight line, est shifted +0.15 m in x")
    else:
        seq = os.path.abspath(args.sequence)
        gt = load_pose_xyz(seq)
        print(f"gt sequence: {seq}")
        print(f"poses: {len(gt)}")

        if args.est:
            est_seq = os.path.abspath(args.est)
            est = load_pose_xyz(est_seq)
            if len(est) != len(gt):
                print(
                    f"warning: gt={len(gt)} est={len(est)} rows, "
                    "comparing prefix only"
                )
                n = min(len(gt), len(est))
                gt, est = gt[:n], est[:n]
            print(f"est sequence: {est_seq}")
        elif args.selfcheck:
            # 仅自检：固定偏移，不读估计文件
            est = gt + np.array([0.10, 0.0, 0.0])
            print("selfcheck mode: est = gt + [0.1, 0, 0] (not a real estimate)")
        else:
            print("error: provide --est <est_seq> or --selfcheck")
            return 2

    gt_l, est_l = as_list(gt), as_list(est)
    ate = ate_rmse(gt_l, est_l, align=args.align)
    rpe1 = rpe(gt_l, est_l, delta=1)
    rpe10 = rpe(gt_l, est_l, delta=10)
    print(f"ATE RMSE = {ate:.4f} m" + (" (aligned)" if args.align else ""))
    print(f"RPE delta=1  RMSE = {rpe1:.4f} m")
    print(f"RPE delta=10 RMSE = {rpe10:.4f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
