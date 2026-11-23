"""航位推算 vs 真值：真实 ATE/RPE 示例

从 dataset_norm 序列读取真值位姿；若存在 dvl/ 则用 DVL 速度积分得到
估计轨迹，否则用真值差分加噪声模拟里程计，再与真值对比计算 ATE/RPE。

不启动 HoloOcean。

用法：
    python examples/eval_ate_deadreckon.py <seq_dir>
    python examples/eval_ate_deadreckon.py --demo
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from metrics import ate_rmse, rpe  # noqa: E402


def load_pose(seq_dir: str) -> np.ndarray:
    path = os.path.join(seq_dir, "pose_groundtruth", "data.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 4:
                rows.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
    if not rows:
        raise ValueError(f"empty pose: {path}")
    return np.asarray(rows, dtype=float)  # t_ns, x, y, z


def load_dvl(seq_dir: str) -> np.ndarray | None:
    """返回 (N,4): t_ns, vx, vy, vz；无 DVL 则 None。"""
    path = os.path.join(seq_dir, "dvl", "data.csv")
    if not os.path.isfile(path):
        return None
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 4:
                rows.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
    if not rows:
        return None
    return np.asarray(rows, dtype=float)


def integrate_dvl(pose: np.ndarray, dvl: np.ndarray) -> np.ndarray:
    """用 DVL 速度在位姿时间轴上积分，初值取第一帧真值。"""
    t_ns, xyz = pose[:, 0], pose[:, 1:4]
    est = np.zeros_like(xyz)
    est[0] = xyz[0]
    # 预处理 DVL 时间与速度
    dt_dvl = np.diff(dvl[:, 0]) / 1e9
    dt_dvl = np.clip(dt_dvl, 1e-6, 1.0)
    for i in range(1, len(t_ns)):
        t0, t1 = t_ns[i - 1], t_ns[i]
        dt = (t1 - t0) / 1e9
        if dt <= 0:
            est[i] = est[i - 1]
            continue
        # 取区间内最近 DVL 速度（简化：<= t1 的最后一条）
        mask = dvl[:, 0] <= t1
        if not np.any(mask):
            est[i] = est[i - 1]
            continue
        v = dvl[np.where(mask)[0][-1], 1:4]
        est[i] = est[i - 1] + v * dt
    return est


def integrate_noisy_odometry(pose: np.ndarray, noise_std: float, seed: int) -> np.ndarray:
    """无 DVL 时：真值差分当速度，加噪声积分（演示用）。"""
    rng = np.random.default_rng(seed)
    t_ns, xyz = pose[:, 0], pose[:, 1:4]
    est = np.zeros_like(xyz)
    est[0] = xyz[0]
    for i in range(1, len(t_ns)):
        dt = (t_ns[i] - t_ns[i - 1]) / 1e9
        if dt <= 0:
            est[i] = est[i - 1]
            continue
        v_true = (xyz[i] - xyz[i - 1]) / dt
        v = v_true + rng.normal(0.0, noise_std, size=3)
        est[i] = est[i - 1] + v * dt
    return est


def as_list(xyz: np.ndarray) -> list[tuple[float, float, float]]:
    return [(float(a), float(b), float(c)) for a, b, c in xyz]


def main() -> int:
    ap = argparse.ArgumentParser(description="Dead-reckoning ATE vs ground truth")
    ap.add_argument("sequence", nargs="?", help="dataset_norm 序列目录")
    ap.add_argument("--demo", action="store_true", help="合成轨迹演示")
    ap.add_argument("--noise", type=float, default=0.05, help="无 DVL 时速度噪声 std")
    ap.add_argument("--align", action="store_true", help="ATE 平移对齐")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.demo or not args.sequence:
        t = np.linspace(0, 20, 400)
        xyz = np.column_stack([t, 0.3 * np.sin(0.5 * t), -5 * np.ones_like(t)])
        pose = np.column_stack([t * 1e9, xyz])
        print("demo: synthetic trajectory with noisy odometry")
        mode = "noisy_odom"
        est = integrate_noisy_odometry(pose, args.noise, args.seed)
    else:
        seq = os.path.abspath(args.sequence)
        pose = load_pose(seq)
        print(f"sequence: {seq}  poses={len(pose)}")
        dvl = load_dvl(seq)
        if dvl is not None and len(dvl) > 1:
            print(f"dvl rows: {len(dvl)}")
            est = integrate_dvl(pose, dvl)
            mode = "dvl"
        else:
            print(f"no dvl, using noisy odometry (noise={args.noise})")
            est = integrate_noisy_odometry(pose, args.noise, args.seed)
            mode = "noisy_odom"

    gt_xyz = pose[:, 1:4]
    gt_l = as_list(gt_xyz)
    est_l = as_list(est)
    ate = ate_rmse(gt_l, est_l, align=args.align)
    r1 = rpe(gt_l, est_l, delta=1)
    r10 = rpe(gt_l, est_l, delta=min(10, len(gt_l) - 1))
    print(f"mode: {mode}")
    print(f"ATE RMSE = {ate:.4f} m" + (" (aligned)" if args.align else ""))
    print(f"RPE delta=1  RMSE = {r1:.4f} m")
    print(f"RPE delta=10 RMSE = {r10:.4f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
