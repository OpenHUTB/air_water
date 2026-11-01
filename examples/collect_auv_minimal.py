"""最小 AUV 采集脚本

在 HoloOcean 中驱动悬停 AUV 做短时定深巡航，把真值位姿与 IMU
按 dataset_norm 规范落盘，并自动跑一遍校验。

依赖：
    pip install holoocean numpy
    本仓库根目录需能 import dataset_norm（脚本会自动把仓库根加入 sys.path）

用法：
    python examples/collect_auv_minimal.py --headless --seconds 5
    python examples/collect_auv_minimal.py --list   # 只检查环境

产出：
    out/SimpleUnderwater/auv0/track0/
        settings.json
        pose_groundtruth/{sensor.yaml, data.csv}
        imu0/{sensor.yaml, data.csv}
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

# 允许直接运行本文件时找到同仓库的 dataset_norm
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dataset_norm import SequenceWriter, Settings  # noqa: E402

# HoveringAUV：后 4 个推进器负责水平推力
FORWARD = np.array([0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 6.0, 6.0])
IDLE = np.zeros(8)

SCENE = "SimpleUnderwater-Hovering"
TICKS_PER_SEC = 30


def rot_to_quat_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 旋转矩阵 -> 四元数 (w, x, y, z)。"""
    m = np.asarray(R, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] >= m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def t_to_ns(t_sec: float) -> int:
    return int(round(float(t_sec) * 1e9))


def collect(out_root: str, seconds: float, show: bool) -> str:
    import holoocean

    n_steps = max(1, int(seconds * TICKS_PER_SEC))
    settings = Settings(
        world="SimpleUnderwater",
        package="Ocean",
        ticks_per_sec=TICKS_PER_SEC,
        frames_per_sec=TICKS_PER_SEC,
        agents=("auv0",),
        random_seed=0,
        holoocean_version=getattr(holoocean, "__version__", "unknown"),
        notes="minimal AUV cruise + pose/imu via dataset_norm",
    )
    seq_root = os.path.join(out_root, "AirSea-UV")
    print(f"scene={SCENE}  steps={n_steps}  out={seq_root}")

    last_t_ns = {"pose": -1, "imu": -1}

    with holoocean.make(SCENE, show_viewport=show) as env:
        with SequenceWriter(
            root=seq_root,
            world=settings.world,
            agent="auv0",
            track="track0",
            settings=settings,
        ) as w:
            w.register_pose(name="pose_groundtruth", rate_hz=float(TICKS_PER_SEC), frame_id="world")
            w.register_imu(name="imu0", rate_hz=float(TICKS_PER_SEC), frame_id="auv0/imu")

            t0 = time.time()
            state = None
            for k in range(n_steps):
                cmd = FORWARD if k < int(0.8 * n_steps) else IDLE
                state = env.step(cmd)
                t_ns = t_to_ns(state["t"])

                # 位姿真值
                pose = state["PoseSensor"]  # 4x4
                p = pose[:3, 3]
                q = rot_to_quat_wxyz(pose[:3, :3])
                if t_ns > last_t_ns["pose"]:
                    w.add_pose("pose_groundtruth", t_ns, (p[0], p[1], p[2]), q)
                    last_t_ns["pose"] = t_ns

                # IMU: HoloOcean 一般为 [acc(3), gyr(3), ...]
                imu = state["IMUSensor"]
                acc = np.asarray(imu[0], dtype=float).reshape(3)
                gyr = np.asarray(imu[1], dtype=float).reshape(3)
                if t_ns > last_t_ns["imu"]:
                    # writer 约定：gyro 在前，acc 在后（与 IMU_FIELDS 一致）
                    w.add_imu("imu0", t_ns, (gyr[0], gyr[1], gyr[2]), (acc[0], acc[1], acc[2]))
                    last_t_ns["imu"] = t_ns

                if k % max(1, n_steps // 5) == 0 and state is not None:
                    z = float(p[2])
                    print(f"  step {k:4d}/{n_steps}  t={state['t']:6.2f}s  z={z:7.2f} m")

            print(f"  done in {time.time() - t0:.1f}s real time")
            print(f"  stats: {w.stats()}")

    return os.path.join(seq_root, settings.world, "auv0", "track0")


def main() -> int:
    ap = argparse.ArgumentParser(description="Minimal AUV collect (HoloOcean -> dataset_norm)")
    ap.add_argument("--headless", action="store_true", help="不显示仿真窗口")
    ap.add_argument("--seconds", type=float, default=5.0, help="仿真时长（秒）")
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"),
        help="输出根目录",
    )
    ap.add_argument("--list", action="store_true", help="仅检查 holoocean 是否可导入")
    args = ap.parse_args()

    if args.list:
        try:
            import holoocean
            print("holoocean OK", getattr(holoocean, "__version__", ""))
            print("path", getattr(holoocean, "__file__", ""))
        except Exception as e:
            print("holoocean import failed:", e)
            return 1
        return 0

    try:
        import holoocean  # noqa: F401
    except Exception as e:
        print("需要先安装 HoloOcean 与 numpy：", e)
        return 1

    seq_dir = collect(args.out, seconds=args.seconds, show=not args.headless)

    # 自动校验
    from dataset_norm.validate import main as validate_main

    print()
    print("validate:", seq_dir)
    return validate_main([seq_dir])


if __name__ == "__main__":
    sys.exit(main())
