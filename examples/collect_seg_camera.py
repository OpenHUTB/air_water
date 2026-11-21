"""AUV 采集 + 语义分割相机示例

自定义场景为悬停 AUV 挂载 SemanticSegmentationCamera，
位姿、IMU 与分割掩码按 dataset_norm 落盘。

掩码保存说明：
- 若输出为 (H,W) 整数标签 → 单通道 PNG（像素值=类别 id）
- 若输出为 (H,W,3) 彩色可视化 → RGB PNG（便于预览；类别表见 settings 注释）

依赖：holoocean、numpy、pillow

用法（仓库根目录）：
    python examples/collect_seg_camera.py --headless --seconds 3

产物：
    examples/out/AirSea-UV/SimpleUnderwater/auv0/track_seg/
        pose_groundtruth/  imu0/
        segmentation/cam0/{sensor.yaml, data.csv, data/*.png}
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dataset_norm import SequenceWriter, Settings  # noqa: E402

FORWARD = np.array([0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 6.0, 6.0])
IDLE = np.zeros(8)

WORLD = "SimpleUnderwater"
TICKS_PER_SEC = 30
SEG_HZ = 5
SAVE_EVERY = TICKS_PER_SEC // SEG_HZ  # 6
CAM_SIZE = 256

SCENARIO = {
    "name": "collect_seg_camera",
    "world": WORLD,
    "package_name": "Ocean",
    "main_agent": "auv0",
    "ticks_per_sec": TICKS_PER_SEC,
    "agents": [
        {
            "agent_name": "auv0",
            "agent_type": "HoveringAUV",
            "control_scheme": 0,
            "location": [0.0, 0.0, -5.0],
            "rotation": [0.0, 0.0, 0.0],
            "sensors": [
                {"sensor_type": "PoseSensor"},
                {
                    "sensor_type": "IMUSensor",
                    "socket": "IMUSocket",
                    "Hz": TICKS_PER_SEC,
                    "configuration": {
                        "AccelSigma": 0.00277,
                        "AngVelSigma": 0.00123,
                        "AccelBiasSigma": 0.00141,
                        "AngVelBiasSigma": 0.00388,
                        "ReturnBias": True,
                    },
                },
                {
                    "sensor_type": "SemanticSegmentationCamera",
                    "socket": "CameraLeftSocket",
                    "Hz": SEG_HZ,
                    "configuration": {
                        "CaptureWidth": CAM_SIZE,
                        "CaptureHeight": CAM_SIZE,
                    },
                },
            ],
        }
    ],
}


def rot_to_quat_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    m = np.asarray(R, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x = 0.25 * s, (m[2, 1] - m[1, 2]) / s
        y, z = (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x = (m[2, 1] - m[1, 2]) / s, 0.25 * s
        y, z = (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] >= m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s
        y, z = 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s
        y, z = (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def t_to_ns(t_sec: float) -> int:
    return int(round(float(t_sec) * 1e9))


def save_mask(arr: np.ndarray, path: str) -> str:
    """按 dtype 保存：整数标签 → 单通道；RGB → 彩色。"""
    a = np.asarray(arr)
    if a.ndim == 2:
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        Image.fromarray(a).save(path)
        return "L"
    if a.ndim == 3 and a.shape[2] >= 3:
        rgb = a[:, :, :3].astype(np.uint8)
        Image.fromarray(rgb).save(path)
        return "RGB"
    raise ValueError(f"unexpected seg shape {a.shape} dtype={a.dtype}")


def collect(out_root: str, seconds: float, show: bool) -> str:
    import holoocean

    n_steps = max(1, int(seconds * TICKS_PER_SEC))
    settings = Settings(
        world=WORLD,
        package="Ocean",
        ticks_per_sec=TICKS_PER_SEC,
        frames_per_sec=float(SEG_HZ),
        agents=("auv0",),
        random_seed=0,
        holoocean_version=getattr(holoocean, "__version__", "unknown"),
        notes="AUV cruise with SemanticSegmentationCamera via dataset_norm",
    )
    seq_root = os.path.join(out_root, "AirSea-UV")
    print(f"scene={SCENARIO['name']}  steps={n_steps}  out={seq_root}")

    last: dict[str, int] = {}

    def bump(key: str, t_ns: int) -> bool:
        if t_ns > last.get(key, -1):
            last[key] = t_ns
            return True
        return False

    n_seg = 0
    mask_mode = ""

    with holoocean.make(scenario_cfg=SCENARIO, show_viewport=show) as env:
        with SequenceWriter(
            root=seq_root, world=WORLD, agent="auv0",
            track="track_seg", settings=settings,
        ) as w:
            w.register_pose("pose_groundtruth", rate_hz=float(TICKS_PER_SEC), frame_id="world")
            w.register_imu("imu0", rate_hz=float(TICKS_PER_SEC), frame_id="auv0/imu")
            w.register_camera(
                "segmentation", "segmentation", rate_hz=float(SEG_HZ),
                frame_id="auv0/cam_left",
                notes="semantic mask; L=index PNG or RGB=colored",
            )

            t0 = time.time()
            for k in range(n_steps):
                env.act("auv0", FORWARD if k < int(0.8 * n_steps) else IDLE)
                state = env.tick()
                t_ns = t_to_ns(state["t"])

                pose = state.get("PoseSensor")
                if pose is not None and bump("pose", t_ns):
                    p = pose[:3, 3]
                    q = rot_to_quat_wxyz(pose[:3, :3])
                    w.add_pose("pose_groundtruth", t_ns, (p[0], p[1], p[2]), q)

                imu = state.get("IMUSensor")
                if imu is not None and bump("imu", t_ns):
                    acc = np.asarray(imu[0], dtype=float).reshape(3)
                    gyr = np.asarray(imu[1], dtype=float).reshape(3)
                    w.add_imu(
                        "imu0", t_ns,
                        (gyr[0], gyr[1], gyr[2]),
                        (acc[0], acc[1], acc[2]),
                    )

                seg_key = None
                for key in state:
                    if "Semantic" in key or "Segmentation" in key:
                        seg_key = key
                        break
                seg = state.get(seg_key) if seg_key else None
                if seg is not None and k % SAVE_EVERY == 0 and t_ns > last.get("seg", -1):
                    last["seg"] = t_ns
                    fn = f"{t_ns:019d}.png"
                    mask_mode = save_mask(seg, w.camera_image_path("segmentation", fn))
                    w.add_camera_frame("segmentation", t_ns, fn)
                    n_seg += 1

                if k % max(1, n_steps // 5) == 0:
                    p = pose[:3, 3] if pose is not None else np.zeros(3)
                    print(
                        f"  step {k:4d}/{n_steps}  t={state['t']:6.2f}s  "
                        f"z={p[2]:7.2f}  seg={n_seg} mode={mask_mode or '-'}"
                    )

            print(f"  done in {time.time() - t0:.1f}s  seg_frames={n_seg} mode={mask_mode}")
            print(f"  stats: {w.stats()}")

    return os.path.join(seq_root, settings.world, "auv0", "track_seg")


def main() -> int:
    ap = argparse.ArgumentParser(description="AUV semantic segmentation camera collect")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"),
    )
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        try:
            import holoocean

            print("holoocean OK", getattr(holoocean, "__version__", ""))
        except Exception as e:
            print("holoocean import failed:", e)
            return 1
        return 0

    try:
        import holoocean  # noqa: F401
    except Exception as e:
        print("需要 holoocean / numpy / pillow:", e)
        return 1

    seq_dir = collect(args.out, seconds=args.seconds, show=not args.headless)
    from dataset_norm.validate import main as validate_main

    print()
    print("validate:", seq_dir)
    return validate_main([seq_dir])


if __name__ == "__main__":
    sys.exit(main())
