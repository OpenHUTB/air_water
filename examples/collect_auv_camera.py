"""AUV 采集 + RGB / 深度相机示例

在自定义场景中为悬停 AUV 挂载 RGBCamera 与 DepthCamera，同步落盘：
位姿、IMU、RGB 图、深度图（16-bit PNG，单位 mm）。

依赖：holoocean、numpy、pillow

用法（仓库根目录）：
    python examples/collect_auv_camera.py --headless --seconds 4
    python examples/collect_auv_camera.py --list

产物：
    examples/out/AirSea-UV/SimpleUnderwater/auv0/track0/
        settings.json
        pose_groundtruth/
        imu0/
        rgb/cam0/{sensor.yaml, data.csv, data/*.png}
        depth/cam0/{sensor.yaml, data.csv, data/*.png}
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
RGB_HZ = 5
# DepthCamera 实测每 tick 都出帧；我们按 RGB_HZ 降采样后再落盘
DEPTH_SAVE_EVERY = TICKS_PER_SEC // RGB_HZ  # 6
CAM_SIZE = 256

SCENARIO = {
    "name": "collect_auv_camera",
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
                    "sensor_type": "RGBCamera",
                    "socket": "CameraLeftSocket",
                    "Hz": RGB_HZ,
                    "configuration": {
                        "CaptureWidth": CAM_SIZE,
                        "CaptureHeight": CAM_SIZE,
                    },
                },
                {
                    "sensor_type": "DepthCamera",
                    "socket": "CameraLeftSocket",
                    "Hz": RGB_HZ,
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


def save_rgb_png(img: np.ndarray, path: str) -> None:
    """RGBA/RGB uint8 -> PNG（按 dtype 自动推断模式，兼容 Pillow 12/13）。"""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[:, :, :3].astype(np.uint8)
    else:
        raise ValueError(f"unexpected RGB shape {arr.shape}")
    Image.fromarray(rgb).save(path)


def save_depth_png(depth_m: np.ndarray, path: str) -> None:
    """深度（米）-> 16-bit PNG，像素值 = 毫米（0 表示无效/过远）。

    不传 mode=，由 Pillow 按 uint16 推断为 16 位灰度（兼容 Pillow 13）。
    """
    d = np.asarray(depth_m, dtype=np.float32)
    mm = np.clip(np.round(d * 1000.0), 0, 65535).astype(np.uint16)
    mm[~np.isfinite(d)] = 0
    Image.fromarray(mm).save(path)


def collect(out_root: str, seconds: float, show: bool) -> str:
    import holoocean

    n_steps = max(1, int(seconds * TICKS_PER_SEC))
    settings = Settings(
        world=WORLD,
        package="Ocean",
        ticks_per_sec=TICKS_PER_SEC,
        frames_per_sec=RGB_HZ,
        agents=("auv0",),
        random_seed=0,
        holoocean_version=getattr(holoocean, "__version__", "unknown"),
        notes="AUV cruise with RGB+Depth cameras via dataset_norm",
    )
    seq_root = os.path.join(out_root, "AirSea-UV")
    print(f"scene={SCENARIO['name']}  steps={n_steps}  out={seq_root}")

    last_t_ns = {"pose": -1, "imu": -1, "rgb": -1, "depth": -1}
    start_xyz: np.ndarray | None = None
    n_rgb = n_depth = 0

    with holoocean.make(scenario_cfg=SCENARIO, show_viewport=show) as env:
        with SequenceWriter(
            root=seq_root,
            world=settings.world,
            agent="auv0",
            track="track0",
            settings=settings,
        ) as w:
            w.register_pose("pose_groundtruth", rate_hz=float(TICKS_PER_SEC), frame_id="world")
            w.register_imu("imu0", rate_hz=float(TICKS_PER_SEC), frame_id="auv0/imu")
            w.register_camera(
                "rgb", "rgb", rate_hz=float(RGB_HZ), frame_id="auv0/cam_left",
                notes="RGBA converted to RGB PNG",
            )
            w.register_camera(
                "depth", "depth", rate_hz=float(RGB_HZ), frame_id="auv0/cam_left",
                notes="depth PNG 16-bit, unit mm",
            )

            t0 = time.time()
            for k in range(n_steps):
                cmd = FORWARD if k < int(0.8 * n_steps) else IDLE
                state = env.step(cmd)
                t_ns = t_to_ns(state["t"])

                pose = state["PoseSensor"]
                p = pose[:3, 3]
                if start_xyz is None:
                    start_xyz = p.copy()
                q = rot_to_quat_wxyz(pose[:3, :3])
                if t_ns > last_t_ns["pose"]:
                    w.add_pose("pose_groundtruth", t_ns, (p[0], p[1], p[2]), q)
                    last_t_ns["pose"] = t_ns

                imu = state["IMUSensor"]
                acc = np.asarray(imu[0], dtype=float).reshape(3)
                gyr = np.asarray(imu[1], dtype=float).reshape(3)
                if t_ns > last_t_ns["imu"]:
                    w.add_imu(
                        "imu0", t_ns,
                        (gyr[0], gyr[1], gyr[2]),
                        (acc[0], acc[1], acc[2]),
                    )
                    last_t_ns["imu"] = t_ns

                # RGB：仅在相机帧到达时写
                if "RGBCamera" in state and state["RGBCamera"] is not None:
                    img = np.asarray(state["RGBCamera"])
                    if t_ns > last_t_ns["rgb"]:
                        fn = f"{t_ns:019d}.png"
                        save_rgb_png(img, w.camera_image_path("rgb", fn))
                        w.add_camera_frame("rgb", t_ns, fn)
                        last_t_ns["rgb"] = t_ns
                        n_rgb += 1

                # DepthCamera 每 tick 可能出帧：按 DEPTH_SAVE_EVERY 降采样
                if (
                    "DepthCamera" in state
                    and state["DepthCamera"] is not None
                    and k % DEPTH_SAVE_EVERY == 0
                    and t_ns > last_t_ns["depth"]
                ):
                    dcam = state["DepthCamera"]
                    # 常见格式: dict(image=..., depth=...) 或直接 depth 数组
                    if isinstance(dcam, dict):
                        depth_m = dcam.get("depth", dcam.get("image"))
                    else:
                        depth_m = dcam
                    depth_m = np.asarray(depth_m, dtype=np.float32)
                    if depth_m.ndim == 3:
                        depth_m = depth_m[..., 0]
                    fn = f"{t_ns:019d}.png"
                    save_depth_png(depth_m, w.camera_image_path("depth", fn))
                    w.add_camera_frame("depth", t_ns, fn)
                    last_t_ns["depth"] = t_ns
                    n_depth += 1

                if k % max(1, n_steps // 5) == 0:
                    dist = (
                        float(np.linalg.norm(p - start_xyz))
                        if start_xyz is not None
                        else float("nan")
                    )
                    print(
                        f"  step {k:4d}/{n_steps}  t={state['t']:6.2f}s  "
                        f"xyz=[{p[0]:7.2f} {p[1]:7.2f} {p[2]:7.2f}]  "
                        f"|p-p0|={dist:5.2f}  rgb={n_rgb} depth={n_depth}"
                    )

            print(f"  done in {time.time() - t0:.1f}s real time")
            print(f"  stats: {w.stats()}")

    return os.path.join(seq_root, settings.world, "auv0", "track0")


def main() -> int:
    ap = argparse.ArgumentParser(description="AUV collect with RGB+Depth cameras")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=4.0)
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
