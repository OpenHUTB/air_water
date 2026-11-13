"""AUV 导航套件采集：位姿 + IMU + DVL + 深度

定位/航位推算常用输入一次落盘。DVL 用 15 Hz（整除 ticks=30），
深度计 30 Hz；DVL/深度写入 dataset_norm 通用时序目录。

依赖：holoocean、numpy

用法（仓库根目录）：
    python examples/collect_auv_nav.py --headless --seconds 4

产物：
    examples/out/AirSea-UV/SimpleUnderwater/auv0/track_nav/
        pose_groundtruth/  imu0/  dvl/  depth/
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dataset_norm import SequenceWriter, Settings  # noqa: E402

WORLD = "SimpleUnderwater"
TICKS_PER_SEC = 30
DVL_HZ = 15  # 必须整除 30，否则引擎会警告并改采样
DEPTH_HZ = TICKS_PER_SEC

FORWARD = np.array([0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 6.0, 6.0])
IDLE = np.zeros(8)

SCENARIO = {
    "name": "collect_auv_nav",
    "world": WORLD,
    "package_name": "Ocean",
    "main_agent": "auv0",
    "ticks_per_sec": TICKS_PER_SEC,
    "agents": [
        {
            "agent_name": "auv0",
            "agent_type": "HoveringAUV",
            "control_scheme": 0,
            "location": [0.0, 0.0, -10.0],
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
                    "sensor_type": "DVLSensor",
                    "socket": "DVLSocket",
                    "Hz": DVL_HZ,
                    "configuration": {
                        "VelSigma": 0.02626,
                        "ReturnRange": True,
                        "MaxRange": 50,
                        "RangeSigma": 0.1,
                    },
                },
                {
                    "sensor_type": "DepthSensor",
                    "socket": "DepthSocket",
                    "Hz": DEPTH_HZ,
                    "configuration": {"Sigma": 0.255},
                },
            ],
        }
    ],
}

DVL_HEADER = [
    "#timestamp [ns]",
    "vx [m s^-1]",
    "vy [m s^-1]",
    "vz [m s^-1]",
    "range_fwd_x [m]",
    "range_fwd_y [m]",
    "range_back_x [m]",
    "range_back_y [m]",
]
DEPTH_HEADER = ["#timestamp [ns]", "depth [m]"]


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


def write_meta(sensor_dir: str, sensor_type: str, rate_hz: float, frame_id: str, notes: str = "") -> None:
    os.makedirs(sensor_dir, exist_ok=True)
    with open(os.path.join(sensor_dir, "sensor.yaml"), "w", encoding="utf-8") as f:
        f.write(
            f"sensor_type: {sensor_type}\n"
            f"rate_hz: {rate_hz:g}\n"
            f"frame_id: {frame_id}\n"
            f"topic: {sensor_type}\n"
        )
        if notes:
            f.write(f'notes: "{notes}"\n')


def write_csv_header(path: str, header: list[str]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.isfile(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)
    return path


def collect(out_root: str, seconds: float, show: bool) -> str:
    import holoocean

    n_steps = max(1, int(seconds * TICKS_PER_SEC))
    settings = Settings(
        world=WORLD,
        package="Ocean",
        ticks_per_sec=TICKS_PER_SEC,
        frames_per_sec=DVL_HZ,
        agents=("auv0",),
        random_seed=0,
        holoocean_version=getattr(holoocean, "__version__", "unknown"),
        notes="AUV nav suite: pose/imu/dvl/depth",
    )
    seq_root = os.path.join(out_root, "AirSea-UV")
    base = os.path.join(seq_root, WORLD, "auv0", "track_nav")
    print(f"scene={SCENARIO['name']}  steps={n_steps}  out={seq_root}")

    write_meta(os.path.join(base, "dvl"), "dvl", float(DVL_HZ), "auv0/dvl")
    write_meta(os.path.join(base, "depth"), "depth_sensor", float(DEPTH_HZ), "auv0/depth",
               notes="reading sign matches z (negative underwater)")
    dvl_csv = write_csv_header(os.path.join(base, "dvl", "data.csv"), DVL_HEADER)
    depth_csv = write_csv_header(os.path.join(base, "depth", "data.csv"), DEPTH_HEADER)

    last: dict[str, int] = {}

    def bump(key: str, t_ns: int) -> bool:
        if t_ns > last.get(key, -1):
            last[key] = t_ns
            return True
        return False

    n_dvl = n_depth = 0
    with holoocean.make(scenario_cfg=SCENARIO, show_viewport=show) as env:
        with SequenceWriter(
            root=seq_root, world=WORLD, agent="auv0",
            track="track_nav", settings=settings,
        ) as w:
            w.register_pose("pose_groundtruth", rate_hz=float(TICKS_PER_SEC), frame_id="world")
            w.register_imu("imu0", rate_hz=float(TICKS_PER_SEC), frame_id="auv0/imu")

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

                dvl = state.get("DVLSensor")
                if dvl is not None and bump("dvl", t_ns):
                    d = np.asarray(dvl, dtype=float).reshape(-1)
                    if d.size < 3:
                        continue
                    # 兼容 (3,) 与 (7,)
                    if d.size >= 7:
                        row = [t_ns, *d[:7]]
                    else:
                        row = [t_ns, *d[:3], np.nan, np.nan, np.nan, np.nan]
                    with open(dvl_csv, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(row)
                    n_dvl += 1

                depth = state.get("DepthSensor")
                if depth is not None and bump("depth", t_ns):
                    dz = float(np.asarray(depth).reshape(-1)[0])
                    with open(depth_csv, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([t_ns, dz])
                    n_depth += 1

                if k % max(1, n_steps // 5) == 0:
                    p = pose[:3, 3] if pose is not None else np.zeros(3)
                    print(
                        f"  step {k:4d}/{n_steps}  t={state['t']:6.2f}s  "
                        f"z={p[2]:7.2f}  dvl={n_dvl} depth={n_depth}"
                    )

            print(f"  done in {time.time() - t0:.1f}s  dvl={n_dvl} depth={n_depth}")
            print(f"  stats: {w.stats()}")

    return base


def main() -> int:
    ap = argparse.ArgumentParser(description="AUV nav suite: pose/imu/dvl/depth")
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
        print("需要 holoocean 与 numpy:", e)
        return 1

    seq_dir = collect(args.out, seconds=args.seconds, show=not args.headless)
    from dataset_norm.validate import main as validate_main

    print()
    print("validate:", seq_dir)
    return validate_main([seq_dir])


if __name__ == "__main__":
    sys.exit(main())
