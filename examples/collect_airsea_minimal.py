"""最小空海双智能体采集：UAV + AUV

同一场景中：
- uav0（UavAgent）保持定高
- auv0（HoveringAUV）定深巡航
双方真值位姿（AUV 另含 IMU）按 dataset_norm 分目录落盘，并自动 validate。

依赖：holoocean、numpy

用法（仓库根目录）：
    python examples/collect_airsea_minimal.py --headless --seconds 4
    python examples/collect_airsea_minimal.py --list

产物：
    examples/out/AirSea-UV/SimpleUnderwater/
        uav0/track_airsea/{settings.json, pose_groundtruth/}
        auv0/track_airsea/{settings.json, pose_groundtruth/, imu0/}
"""

from __future__ import annotations

import argparse
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
# 高度指令单位是厘米（HoloOcean UavAgent control_scheme 1）
UAV_ALTITUDE_CM = 2500.0  # 25 m

AUV_FORWARD = np.array([0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 6.0, 6.0])
AUV_IDLE = np.zeros(8)

SCENARIO = {
    "name": "collect_airsea_minimal",
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
            ],
        },
        {
            "agent_name": "uav0",
            "agent_type": "UavAgent",  # 类名，不是 "UAV"
            "control_scheme": 1,  # [roll, pitch, yaw_rate, altitude_cm]
            "location": [0.0, 40.0, 45.0],  # 起始高度留余量，抵消 pre_start 下落
            "sensors": [
                {"sensor_type": "PoseSensor"},
            ],
        },
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


def uav_cmd() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, UAV_ALTITUDE_CM])


def collect(out_root: str, seconds: float, show: bool) -> list[str]:
    import holoocean

    n_steps = max(1, int(seconds * TICKS_PER_SEC))
    settings = Settings(
        world=WORLD,
        package="Ocean",
        ticks_per_sec=TICKS_PER_SEC,
        frames_per_sec=TICKS_PER_SEC,
        agents=("uav0", "auv0"),
        random_seed=0,
        holoocean_version=getattr(holoocean, "__version__", "unknown"),
        notes="minimal air-sea: UAV altitude hold + AUV depth cruise",
    )
    seq_root = os.path.join(out_root, "AirSea-UV")
    print(f"scene={SCENARIO['name']}  steps={n_steps}  out={seq_root}")

    last_t_ns: dict[tuple[str, str], int] = {}

    def bump(key: tuple[str, str], t_ns: int) -> bool:
        if t_ns > last_t_ns.get(key, -1):
            last_t_ns[key] = t_ns
            return True
        return False

    seqs: list[str] = []
    with holoocean.make(scenario_cfg=SCENARIO, show_viewport=show) as env:
        w_uav = SequenceWriter(
            root=seq_root, world=WORLD, agent="uav0", track="track_airsea", settings=settings
        )
        w_auv = SequenceWriter(
            root=seq_root, world=WORLD, agent="auv0", track="track_airsea", settings=settings
        )
        try:
            w_uav.register_pose("pose_groundtruth", rate_hz=float(TICKS_PER_SEC), frame_id="world")
            w_auv.register_pose("pose_groundtruth", rate_hz=float(TICKS_PER_SEC), frame_id="world")
            w_auv.register_imu("imu0", rate_hz=float(TICKS_PER_SEC), frame_id="auv0/imu")

            t0 = time.time()
            for k in range(n_steps):
                env.act("uav0", uav_cmd())
                env.act(
                    "auv0",
                    AUV_FORWARD if k < int(0.8 * n_steps) else AUV_IDLE,
                )
                state = env.tick()
                t_ns = t_to_ns(state["t"])

                for name, writer in (("uav0", w_uav), ("auv0", w_auv)):
                    agent_state = state.get(name) or {}
                    pose = agent_state.get("PoseSensor")
                    if pose is None:
                        continue
                    p = pose[:3, 3]
                    q = rot_to_quat_wxyz(pose[:3, :3])
                    if bump((name, "pose"), t_ns):
                        writer.add_pose(
                            "pose_groundtruth", t_ns, (p[0], p[1], p[2]), q
                        )

                auv = state.get("auv0") or {}
                imu = auv.get("IMUSensor")
                if imu is not None and bump(("auv0", "imu"), t_ns):
                    acc = np.asarray(imu[0], dtype=float).reshape(3)
                    gyr = np.asarray(imu[1], dtype=float).reshape(3)
                    w_auv.add_imu(
                        "imu0", t_ns,
                        (gyr[0], gyr[1], gyr[2]),
                        (acc[0], acc[1], acc[2]),
                    )

                if k % max(1, n_steps // 5) == 0:
                    up = (state.get("uav0") or {}).get("PoseSensor")
                    ap = (state.get("auv0") or {}).get("PoseSensor")
                    uz = up[:3, 3] if up is not None else np.zeros(3)
                    az = ap[:3, 3] if ap is not None else np.zeros(3)
                    print(
                        f"  step {k:4d}/{n_steps}  t={state['t']:6.2f}s  "
                        f"uav=[{uz[0]:6.2f} {uz[1]:6.2f} {uz[2]:7.2f}]  "
                        f"auv=[{az[0]:6.2f} {az[1]:6.2f} {az[2]:7.2f}]"
                    )

            print(f"  done in {time.time() - t0:.1f}s real time")
            print(f"  uav stats: {w_uav.stats()}")
            print(f"  auv stats: {w_auv.stats()}")
            seqs = [
                os.path.join(seq_root, WORLD, "uav0", "track_airsea"),
                os.path.join(seq_root, WORLD, "auv0", "track_airsea"),
            ]
        finally:
            w_uav.close()
            w_auv.close()

    return seqs


def main() -> int:
    ap = argparse.ArgumentParser(description="Minimal UAV+AUV collect")
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

    seqs = collect(args.out, seconds=args.seconds, show=not args.headless)
    from dataset_norm.validate import main as validate_main

    rc = 0
    for seq in seqs:
        print()
        print("validate:", seq)
        rc = max(rc, validate_main([seq]))
    return rc


if __name__ == "__main__":
    sys.exit(main())
