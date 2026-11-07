"""最小空海声学示例：AUV 向 UAV 定时 ping

双智能体（AUV + UAV），AUV 周期性发送声学消息，异步收集回传观测。
位姿经 dataset_norm 落盘；声学观测写入同序列的 acoustic_beacon/ 目录。

依赖：holoocean、numpy

用法（仓库根目录）：
    python examples/collect_acoustic_minimal.py --headless --seconds 6

产物：
    examples/out/AirSea-UV/SimpleUnderwater/
        uav0/track_acoustic/pose_groundtruth/
        auv0/track_acoustic/{pose_groundtruth/, imu0/, acoustic_beacon/}
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
UAV_ALTITUDE_CM = 2500.0
PING_EVERY_SEC = 2.0
PING_EVERY_TICKS = int(PING_EVERY_SEC * TICKS_PER_SEC)
BEACON_MAX_DIST = 500.0
ID_AUV, ID_UAV = 1, 0

AUV_FORWARD = np.array([0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 6.0, 6.0])
AUV_IDLE = np.zeros(8)

SCENARIO = {
    "name": "collect_acoustic_minimal",
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
                    "sensor_type": "AcousticBeaconSensor",
                    "configuration": {"id": ID_AUV, "MaxDistance": BEACON_MAX_DIST},
                },
            ],
        },
        {
            "agent_name": "uav0",
            "agent_type": "UavAgent",
            "control_scheme": 1,
            "location": [0.0, 40.0, 45.0],
            "sensors": [
                {"sensor_type": "PoseSensor"},
                {
                    "sensor_type": "AcousticBeaconSensor",
                    "configuration": {"id": ID_UAV, "MaxDistance": BEACON_MAX_DIST},
                },
            ],
        },
    ],
}

ACOUSTIC_HEADER = [
    "#timestamp [ns]",
    "r [m]",
    "phi [rad]",
    "theta [rad]",
    "anchor_x [m]",
    "anchor_y [m]",
    "anchor_z [m]",
    "gt_x [m]",
    "gt_y [m]",
    "gt_z [m]",
]


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


def write_acoustic_sensor(sensor_dir: str) -> str:
    """按 dataset_norm 约定写 acoustic_beacon 的 sensor.yaml 与空表头。"""
    os.makedirs(sensor_dir, exist_ok=True)
    yaml_path = os.path.join(sensor_dir, "sensor.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(
            "sensor_type: acoustic_beacon\n"
            f"rate_hz: {1.0 / PING_EVERY_SEC:.6g}\n"
            "frame_id: auv0/beacon\n"
            "topic: acoustic_beacon\n"
        )
    csv_path = os.path.join(sensor_dir, "data.csv")
    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(ACOUSTIC_HEADER)
    return csv_path


def append_acoustic(csv_path: str, row: list[float]) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def collect(out_root: str, seconds: float, show: bool, warmup: int) -> list[str]:
    import holoocean
    from holoocean.sensors import AcousticBeaconSensor

    n_steps = max(1, int(seconds * TICKS_PER_SEC))
    settings = Settings(
        world=WORLD,
        package="Ocean",
        ticks_per_sec=TICKS_PER_SEC,
        frames_per_sec=TICKS_PER_SEC,
        agents=("uav0", "auv0"),
        random_seed=0,
        holoocean_version=getattr(holoocean, "__version__", "unknown"),
        notes="minimal acoustic: AUV ping to UAV, async collect",
    )
    seq_root = os.path.join(out_root, "AirSea-UV")

    # 必须在 make 前清空类级残留，否则 beacon id 错位、ping 静默丢失
    n_stale = len(AcousticBeaconSensor.instances)
    AcousticBeaconSensor.instances = dict()
    if n_stale:
        print(f"  [清理] 清掉 {n_stale} 个残留 acoustic beacon 实例")

    print(f"scene={SCENARIO['name']}  steps={n_steps}  out={seq_root}")
    last: dict[tuple[str, str], int] = {}

    def bump(key: tuple[str, str], t_ns: int) -> bool:
        if t_ns > last.get(key, -1):
            last[key] = t_ns
            return True
        return False

    n_sent = n_recv = 0
    acoustic_csv = os.path.join(
        seq_root, WORLD, "auv0", "track_acoustic", "acoustic_beacon", "data.csv"
    )
    seqs: list[str] = []

    with holoocean.make(scenario_cfg=SCENARIO, show_viewport=show) as env:
        w_uav = SequenceWriter(
            root=seq_root, world=WORLD, agent="uav0",
            track="track_acoustic", settings=settings,
        )
        w_auv = SequenceWriter(
            root=seq_root, world=WORLD, agent="auv0",
            track="track_acoustic", settings=settings,
        )
        try:
            w_uav.register_pose("pose_groundtruth", rate_hz=float(TICKS_PER_SEC), frame_id="world")
            w_auv.register_pose("pose_groundtruth", rate_hz=float(TICKS_PER_SEC), frame_id="world")
            w_auv.register_imu("imu0", rate_hz=float(TICKS_PER_SEC), frame_id="auv0/imu")
            # 覆盖 writer 未提供的声学通道
            write_acoustic_sensor(
                os.path.join(seq_root, WORLD, "auv0", "track_acoustic", "acoustic_beacon")
            )

            # warm-up：等 UAV 稳定（pre_start 自由落体）
            state = None
            for _ in range(warmup):
                env.act("uav0", np.array([0.0, 0.0, 0.0, UAV_ALTITUDE_CM]))
                env.act("auv0", AUV_IDLE)
                state = env.tick()

            uav_xyz = np.zeros(3)
            if state and "uav0" in state and "PoseSensor" in state["uav0"]:
                uav_xyz = state["uav0"]["PoseSensor"][:3, 3].copy()

            t0 = time.time()
            for k in range(n_steps):
                # 到点就发，不等待（单锚点无需 stagger）
                if k % PING_EVERY_TICKS == 0:
                    env.send_acoustic_message(ID_AUV, ID_UAV, "MSG_REQU", "ping")
                    n_sent += 1

                env.act("uav0", np.array([0.0, 0.0, 0.0, UAV_ALTITUDE_CM]))
                env.act(
                    "auv0",
                    AUV_FORWARD if k < int(0.8 * n_steps) else AUV_IDLE,
                )
                state = env.tick()
                t_ns = t_to_ns(state["t"])

                for name, writer in (("uav0", w_uav), ("auv0", w_auv)):
                    pose = (state.get(name) or {}).get("PoseSensor")
                    if pose is None:
                        continue
                    p = pose[:3, 3]
                    q = rot_to_quat_wxyz(pose[:3, :3])
                    if bump((name, "pose"), t_ns):
                        writer.add_pose("pose_groundtruth", t_ns, (p[0], p[1], p[2]), q)
                    if name == "uav0":
                        uav_xyz = p.copy()

                auv_state = state.get("auv0") or {}
                imu = auv_state.get("IMUSensor")
                if imu is not None and bump(("auv0", "imu"), t_ns):
                    acc = np.asarray(imu[0], dtype=float).reshape(3)
                    gyr = np.asarray(imu[1], dtype=float).reshape(3)
                    w_auv.add_imu(
                        "imu0", t_ns,
                        (gyr[0], gyr[1], gyr[2]),
                        (acc[0], acc[1], acc[2]),
                    )

                # 异步收集回传
                beacon = auv_state.get("AcousticBeaconSensor")
                if beacon is not None and len(beacon) >= 6:
                    # ["MSG_RESPU", src, payload, phi, theta, r]
                    _kind, _src, _payload, phi, theta, r = beacon[:6]
                    auv_pose = auv_state.get("PoseSensor")
                    gt = (
                        auv_pose[:3, 3]
                        if auv_pose is not None
                        else np.zeros(3)
                    )
                    append_acoustic(
                        acoustic_csv,
                        [
                            t_ns,
                            float(r),
                            float(phi),
                            float(theta),
                            uav_xyz[0],
                            uav_xyz[1],
                            uav_xyz[2],
                            gt[0],
                            gt[1],
                            gt[2],
                        ],
                    )
                    n_recv += 1

                if k % max(1, n_steps // 5) == 0:
                    ap = (state.get("auv0") or {}).get("PoseSensor")
                    az = ap[:3, 3] if ap is not None else np.zeros(3)
                    print(
                        f"  step {k:4d}/{n_steps}  t={state['t']:6.2f}s  "
                        f"auv_z={az[2]:6.2f}  ping {n_sent}/{n_recv}"
                    )

            print(f"  done in {time.time() - t0:.1f}s  sent={n_sent} recv={n_recv}")
            print(f"  uav {w_uav.stats()}  auv {w_auv.stats()}")
            seqs = [
                os.path.join(seq_root, WORLD, "uav0", "track_acoustic"),
                os.path.join(seq_root, WORLD, "auv0", "track_acoustic"),
            ]
        finally:
            w_uav.close()
            w_auv.close()

    return seqs


def main() -> int:
    ap = argparse.ArgumentParser(description="Minimal AUV→UAV acoustic collect")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--warmup", type=int, default=90, help="预热步数（丢弃）")
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
            print("holocean import failed:", e)
            return 1
        return 0

    try:
        import holoocean  # noqa: F401
    except Exception as e:
        print("需要 holoocean 与 numpy:", e)
        return 1

    seqs = collect(
        args.out, seconds=args.seconds, show=not args.headless, warmup=args.warmup
    )
    from dataset_norm.validate import main as validate_main

    rc = 0
    for seq in seqs:
        print()
        print("validate:", seq)
        rc = max(rc, validate_main([seq]))
    return rc


if __name__ == "__main__":
    sys.exit(main())
