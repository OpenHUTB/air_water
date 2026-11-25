"""空海三层 + 声学：AUV 向 UAV / USV 定时 ping

UAV、USV、AUV 同场景真值位姿落盘；AUV 周期向两个锚点发送声学消息
（错开 tick，避免同 tick 丢包），异步记录测距观测。

依赖：holoocean、numpy

用法（仓库根目录）：
    python examples/collect_3layer_acoustic.py --headless --seconds 6

产物：
    examples/out/AirSea-UV/SimpleUnderwater/
        uav0/track_3ac/pose_groundtruth/
        sv0/track_3ac/pose_groundtruth/
        auv0/track_3ac/{pose_groundtruth/, imu0/, acoustic_beacon/}
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dataset_norm import SequenceWriter, Settings  # noqa: E402

try:
    from ._common import AUV_FORWARD, AUV_IDLE, rot_to_quat_wxyz, t_to_ns
except ImportError:
    from _common import AUV_FORWARD, AUV_IDLE, rot_to_quat_wxyz, t_to_ns

WORLD = "SimpleUnderwater"
TICKS_PER_SEC = 30
UAV_ALTITUDE_CM = 2500.0
PING_EVERY_SEC = 2.0
PING_EVERY_TICKS = int(PING_EVERY_SEC * TICKS_PER_SEC)
# 实测往返约 3 步；多锚点发送间隔须 > RTT（见 lesson08）
PING_STAGGER_TICKS = 6
BEACON_MAX_DIST = 500.0
ID_UAV, ID_SURFACE, ID_AUV = 0, 2, 1
ANCHORS = ("uav0", "sv0")  # 锚点名 -> 见 ID 表
NAME2ID = {"uav0": ID_UAV, "sv0": ID_SURFACE}

SCENARIO = {
    "name": "collect_3layer_acoustic",
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
            "agent_name": "sv0",
            "agent_type": "SurfaceVessel",
            "control_scheme": 0,
            "location": [40.0, 0.0, 0.0],
            "sensors": [
                {"sensor_type": "PoseSensor"},
                {
                    "sensor_type": "AcousticBeaconSensor",
                    "configuration": {"id": ID_SURFACE, "MaxDistance": BEACON_MAX_DIST},
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
    "anchor",
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


def vessel_cmd(k: int) -> np.ndarray:
    t = k / TICKS_PER_SEC
    if t < 3:
        return np.array([700.0, 700.0])
    return np.array([900.0, 500.0])


def write_acoustic_meta(sensor_dir: str) -> str:
    os.makedirs(sensor_dir, exist_ok=True)
    with open(os.path.join(sensor_dir, "sensor.yaml"), "w", encoding="utf-8") as f:
        f.write(
            "sensor_type: acoustic_beacon\n"
            # 两锚点轮流 ping，观测速率约 1 Hz 量级
            "rate_hz: 1.5\n"
            "frame_id: auv0/beacon\n"
            "topic: acoustic_beacon\n"
        )
    csv_path = os.path.join(sensor_dir, "data.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(ACOUSTIC_HEADER)
    return csv_path


def collect(out_root: str, seconds: float, show: bool, warmup: int) -> list[str]:
    import holoocean
    from holoocean.sensors import AcousticBeaconSensor

    n_steps = max(1, int(seconds * TICKS_PER_SEC))
    settings = Settings(
        world=WORLD,
        package="Ocean",
        ticks_per_sec=TICKS_PER_SEC,
        frames_per_sec=TICKS_PER_SEC,
        agents=("uav0", "sv0", "auv0"),
        random_seed=0,
        holoocean_version=getattr(holoocean, "__version__", "unknown"),
        notes="3-layer pose + AUV acoustic ping to UAV/USV",
    )
    seq_root = os.path.join(out_root, "AirSea-UV")

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

    anchors_now = {n: np.zeros(3) for n in ANCHORS}
    pending: dict[str, list[int]] = {n: [] for n in ANCHORS}
    n_sent = n_recv = 0
    acoustic_csv = write_acoustic_meta(
        os.path.join(seq_root, WORLD, "auv0", "track_3ac", "acoustic_beacon")
    )
    seqs: list[str] = []
    writers: dict[str, SequenceWriter] = {}

    with holoocean.make(scenario_cfg=SCENARIO, show_viewport=show) as env:
        for name in ("uav0", "sv0", "auv0"):
            writers[name] = SequenceWriter(
                root=seq_root, world=WORLD, agent=name,
                track="track_3ac", settings=settings,
            )
        try:
            for name in writers:
                writers[name].register_pose(
                    "pose_groundtruth", rate_hz=float(TICKS_PER_SEC), frame_id="world"
                )
            writers["auv0"].register_imu(
                "imu0", rate_hz=float(TICKS_PER_SEC), frame_id="auv0/imu"
            )

            state = None
            for _ in range(warmup):
                env.act("uav0", np.array([0.0, 0.0, 0.0, UAV_ALTITUDE_CM]))
                env.act("sv0", np.zeros(2))
                env.act("auv0", AUV_IDLE)
                state = env.tick()

            t0 = time.time()
            for k in range(n_steps):
                # 到点错开发往两个锚点的 ping（间隔须 > RTT）
                for i, name in enumerate(ANCHORS):
                    if k % PING_EVERY_TICKS == i * PING_STAGGER_TICKS:
                        env.send_acoustic_message(
                            ID_AUV, NAME2ID[name], "MSG_REQU", "ping"
                        )
                        pending[name].append(k)
                        n_sent += 1

                env.act("uav0", np.array([0.0, 0.0, 0.0, UAV_ALTITUDE_CM]))
                env.act("sv0", vessel_cmd(k))
                env.act(
                    "auv0",
                    AUV_FORWARD if k < int(0.8 * n_steps) else AUV_IDLE,
                )
                state = env.tick()
                t_ns = t_to_ns(state["t"])

                for name in ("uav0", "sv0", "auv0"):
                    pose = (state.get(name) or {}).get("PoseSensor")
                    if pose is None:
                        continue
                    p = pose[:3, 3]
                    q = rot_to_quat_wxyz(pose[:3, :3])
                    if bump((name, "pose"), t_ns):
                        writers[name].add_pose(
                            "pose_groundtruth", t_ns, (p[0], p[1], p[2]), q
                        )
                    if name in ANCHORS:
                        anchors_now[name] = p.copy()

                auv = state.get("auv0") or {}
                imu = auv.get("IMUSensor")
                if imu is not None and bump(("auv0", "imu"), t_ns):
                    acc = np.asarray(imu[0], dtype=float).reshape(3)
                    gyr = np.asarray(imu[1], dtype=float).reshape(3)
                    writers["auv0"].add_imu(
                        "imu0", t_ns,
                        (gyr[0], gyr[1], gyr[2]),
                        (acc[0], acc[1], acc[2]),
                    )

                beacon = auv.get("AcousticBeaconSensor")
                if beacon is not None and len(beacon) >= 6:
                    _kind, src, _payload, phi, theta, r = beacon[:6]
                    src_i = int(src)
                    anchor_name = "uav0" if src_i == ID_UAV else (
                        "sv0" if src_i == ID_SURFACE else f"id{src_i}"
                    )
                    if pending.get(anchor_name):
                        pending[anchor_name].pop(0)
                    gt = (
                        auv["PoseSensor"][:3, 3]
                        if "PoseSensor" in auv
                        else np.zeros(3)
                    )
                    a = anchors_now.get(anchor_name, np.zeros(3))
                    with open(acoustic_csv, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(
                            [
                                t_ns,
                                anchor_name,
                                float(r),
                                float(phi),
                                float(theta),
                                a[0],
                                a[1],
                                a[2],
                                gt[0],
                                gt[1],
                                gt[2],
                            ]
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
            for name, w in writers.items():
                print(f"  {name}: {w.stats()}")
                seqs.append(os.path.join(seq_root, WORLD, name, "track_3ac"))
        finally:
            for w in writers.values():
                w.close()

    return seqs


def main() -> int:
    ap = argparse.ArgumentParser(description="3-layer pose + acoustic ping")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--warmup", type=int, default=60)
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
