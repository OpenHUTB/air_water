"""最小单波束声呐采集示例

自定义 SimpleUnderwater + HoveringAUV + SinglebeamSonar（10 Hz、128 距离单元），
位姿、IMU 与声呐回波按 dataset_norm 落盘，并自动 validate。

依赖：holoocean、numpy

用法（仓库根目录）：
    python examples/collect_sonar_minimal.py --headless --seconds 4

产物：
    examples/out/AirSea-UV/SimpleUnderwater/auv0/track_sonar/
        settings.json
        pose_groundtruth/
        imu0/
        singlebeam_sonar/
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
# 自定义轻量场景（比 OpenWater 预置加载快）
SCENE = {
    "name": "collect_sonar_minimal",
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
                    "sensor_type": "SinglebeamSonar",
                    "socket": "SonarSocket",
                    "Hz": 10,
                    "configuration": {
                        "OpeningAngle": 30,
                        "RangeBins": 128,
                        "RangeMin": 0.5,
                        "RangeMax": 30,
                        "RangeSigma": 0.1,
                        "InitOctreeRange": 40,
                    },
                },
            ],
        }
    ],
}
SONAR_HZ = 10
SONAR_BINS = 128
# HoveringAUV 后 4 推进器前进
FORWARD = np.array([0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 6.0, 6.0])


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


def write_sonar_meta(sensor_dir: str, n_bins: int, rate_hz: float) -> str:
    os.makedirs(sensor_dir, exist_ok=True)
    yaml_path = os.path.join(sensor_dir, "sensor.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(
            "sensor_type: singlebeam_sonar\n"
            f"rate_hz: {rate_hz:g}\n"
            "frame_id: auv0/sonar\n"
            "topic: singlebeam_sonar\n"
            f'notes: "RangeBins={n_bins}"\n'
        )
    header = ["#timestamp [ns]"] + [f"bin_{i} [a.u.]" for i in range(n_bins)]
    csv_path = os.path.join(sensor_dir, "data.csv")
    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)
    return csv_path


def collect(out_root: str, seconds: float, show: bool) -> str:
    import holoocean

    n_steps = max(1, int(seconds * TICKS_PER_SEC))
    n_bins = SONAR_BINS
    sonar_hz = float(SONAR_HZ)

    settings = Settings(
        world=WORLD,
        package="Ocean",
        ticks_per_sec=TICKS_PER_SEC,
        frames_per_sec=sonar_hz,
        agents=("auv0",),
        random_seed=0,
        holoocean_version=getattr(holoocean, "__version__", "unknown"),
        notes="singlebeam sonar collect on HoveringAUV",
    )
    seq_root = os.path.join(out_root, "AirSea-UV")
    print(f"scene={SCENE['name']}  steps={n_steps}  bins={n_bins}  out={seq_root}")

    last: dict[str, int] = {}

    def bump(key: str, t_ns: int) -> bool:
        if t_ns > last.get(key, -1):
            last[key] = t_ns
            return True
        return False

    sonar_csv = os.path.join(
        seq_root, WORLD, "auv0", "track_sonar", "singlebeam_sonar", "data.csv"
    )
    n_sonar = 0

    with holoocean.make(scenario_cfg=SCENE, show_viewport=show) as env:
        with SequenceWriter(
            root=seq_root, world=WORLD, agent="auv0",
            track="track_sonar", settings=settings,
        ) as w:
            w.register_pose("pose_groundtruth", rate_hz=float(TICKS_PER_SEC), frame_id="world")
            w.register_imu("imu0", rate_hz=float(TICKS_PER_SEC), frame_id="auv0/imu")
            write_sonar_meta(
                os.path.join(seq_root, WORLD, "auv0", "track_sonar", "singlebeam_sonar"),
                n_bins,
                sonar_hz,
            )

            t0 = time.time()
            for k in range(n_steps):
                env.act("auv0", FORWARD)
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

                sonar = state.get("SinglebeamSonar")
                if sonar is not None and bump("sonar", t_ns):
                    bins = np.asarray(sonar, dtype=float).reshape(-1)
                    if bins.size != n_bins:
                        # 长度不一致时截断/补零，避免 ragged csv
                        if bins.size > n_bins:
                            bins = bins[:n_bins]
                        else:
                            bins = np.pad(bins, (0, n_bins - bins.size))
                    row = [t_ns] + [float(x) for x in bins]
                    with open(sonar_csv, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(row)
                    n_sonar += 1

                if k % max(1, n_steps // 5) == 0:
                    p = pose[:3, 3] if pose is not None else np.zeros(3)
                    print(
                        f"  step {k:4d}/{n_steps}  t={state['t']:6.2f}s  "
                        f"xyz=[{p[0]:7.2f} {p[1]:7.2f} {p[2]:7.2f}]  sonar={n_sonar}"
                    )

            print(f"  done in {time.time() - t0:.1f}s  sonar_rows={n_sonar}")
            print(f"  stats: {w.stats()}")

    return os.path.join(seq_root, settings.world, "auv0", "track_sonar")


def main() -> int:
    ap = argparse.ArgumentParser(description="Minimal singlebeam sonar collect")
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
