"""空海数据集流水线：采集 → 校验 → 评测

把 examples 里已有的采集与评测串成一个可运行入口，围绕
「得到一条合格序列并算出指标」这一目标组织，而不是分散脚本。

依赖：holoocean、numpy

用法（仓库根目录）：
    python examples/run_pipeline.py --collect auv      --seconds 4
    python examples/run_pipeline.py --collect airsea   --seconds 4
    python examples/run_pipeline.py --collect nav      --seconds 4
    python examples/run_pipeline.py --collect camera   --seconds 3
    python examples/run_pipeline.py --collect acoustic --seconds 6
    python examples/run_pipeline.py --collect sonar    --seconds 3
    python examples/run_pipeline.py --collect none     # 跳过采集，只校验已有序列

阶段：
  1) collect_*  产出 dataset_norm 序列
  2) validate   结构/时间戳/图像索引
  3) eval       轨迹 ATE（selfcheck）+ 分割/检测 demo
  4) summary    打印总表
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# 采集模式 -> (模块名, 默认秒数)
COLLECTORS = {
    "auv": ("collect_auv_minimal", 4.0),
    "camera": ("collect_auv_camera", 3.0),
    "airsea": ("collect_airsea_minimal", 4.0),
    "acoustic": ("collect_acoustic_minimal", 6.0),
    "sonar": ("collect_sonar_minimal", 3.0),
    "nav": ("collect_auv_nav", 4.0),
    "none": (None, 0.0),
}


def run_module(module: str, argv: list[str]) -> int:
    """以子进程跑 examples 下的模块，便于隔离环境与退出码。"""
    cmd = [sys.executable, "-m", module, *argv]
    print(f"\n>>> {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=_REPO_ROOT)


def find_latest_sequence(out_root: str) -> list[str]:
    """扫描 out 下含 settings.json 且至少有一条数据行的序列目录。"""
    base = os.path.join(out_root, "AirSea-UV")
    found: list[str] = []
    if not os.path.isdir(base):
        return found
    for root, _dirs, files in os.walk(base):
        if "settings.json" not in files:
            continue
        has_row = False
        for dirpath, _d, fns in os.walk(root):
            if "data.csv" not in fns:
                continue
            path = os.path.join(dirpath, "data.csv")
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    # 表头 + 至少一行
                    next(f, None)
                    if next(f, None) is not None:
                        has_row = True
                        break
            except OSError:
                continue
        if has_row:
            found.append(root)
    return sorted(found)


def stage_validate(sequences: list[str]) -> int:
    from dataset_norm.validate import main as validate_main

    rc = 0
    for seq in sequences:
        print(f"\n>>> validate {seq}")
        rc = max(rc, validate_main([seq]))
    return rc


def stage_eval(out_root: str) -> int:
    """轨迹 selfcheck + 分割/检测 demo（不依赖额外数据文件）。"""
    rc = 0
    seqs = find_latest_sequence(out_root)
    # 优先含 pose 的 auv 序列
    pose_seqs = [s for s in seqs if os.path.isfile(os.path.join(s, "pose_groundtruth", "data.csv"))]
    if pose_seqs:
        print(f"\n>>> eval ATE selfcheck on {pose_seqs[-1]}")
        rc = max(
            rc,
            run_module(
                "examples.eval_ate_from_sequence",
                [pose_seqs[-1], "--selfcheck"],
            ),
        )
    else:
        print("\n>>> eval ATE: no pose sequence found, skip")

    print("\n>>> eval segmentation demo")
    rc = max(rc, run_module("examples.eval_segmentation_demo", ["--demo"]))
    print("\n>>> eval detection demo")
    rc = max(rc, run_module("examples.eval_detection_demo", ["--demo"]))
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="AirSea dataset pipeline")
    ap.add_argument(
        "--collect",
        choices=sorted(COLLECTORS.keys()),
        default="auv",
        help="采集模式（none=跳过）",
    )
    ap.add_argument("--seconds", type=float, default=None, help="覆盖默认采集时长")
    ap.add_argument(
        "--out",
        default=os.path.join(_HERE, "out"),
        help="输出根目录",
    )
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--show", action="store_true", help="显示仿真窗口")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("AirSea dataset pipeline")
    print(f"collect={args.collect}  out={args.out}")
    print("=" * 60)

    module, default_sec = COLLECTORS[args.collect]
    rc = 0

    # ---- 1) collect ----
    if module is not None:
        seconds = args.seconds if args.seconds is not None else default_sec
        argv = [f"--seconds={seconds}", f"--out={args.out}"]
        argv.append("--show" if args.show else "--headless")
        rc = max(rc, run_module(f"examples.{module}", argv))
        if rc != 0:
            print(f"\n[FAIL] collect stage rc={rc}")
            return rc
        print("\n[OK] collect stage")
    else:
        print("\n[SKIP] collect stage")

    # ---- 2) validate ----
    sequences = find_latest_sequence(args.out)
    if not sequences:
        print("\n[FAIL] no sequences under", os.path.join(args.out, "AirSea-UV"))
        return 1
    vrc = stage_validate(sequences)
    rc = max(rc, vrc)
    print("\n[OK] validate stage" if vrc == 0 else f"\n[FAIL] validate rc={vrc}")
    if vrc != 0:
        return vrc

    # ---- 3) eval ----
    if args.skip_eval:
        print("\n[SKIP] eval stage")
    else:
        erc = stage_eval(args.out)
        rc = max(rc, erc)
        print("\n[OK] eval stage" if erc == 0 else f"\n[FAIL] eval rc={erc}")

    # ---- 4) summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  sequences: {len(sequences)}")
    for s in sequences:
        print(f"    - {s}")
    print(f"  pipeline_rc: {rc}")
    print(f"  wall_time:   {time.time() - t0:.1f} s")
    print("  result:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
