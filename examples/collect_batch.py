"""批量多 track 采集：按列表循环调用已有 collect 脚本

围绕「一次跑出多条序列」组织，复用 examples 内已有采集模块，
便于把数据集从单条 track 扩展到多条（仍保持每条可独立 validate）。

依赖：holoocean、numpy

用法（仓库根目录）：
    # 默认 auv 采集 3 条（track0 / track1 / track2）
    python examples/collect_batch.py --mode auv --count 3 --seconds 3

    # 指定模式列表（空格分隔）
    python examples/collect_batch.py --modes auv nav --count 2 --seconds 2

    # 仅列出将执行的计划，不启动仿真
    python examples/collect_batch.py --mode auv --count 3 --dry-run

说明：
  - 各 collect_* 自身固定 track 目录名；本脚本通过 --out 下不同批次子目录
    或顺序覆盖同一 track（由子脚本行为决定）。当前实现为顺序执行，
    产物落在 examples/out/，批次信息写入 settings.notes 后缀（由子脚本决定）。
  - 实际多 track 目录隔离：使用 --tag 追加到 out 根目录名，例如 out_batch1。
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

# 模式 -> 子模块与默认秒数
MODES = {
    "auv": ("collect_auv_minimal", 4.0),
    "camera": ("collect_auv_camera", 3.0),
    "seg": ("collect_seg_camera", 3.0),
    "airsea": ("collect_airsea_minimal", 4.0),
    "3layer": ("collect_airsea_3layer", 4.0),
    "3ac": ("collect_3layer_acoustic", 6.0),
    "acoustic": ("collect_acoustic_minimal", 6.0),
    "sonar": ("collect_sonar_minimal", 3.0),
    "nav": ("collect_auv_nav", 4.0),
}

# 支持 --track 参数的子脚本（可产生多条独立 track）
SUPPORT_TRACK = {"collect_auv_minimal"}


def plan(modes: list[str], count: int) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    for i in range(count):
        for m in modes:
            mod, sec = MODES[m]
            rows.append((m, mod, sec))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch multi-run collect")
    ap.add_argument(
        "--modes",
        nargs="+",
        default=["auv"],
        help=f"采集模式列表，可选: {', '.join(sorted(MODES))}",
    )
    ap.add_argument("--count", type=int, default=3, help="每个模式重复次数")
    ap.add_argument("--seconds", type=float, default=None, help="覆盖默认时长")
    ap.add_argument(
        "--tag",
        default="",
        help="附加到输出根目录名，如 --tag b1 → out_b1（隔离批次）",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    bad = [m for m in args.modes if m not in MODES]
    if bad:
        print(f"unknown modes: {bad}; allowed: {sorted(MODES)}")
        return 2
    if args.count < 1:
        print("--count must be >= 1")
        return 2

    out = os.path.join(_HERE, "out")
    if args.tag:
        out = os.path.join(_HERE, f"out_{args.tag}")

    rows = plan(args.modes, args.count)
    print("=" * 60)
    print(f"batch plan: {len(rows)} runs  modes={args.modes}  count={args.count}")
    print(f"out root:   {out}")
    print("=" * 60)
    for i, (m, mod, sec) in enumerate(rows, 1):
        s = args.seconds if args.seconds is not None else sec
        print(f"  [{i}/{len(rows)}] {m:10s}  {mod}  {s}s")

    if args.dry_run:
        print("\n[dry-run] 不启动仿真")
        return 0

    t0 = time.time()
    fails = 0
    run_idx = 0
    for i, (m, mod, sec) in enumerate(rows, 1):
        s = args.seconds if args.seconds is not None else sec
        cmd = [
            sys.executable,
            "-m",
            f"examples.{mod}",
            f"--seconds={s}",
            f"--out={out}",
        ]
        # 支持多 track 的模式：顺序写入 track0, track1, ...
        if mod in SUPPORT_TRACK:
            cmd.append(f"--track=track{run_idx}")
        cmd.append("--show" if args.show else "--headless")
        print(f"\n>>> [{i}/{len(rows)}] {' '.join(cmd)}")
        rc = subprocess.call(cmd, cwd=_REPO_ROOT)
        run_idx += 1
        if rc != 0:
            fails += 1
            print(f"[FAIL] {m} rc={rc}")
        else:
            print(f"[OK] {m}")

    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(f"  total: {len(rows)}  fail: {fails}")
    print(f"  wall:  {time.time() - t0:.1f}s")
    print(f"  out:   {out}")
    print("  result:", "PASS" if fails == 0 else "FAIL")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
