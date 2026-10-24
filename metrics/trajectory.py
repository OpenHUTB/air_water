"""轨迹评测：ATE（绝对轨迹误差）与 RPE（相对位姿误差）。

约定：
- 位姿为 4x4 齐次变换矩阵，或 (position, quaternion_wxyz)；
- 真值与估计序列已按时间对齐（同长度、同索引对应）；
- 平移单位为米。
"""

from __future__ import annotations

import math
from typing import Sequence


def _as_positions(
    poses: Sequence[object],
) -> list[tuple[float, float, float]]:
    """接受 4x4 矩阵（list of list）或 (x, y, z)。"""
    out: list[tuple[float, float, float]] = []
    for p in poses:
        if isinstance(p, (list, tuple)) and len(p) == 3:
            out.append((float(p[0]), float(p[1]), float(p[2])))
        elif isinstance(p, (list, tuple)) and len(p) == 4:
            # 4x4 行或 [x,y,z,w] 齐次——取前 3 作平移仅当是矩阵行时另论
            # 矩阵：4 行 4 列
            if len(p[0]) == 4:  # type: ignore[index]
                out.append((float(p[0][3]), float(p[1][3]), float(p[2][3])))  # type: ignore[index]
            else:
                raise TypeError("pose row must be 4x4 matrix or xyz")
        else:
            raise TypeError(f"unsupported pose type: {type(p)!r}")
    return out


def _norm(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def ate_rmse(
    groundtruth: Sequence[object],
    estimate: Sequence[object],
    *,
    align: bool = False,
) -> float:
    """Absolute Trajectory Error RMSE（米）。

    align=True 时先做 Umeyama 式平移对齐（仅平移均值，不做旋转/尺度）。
    """
    if len(groundtruth) != len(estimate):
        raise ValueError(f"length mismatch: gt={len(groundtruth)} est={len(estimate)}")
    if len(groundtruth) == 0:
        raise ValueError("empty trajectory")

    gt = _as_positions(groundtruth)
    est = _as_positions(estimate)

    if align:
        def mean(pts: list[tuple[float, float, float]]) -> tuple[float, float, float]:
            n = len(pts)
            return (
                sum(p[0] for p in pts) / n,
                sum(p[1] for p in pts) / n,
                sum(p[2] for p in pts) / n,
            )

        mg, me = mean(gt), mean(est)
        gt = [(p[0] - mg[0], p[1] - mg[1], p[2] - mg[2]) for p in gt]
        est = [(p[0] - me[0], p[1] - me[1], p[2] - me[2]) for p in est]

    total = sum(_norm(g, e) ** 2 for g, e in zip(gt, est))
    return math.sqrt(total / len(gt))


def rpe(
    groundtruth: Sequence[object],
    estimate: Sequence[object],
    *,
    delta: int = 1,
) -> float:
    """Relative Position Error RMSE（米）：平移段长度误差。

    对每对 (i, i+delta) 比较相对位移向量的欧氏距离。
    """
    if len(groundtruth) != len(estimate):
        raise ValueError(f"length mismatch: gt={len(groundtruth)} est={len(estimate)}")
    if delta < 1:
        raise ValueError("delta must be >= 1")
    n = len(groundtruth)
    if n <= delta:
        raise ValueError(f"need length > delta ({delta}), got {n}")

    gt = _as_positions(groundtruth)
    est = _as_positions(estimate)

    total = 0.0
    count = 0
    for i in range(n - delta):
        g0, g1 = gt[i], gt[i + delta]
        e0, e1 = est[i], est[i + delta]
        dg = (g1[0] - g0[0], g1[1] - g0[1], g1[2] - g0[2])
        de = (e1[0] - e0[0], e1[1] - e0[1], e1[2] - e0[2])
        total += _norm(dg, de) ** 2
        count += 1
    return math.sqrt(total / count)
