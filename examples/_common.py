"""examples 公共工具：被各采集脚本复用。"""

from __future__ import annotations

import math

import numpy as np

# HoveringAUV：后 4 个推进器水平推力
AUV_FORWARD = np.array([0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 6.0, 6.0])
AUV_IDLE = np.zeros(8)


def rot_to_quat_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 旋转矩阵 → 四元数 (w, x, y, z)。"""
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
    """仿真时间（秒）→ 纳秒整数。"""
    return int(round(float(t_sec) * 1e9))


def auv_forward(k: int, n_steps: int) -> np.ndarray:
    """前 80% 巡航，后段怠速。"""
    return AUV_FORWARD if k < int(0.8 * n_steps) else AUV_IDLE
