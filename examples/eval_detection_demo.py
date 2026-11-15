"""检测精度示例：Precision / Recall / F1

不依赖 HoloOcean。演示如何把「预测是否命中」交给 metrics 计数。

用法：
    python examples/eval_detection_demo.py --demo
    python examples/eval_detection_demo.py --matched 1,0,1,1 --gt 5
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from metrics import precision_recall_f1  # noqa: E402


def parse_matched(s: str) -> list[bool]:
    """'1,0,1' -> [True, False, True]"""
    out: list[bool] = []
    for part in s.split(","):
        p = part.strip().lower()
        if p in ("1", "true", "t", "y", "yes"):
            out.append(True)
        elif p in ("0", "false", "f", "n", "no"):
            out.append(False)
        else:
            raise ValueError(f"bad token {part!r} in --matched")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Detection P/R/F1 demo")
    ap.add_argument("--demo", action="store_true", help="演示：5 真值，4 预测中 3 命中")
    ap.add_argument(
        "--matched",
        default=None,
        help="预测命中列表，如 1,0,1,1（1=TP，0=FP）",
    )
    ap.add_argument(
        "--gt",
        type=int,
        default=None,
        help="真值目标个数（与 --matched 联用）",
    )
    args = ap.parse_args()

    if args.matched is not None:
        if args.gt is None:
            print("error: --matched 需要同时给出 --gt")
            return 2
        matched = parse_matched(args.matched)
        n_gt = args.gt
        print(f"input: matched={matched}  n_gt={n_gt}")
    elif args.demo or (args.matched is None and args.gt is None):
        # 演示：真值 5，预测 4 条，其中 3 条命中
        matched = [True, True, False, True]
        n_gt = 5
        print("demo: n_gt=5, 4 predictions (3 hit, 1 false positive)")
    else:
        print("error: 提供 --demo 或 --matched/--gt")
        return 2

    m = precision_recall_f1(matched, n_gt)
    print(f"TP={int(m['tp'])} FP={int(m['fp'])} FN={int(m['fn'])}")
    print(f"Precision = {m['precision']:.4f}")
    print(f"Recall    = {m['recall']:.4f}")
    print(f"F1        = {m['f1']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
