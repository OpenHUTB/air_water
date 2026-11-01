# examples

## collect_auv_minimal.py

最小可运行采集：HoloOcean 悬停 AUV 短时巡航 → 位姿/IMU 按 `dataset_norm` 落盘 → 自动 `validate`。

```bash
# 在仓库根目录
python examples/collect_auv_minimal.py --list
python examples/collect_auv_minimal.py --headless --seconds 5
```

依赖：`holoocean`、`numpy`。产物默认在 `examples/out/SimpleUnderwater/auv0/track0/`。

## eval_ate_from_sequence.py

读取序列里的真值轨迹，计算 ATE / RPE（不需要 HoloOcean）。

```bash
python examples/eval_ate_from_sequence.py --demo
python examples/eval_ate_from_sequence.py examples/out/AirSea-UV/SimpleUnderwater/auv0/track0
```
