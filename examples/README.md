# examples

## collect_auv_minimal.py

最小可运行采集：HoloOcean 悬停 AUV 短时巡航 → 位姿/IMU 按 `dataset_norm` 落盘 → 自动 `validate`。

```bash
# 在仓库根目录
python examples/collect_auv_minimal.py --list
python examples/collect_auv_minimal.py --headless --seconds 5
```

依赖：`holoocean`、`numpy`。产物默认在 `examples/out/SimpleUnderwater/auv0/track0/`。
