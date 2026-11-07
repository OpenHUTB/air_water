# examples

## collect_auv_minimal.py

最小可运行采集：HoloOcean 悬停 AUV 短时巡航 → 位姿/IMU 按 `dataset_norm` 落盘 → 自动 `validate`。

```bash
# 在仓库根目录
python examples/collect_auv_minimal.py --list
python examples/collect_auv_minimal.py --headless --seconds 5
```

依赖：`holoocean`、`numpy`。

产物路径（与下文 eval 示例一致）：

```text
examples/out/AirSea-UV/SimpleUnderwater/auv0/track0/
```

## eval_ate_from_sequence.py

读取真值与估计轨迹，计算 ATE / RPE（不需要 HoloOcean）。

```bash
# 合成轨迹演示
python examples/eval_ate_from_sequence.py --demo

# 真值 + 估计两套序列
python examples/eval_ate_from_sequence.py \
  examples/out/AirSea-UV/SimpleUnderwater/auv0/track0 \
  --est path/to/estimate_seq

# 自检：真值 + 固定偏移（不读估计文件）
python examples/eval_ate_from_sequence.py \
  examples/out/AirSea-UV/SimpleUnderwater/auv0/track0 --selfcheck
```

## collect_auv_camera.py

在 AUV 上挂载 RGB 与深度相机，位姿/IMU/图像一并按 `dataset_norm` 落盘并 `validate`。

```bash
python examples/collect_auv_camera.py --headless --seconds 4
```

依赖：`holoocean`、`numpy`、`pillow`。深度图以 16-bit PNG 保存，像素值单位为毫米。

## collect_airsea_minimal.py

最小空海双智能体：UAV 定高 + AUV 定深巡航，双方真值位姿（AUV 另含 IMU）分目录落盘并 `validate`。

```bash
python examples/collect_airsea_minimal.py --headless --seconds 4
```

产物：

```text
examples/out/AirSea-UV/SimpleUnderwater/uav0/track_airsea/
examples/out/AirSea-UV/SimpleUnderwater/auv0/track_airsea/
```

## collect_acoustic_minimal.py

最小声学示例：AUV 每 2 秒向 UAV ping，异步收集测距观测；位姿与 IMU 走 `dataset_norm`，声学写入 `acoustic_beacon/data.csv`。

```bash
python examples/collect_acoustic_minimal.py --headless --seconds 6
```

注意：创建仿真前会清空 `AcousticBeaconSensor` 类级实例，避免 id 残留导致静默丢包。
