# examples



## 一览

| 类型 | 脚本 |
|------|------|
| 入口 | `run_pipeline.py` |
| 采集 | `collect_auv_minimal` / `camera` / `seg` / `nav` / `airsea` / `3layer` / `acoustic` / `sonar` |
| 组合 | `collect_3layer_acoustic` |
| 评测 | `eval_ate_from_sequence` / `eval_ate_deadreckon` / `eval_segmentation_demo` / `eval_detection_demo` |
| 公共 | `_common.py`（四元数、时间戳、推进指令） |
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

## eval_ate_deadreckon.py

航位推算 ATE：序列含 `dvl/` 则积分 DVL，否则用带噪里程计与真值对比。

```bash
python examples/eval_ate_deadreckon.py --demo
python examples/eval_ate_deadreckon.py \
  examples/out/AirSea-UV/SimpleUnderwater/auv0/track_nav
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

## collect_sonar_minimal.py

最小单波束声呐：HoveringAUV 挂 `SinglebeamSonar`（10 Hz、128 距离单元），位姿/IMU/回波按 `dataset_norm` 落盘。

```bash
python examples/collect_sonar_minimal.py --headless --seconds 4
```

产物：`examples/out/AirSea-UV/SimpleUnderwater/auv0/track_sonar/`

## collect_auv_nav.py

AUV 导航套件：位姿 + IMU + DVL（15 Hz）+ 深度计（30 Hz），一次落盘并 `validate`。

```bash
python examples/collect_auv_nav.py --headless --seconds 4
```

产物：`examples/out/AirSea-UV/SimpleUnderwater/auv0/track_nav/`

## eval_segmentation_demo.py

分割精度示例：IoU / mIoU / 像素准确率（不依赖 HoloOcean）。

```bash
python examples/eval_segmentation_demo.py --demo
python examples/eval_segmentation_demo.py --dir path/to/masks   # 含 pred.npy 与 gt.npy
```

## eval_detection_demo.py

检测精度示例：Precision / Recall / F1（不依赖 HoloOcean）。

```bash
python examples/eval_detection_demo.py --demo
python examples/eval_detection_demo.py --matched 1,0,1,1 --gt 5
```

## collect_seg_camera.py

语义分割相机：AUV 挂载 `SemanticSegmentationCamera`，位姿/IMU/掩码按 `dataset_norm` 落盘。

```bash
python examples/collect_seg_camera.py --headless --seconds 3
```

产物：`examples/out/AirSea-UV/SimpleUnderwater/auv0/track_seg/`（掩码为彩色 RGB PNG 或单通道标签 PNG，视引擎输出而定）。

## run_pipeline.py

统一入口：采集 → validate → 轨迹/分割/检测评测 → 总表。

```bash
python examples/run_pipeline.py --collect auv --seconds 4
python examples/run_pipeline.py --collect airsea
python examples/run_pipeline.py --collect none   # 只校验已有序列并评测
```

`--collect` 可选：`auv` / `camera` / `airsea` / `acoustic` / `sonar` / `nav` / `none`。

## collect_airsea_3layer.py

空海三层最小采集：UAV 定高 + USV 弧线 + AUV 巡航，三方真值位姿与 AUV IMU 分目录落盘。

```bash
python examples/collect_airsea_3layer.py --headless --seconds 4
```
