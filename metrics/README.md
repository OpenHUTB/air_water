# metrics

空海数据集评测指标库。与 `dataset_norm` 配套：数据按规范落盘后，用本模块计算算法精度。

## 模块

| 模块 | 指标 |
|------|------|
| `trajectory` | ATE RMSE、RPE（米） |
| `segmentation` | IoU / mIoU / Pixel Accuracy |
| `detection` | Precision / Recall / F1 |

## 约定

- **轨迹**：位姿支持 `(x,y,z)` 或 4×4 齐次矩阵；时间序列需先对齐到同一索引。
- **分割**：`pred` / `gt` 为整型标签数组，形状相同；类别为 `0..C-1`。
- **检测**：由调用方完成匹配后传入 `matched` 布尔列表与真值数量。

## 用法

```python
from metrics import ate_rmse, rpe, miou, precision_recall_f1

ate = ate_rmse(gt_xyz, est_xyz, align=True)
err = rpe(gt_xyz, est_xyz, delta=10)

m = miou(pred, gt, classes=list(range(10)))
prf = precision_recall_f1([True, False, True], n_groundtruth=5)
```

## 自测

```bash
python -m tests.test_metrics
```

依赖：`numpy`（分割模块）。轨迹与检测模块仅用标准库。
