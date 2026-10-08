# 模型权重说明

权重体积较大，**不随仓库分发**（`.gitignore` 已排除 `*.pt`）。
清单、用途、类别口径与 SHA-256 见 [manifest.json](manifest.json)。

## 当前默认权重

```
outputs/train/v7/weights/best.pt     yolo11n 链式微调 · 7 类合并口径
```

命令行默认就指向它，所以**只要这个文件到位，`python -m trafficvision.cli` 就能直接跑**。
文件不在时程序会给出明确提示（而不是把 Ultralytics 的文件找不到错误抛给用户）。

## 获取

1. **本项目微调权重**（v1~v9）：随作品成果链接提供（Release 附件或网盘）。
2. **上游基础权重**：
   - `yolov8s-worldv2.pt`、`yolo11n.pt` —— Ultralytics 官方发布（AGPL-3.0）
   - `ViT-B-32.pt` —— OpenAI CLIP 官方权重（MIT），放到 `weights/clip/`

## 校验下载是否完整

```bash
python bench/verify_models.py          # 按 manifest.json 逐个核对 SHA-256
```

清单里没有的文件会被标为“缺失”，哈希不符的标为“不匹配”——这两类都不能用于
正式评测，否则指标无法复现。

## 类别口径对照（**最容易出错的地方**）

| 口径 | 类别 | 用它的权重/数据 |
|---|---|---|
| 12 类基础 | car, truck, bus, motorcycle, bicycle, person, sign, bollard, tricycle, roadlamp, trafficlight, rider | `videoset_02_label2` 的原始标注 |
| **10 类合并** | … motorcycle/bicycle/tricycle → **nonmotor**，另含 roadlamp / trafficlight / rider | `datasets/train_v9`、v9 权重 |
| 7 类合并 | car, truck, bus, nonmotor, person, sign, bollard | **v7（当前默认）**、v8n、v8s |
| 9 类旧口径 | … 含 motorcycle / bicycle / tricycle 分开 | v1~v6、`videoset_01_label` |

> 用错口径的后果不是报错，而是**指标静默错位**。评测前先确认
> “权重类别数 == 数据集 yaml 里的类别数”。

## 许可

本项目全部微调权重都由 Ultralytics AGPL-3.0 权重微调而来，属衍生作品，
同样以 **AGPL-3.0** 提供。详见根目录 `LICENSE` 与 `NOTICE`。
