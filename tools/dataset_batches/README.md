# 数据集分发批次（人工复核包）

`datasets\videoset_02_label`（332 帧待人工复核）要分给多个人同时干，
按 **150 张一包** 切成 3 包。图片体积大，不进 `main` 分支，
放在独立分支 `codex/dataset-batches-v1` 的 `dataset_batches/` 目录下，
用本目录的 bat 一键下载。

| 批次 | 帧数 | 包大小 | 覆盖视频 | 说明 |
|---|---|---|---|---|
| `batch_01` | 150 | 80.2 MB | 14 段 | 优先帧最多（三轮车→自行车→标志牌→路桩→公交/卡车） |
| `batch_02` | 150 | 80.6 MB | 14 段 | 中段优先帧 |
| `batch_03` | 32 | 17.7 MB | 9 段 | 余数包，顺带扫尾 |

划分方式：按 `datasets\videoset_02_label\review_order.txt` 的顺序**顺序切块**，
该顺序已覆盖全部 332 帧、按价值从高到低、跨视频轮流，所以每包都能覆盖多数视频。
每包内部自带一份裁剪过的 `review_order.txt`，复核工具会照它排序。

## 文件清单

| 文件 | 作用 |
|---|---|
| `下载批次01/02/03.bat` | 从 GitHub 分支下载对应压缩包，解压到 `datasets\videoset_02_review\batch_NN\` |
| `打开标注_批次01/02/03.bat` | 用中文复核工具打开已下载的批次（`tools\annotate_zh.py`） |
| `打包复核结果.bat` | 复核完把自己的 `labels\` 打包成 `复核结果_batch_NN_日期.zip` 发回给负责人 |
| `_batch_config.bat` | 仓库 / 分支 / 落地目录配置，换地方只改这一个文件 |
| `make_batches.py` | 重新生成批次包与 bat（换数据集、换包大小都用它） |

## 给别人怎么用

**情况一：对方有本仓库**（`git clone` 过）

1. 双击 `tools\dataset_batches\下载批次NN.bat`（约 80 MB，等提示"完成"）
2. 双击 `tools\dataset_batches\打开标注_批次NN.bat` 开始复核
3. 干完双击 `打包复核结果.bat`，把生成的 zip 发回

**情况二：对方只有一台普通电脑**

1. 直接下载并双击这三个 bat 里属于他的那个（右键"链接另存为"）：
   [下载批次01.bat](https://raw.githubusercontent.com/Chanety/ai-software-opensource/main/tools/dataset_batches/%E4%B8%8B%E8%BD%BD%E6%89%B9%E6%AC%A101.bat) ·
   [下载批次02.bat](https://raw.githubusercontent.com/Chanety/ai-software-opensource/main/tools/dataset_batches/%E4%B8%8B%E8%BD%BD%E6%89%B9%E6%AC%A102.bat) ·
   [下载批次03.bat](https://raw.githubusercontent.com/Chanety/ai-software-opensource/main/tools/dataset_batches/%E4%B8%8B%E8%BD%BD%E6%89%B9%E6%AC%A103.bat)
2. 压缩包里有 `标注说明.txt`，写清了规则与常见误检；
   用任何 YOLO 格式标注工具打开即可（`images/` 是图，`labels/` 是标注，`classes.txt` 是类别表）

压缩包也可直接下（不需要 bat）：

- <https://raw.githubusercontent.com/Chanety/ai-software-opensource/codex/dataset-batches-v1/dataset_batches/batch_01.zip>
- <https://raw.githubusercontent.com/Chanety/ai-software-opensource/codex/dataset-batches-v1/dataset_batches/batch_02.zip>
- <https://raw.githubusercontent.com/Chanety/ai-software-opensource/codex/dataset-batches-v1/dataset_batches/batch_03.zip>

每个包的 sha256 见分支上的 `dataset_batches/manifest.json`。

## 复核规则（包内 `标注说明.txt` 也是同一份）

- **先载具、后行人**：骑车人、驾驶位里的人只标载具不标 person；工具按 `R` 一键清理重叠行人框。
- **类别 9 类不动**：`0 car, 1 truck, 2 bus, 3 motorcycle, 4 bicycle, 5 person, 6 sign, 7 bollard, 8 tricycle`。
- 电动车/踏板车归 `motorcycle`；三轮车与三轮棚车归 `tricycle`，不细分。
- 标志牌 / 路桩是静止类，必须保留；其余静止目标不影响预判。
- **拿不准的框留着别删**——漏标比误检对模型伤害大得多。
- 常见误检（这批素材实测）：红色招牌/灯箱标成标志牌、花坛边缘/路缘石标成路桩、
  店面招牌标成卡车、紧贴车辆行走的人标成骑行者。

## 重新生成

```powershell
# 默认：datasets\videoset_02_label → 150 张一包，同时刷新 bat
"trafficlab-3d\env\Scripts\python.exe" tools\dataset_batches\make_batches.py --bats

# 换数据集 / 换包大小
"trafficlab-3d\env\Scripts\python.exe" tools\dataset_batches\make_batches.py `
    --source datasets/videoset_02_label2 --name videoset_02_review2 --size 100 --bats
```

> 注意：`videoset_02_label2` 是 10 类（多了 `nonmotor / roadlamp / trafficlight / rider`），
> 而 `tools\annotate_zh.py` 目前只认 9 类，直接发 10 类标注会错类别。
> 要发 10 类那版，得先把标注工具升级到 10 类；本次分发用的是 9 类那份。

## 上传批次包到分支

```powershell
git checkout codex/dataset-batches-v1        # 首次：git checkout -b codex/dataset-batches-v1
git add -f dataset_batches                   # 该目录在 main 被 .gitignore 忽略，需要 -f
git commit -m "data: videoset_02 人工复核批次包（150 张一包）"
git push -u origin codex/dataset-batches-v1
git checkout main
```
