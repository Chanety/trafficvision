# 数据集分发批次（人工复核包）

## 这批数据是什么

`datasets\videoset_02_label3`：332 帧 1080p、**尚未人工复核**（只有自动标注草稿，10 类口径）。
要分给多个人同时干，所以按 **120 张一包**切成 2 包，另有 100 帧由项目方自留。

> **2026-10-10 起改为线下分发**：批次包不再放 git 分支。原因见文末《为什么不再用分支分发》——
> 两版共 356 MB 的包把仓库撑到 520 MB，clone 一次要下 520 MB；现在普通 clone 是 28.8 MB。
> 包本体留在项目根 `dataset_batches\`（已被 .gitignore 排除），需要时直接转发/网盘给对方。

| 批次 | 帧数 | 包大小 | 覆盖视频 | 说明 |
|---|---|---|---|---|
| `batch_01` | 120 | 64.1 MB | 14 段 | 优先帧最多（三轮车→自行车→标志牌→路桩→公交/卡车） |
| `batch_02` | 112 | 61.1 MB | 11 段 | 后段优先帧 |
| `batch_self_keep` | 100 | 53.4 MB | 14 段 | 优先级最前的自有部分，**不打人工任务标记**，不外发 |

划分依据：`datasets\videoset_02_label\review_order.txt` 覆盖全部 332 帧、
按价值从高到低、跨视频轮流；脚本按这个顺序**顺序切块**，
所以每包都覆盖多数视频、批次号越小优先帧越多。每包内部自带裁剪过的
`review_order.txt`，复核工具会照它排序。

> ⚠️ 重新生成时**必须带 `--order-file datasets\videoset_02_label\review_order.txt`**，
> 否则会退化成"按文件名排序"切块，包内帧集合与已发出去的完全不同（2026-10-10 踩过一次）。

## 文件清单

| 文件 | 作用 |
|---|---|
| `解压批次.bat` | 把别人发来的 `batch_NN.zip` 解压到 `datasets\videoset_02_review_v2\batch_NN\`（不联网） |
| `打开标注_批次01/02.bat` | 用中文复核工具打开已下载的批次（`tools\annotate_zh.py`） |
| `打包复核结果.bat` + `pack_results.ps1` | 复核完把 `labels\` 打包成 `复核结果_batch_NN_日期.zip` 发回 |
| `_batch_config.bat` | 落地目录配置，换地方只改这一个文件 |
| `make_batches.py` | 重新生成批次包与 bat（换数据集、换包大小都用它） |
| `口径说明.txt` | **类别 ↔ 权重口径**（软件输出哪几类、这份标注喂给哪版权重），随每个包一起发 |

## 拿包（线下）

1. 负责人把 `batch_01.zip` / `batch_02.zip` 直接发给复核人（QQ / 网盘 / U 盘，60 MB 左右一个）。
2. 复核人把 zip 丢进 `tools\dataset_batches\`，双击 `解压批次.bat` 选包解压
   （默认解到 `datasets\videoset_02_review_v2\batch_NN\`，解压前会问，不会覆盖已改过的标注）。
3. 双击 `打开标注_批次01.bat` 开始复核。

为什么不再走链接下载：仓库是**私有**的，`gh-proxy` / `jsdelivr` / `raw.githubusercontent`
这些匿名通道一律 404 或连接重置（2026-10-07 实测），只有"对方本机登录过 GitHub 再走 git"
能通——绕一大圈，最后还是要账号。直接发文件最快，也不要求对方有 GitHub 账号。

> 现在仓库里**不再托管批次包**，所以"谁都能点链接下载"这件事在没有公开数据仓库之前不成立；
> 要公开下载就得先把仓库转公开，或者建一个公开的数据仓库——这是对外可见性的决定，需要先确认。

## 复核规则（包内 `标注说明.txt` 是同一份）

- **先载具、后行人**：骑车/驾车的人单独标 `rider`（非机动车驾驶员），不标 person；纯步行的人才标 person。
- **类别 10 类、顺序不可改**：`0 car, 1 truck, 2 bus, 3 nonmotor, 4 person, 5 sign, 6 bollard, 7 roadlamp, 8 trafficlight, 9 rider`。
- 电动车/踏板车、自行车、三轮车与三轮棚车**统一归 `nonmotor`**（用户要求不拆分）。
- 静态类（sign / bollard / roadlamp / trafficlight）必须保留；其余静止目标不预判。
- **拿不准的框留着别删**——漏标比误检对模型伤害大得多。
- 这批素材实测的常见误检：红色招牌/灯箱标成标志牌、花坛边缘/路缘石标成路桩、
  店面招牌标成卡车、紧贴车辆行走的人标成骑行者。

## 回收结果

对方双击 `打包复核结果.bat` → 输入 1/2/3 → 生成 `复核结果_batch_NN_日期.zip`（只含 labels，几十 KB），
发回给负责人即可。

## 重新生成

```powershell
# 当前这批：10 类口径、120 张一包、自留优先级最前的 30%
"trafficlab-3d\env\Scripts\python.exe" tools\dataset_batches\make_batches.py `
    --source datasets/videoset_02_label3 `
    --order-file datasets/videoset_02_label/review_order.txt `
    --size 120 --name videoset_02_review_v2 --keep-ratio 0.3 --bats

# 只改 bat 模板、不重打 zip（快）
"trafficlab-3d\env\Scripts\python.exe" tools\dataset_batches\make_batches.py --bats --bats-only

# 换数据集 / 换包大小
"trafficlab-3d\env\Scripts\python.exe" tools\dataset_batches\make_batches.py `
    --source datasets/videoset_02_label3 --name videoset_02_review2 --size 100 --bats
```

> 口径提醒（2026-10-10 更新）：现在发的这批是 **10 类**（`classes.txt` 与包内《标注说明.txt》
> 都由数据集目录生成，不再硬编码）。9 类的旧包与旧 bat 已移到 `_stale_9class\`，不要再发。
> 每个包内都有《口径说明.txt》，说明软件输出类别与权重对应关系（完整版见 `docs\28-类别与权重口径对照.md`）。

## 为什么不再用分支分发（2026-10-10）

原来批次包走独立分支 `codex/dataset-batches-v1`（`main` 被 .gitignore 挡住，只能 `git add -f`）。
代价在换包时暴露出来：

| 项 | 数值 |
|---|---|
| 一批包（3 个 zip） | 178 MB |
| 换一版后分支历史里留两版 | 356 MB |
| 仓库 `.git` 因此涨到 | 524.6 MB |
| 评审/协作者 clone 一次要下 | 520 MB |

包本身是**可再生成的产物**（`make_batches.py` 一条命令、逐字节确定），没有版本价值，
不该占着代码仓库的历史。处理方式：用 `git filter-repo --path dataset_batches/ --invert-paths`
把 `dataset_batches/` 从全部历史里摘掉、删除该分支，包改为线下转发。

结果：`.git` 13.6 MB，普通 clone **28.8 MB**（最大的对象是 docs 里 3.3 MB 的 UI 截图）。

> 遗留：GitHub 上那个批次包的 PR（`refs/pull/1/head`）仍指向被删分支的提交，
> 服务端对象要等 GitHub 自行 GC；**普通 clone 不会拉它**，所以不影响使用。
