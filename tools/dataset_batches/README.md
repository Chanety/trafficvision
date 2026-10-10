# 数据集分发批次（人工复核包）

## 这批数据是什么

`datasets\videoset_02_label3`：332 帧 1080p、**尚未人工复核**（只有自动标注草稿，10 类口径）。
要分给多个人同时干，所以按 **120 张一包**切成 2 包，另有 100 帧由项目方自留，
图片体积大不进 `main`，放在独立分支 `codex/dataset-batches-v1` 的 `dataset_batches/` 下。

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
| `下载批次01/02.bat` | 拿到批次包并解压到 `datasets\videoset_02_review_v2\batch_NN\`（调用下面的 ps1） |
| `download_batch.ps1` | 取包逻辑：本地 zip → 镜像直链 → git 稀疏拉取 → 给出求助提示，四步兜底 |
| `打开标注_批次01/02.bat` | 用中文复核工具打开已下载的批次（`tools\annotate_zh.py`） |
| `打包复核结果.bat` + `pack_results.ps1` | 复核完把 `labels\` 打包成 `复核结果_batch_NN_日期.zip` 发回 |
| `_batch_config.bat` | 仓库 / 分支 / 落地目录配置，换地方只改这一个文件 |
| `make_batches.py` | 重新生成批次包与 bat（换数据集、换包大小都用它） |
| `口径说明.txt` | **类别 ↔ 权重口径**（软件输出哪几类、这份标注喂给哪版权重），随每个包一起发 |

## 拿包：下载脚本按这四步走，哪步先成功用哪步

1. **本地已经有 zip**：放在 `tools\dataset_batches\`、仓库根 `dataset_batches\`、
   `下载\` 或桌面，双击 bat 直接解压 —— **不联网**，适合 QQ / 网盘 / U 盘转发
2. **镜像直链**：`gh-proxy.com` → `ghproxy.net` → `raw.githubusercontent.com`
3. **git 稀疏拉取**：`git clone --depth 1 --filter=blob:none --sparse` +
   `sparse-checkout set --no-cone /dataset_batches/batch_NN.zip`，**只下载这一个包**（80 MB 那包实测 ~20 秒）
4. 都不行 → 打印三条明确出路（找负责人要 zip / 加协作者 / 浏览器手动下），不猜不静默

### 实测结论（2026-10-07，本机）

| 通道 | 结果 |
|---|---|
| 本地 zip 解压 | ✅ 通（32 张图 + 32 个标注 + classes.txt + 标注说明.txt） |
| `gh-proxy.com` / `ghproxy.net` 镜像 | ❌ 404 —— 仓库是**私有**的，匿名拿不到 |
| `cdn.jsdelivr.net` | ❌ 404（只服务公开仓库） |
| `raw.githubusercontent.com` | ❌ 连接被重置（国内常见） |
| git 稀疏拉取 | ✅ 通（仓库内已登录账号，只下需要的那个包） |

结论：**当前仓库是私有仓库**，所以给别人数据只有两条实际可行的路——

- **A. 直接把 zip 发给对方**（QQ / 网盘 / U 盘），对方放到 `tools\dataset_batches\` 双击同名 bat 即可；
  最快、不需要对方有 GitHub 账号
- **B. 把对方加为仓库协作者**，并让对方本机 Git 登录过自己的账号，走 git 那条路

如果希望"谁都能点链接下载"，要把仓库改成公开（或另开一个公开的数据仓库）——
这是对外可见性的决定，改之前先确认，改完之后镜像通道才会通。

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

## 上传批次包到分支

```powershell
git checkout codex/dataset-batches-v1        # 首次：git checkout -b codex/dataset-batches-v1
git add -f dataset_batches                   # 该目录在 main 被 .gitignore 忽略，需要 -f
git commit -m "data: videoset_02 人工复核批次包（10 类，120 张一包，含口径说明）"
git push -u origin codex/dataset-batches-v1
git checkout main
```

每个包的 sha256 见分支上的 `dataset_batches/manifest.json`；
同样的参数重跑 `make_batches.py` 产物是逐字节确定的（zip 内时间戳固定为 2026-01-01），
所以校验和不会因为重打而变。**但改了包内容（例如这次补进《口径说明.txt》）sha256 一定会变**，
换包时记得把 manifest 一起更新并通知已下载的人。
