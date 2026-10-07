# 数据集分发批次（人工复核包）

## 这批数据是什么

`datasets\videoset_02_label`：332 帧 1080p、**尚未人工复核**（只有自动标注草稿）。
要分给多个人同时干，所以按 **150 张一包**切成 3 包，
图片体积大不进 `main`，放在独立分支 `codex/dataset-batches-v1` 的 `dataset_batches/` 下。

| 批次 | 帧数 | 包大小 | 覆盖视频 | 说明 |
|---|---|---|---|---|
| `batch_01` | 150 | 76.4 MB | 14 段 | 优先帧最多（三轮车→自行车→标志牌→路桩→公交/卡车） |
| `batch_02` | 150 | 76.9 MB | 14 段 | 中段优先帧 |
| `batch_03` | 32 | 16.9 MB | 9 段 | 余数包 |

划分依据：`datasets\videoset_02_label\review_order.txt` 覆盖全部 332 帧、
按价值从高到低、跨视频轮流；脚本按这个顺序**顺序切块**，
所以每包都覆盖多数视频、批次号越小优先帧越多。每包内部自带裁剪过的
`review_order.txt`，复核工具会照它排序。

## 文件清单

| 文件 | 作用 |
|---|---|
| `下载批次01/02/03.bat` | 拿到批次包并解压到 `datasets\videoset_02_review\batch_NN\`（调用下面的 ps1） |
| `download_batch.ps1` | 取包逻辑：本地 zip → 镜像直链 → git 稀疏拉取 → 给出求助提示，四步兜底 |
| `打开标注_批次01/02/03.bat` | 用中文复核工具打开已下载的批次（`tools\annotate_zh.py`） |
| `打包复核结果.bat` + `pack_results.ps1` | 复核完把 `labels\` 打包成 `复核结果_batch_NN_日期.zip` 发回 |
| `_batch_config.bat` | 仓库 / 分支 / 落地目录配置，换地方只改这一个文件 |
| `make_batches.py` | 重新生成批次包与 bat（换数据集、换包大小都用它） |

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

- **先载具、后行人**：骑车人、驾驶位里的人只标载具不标 person；工具按 `R` 一键清理重叠行人框。
- **类别 9 类不动**：`0 car, 1 truck, 2 bus, 3 motorcycle, 4 bicycle, 5 person, 6 sign, 7 bollard, 8 tricycle`。
- 电动车/踏板车归 `motorcycle`；三轮车与三轮棚车归 `tricycle`，不细分。
- 标志牌 / 路桩是静止类必须保留；其余静止目标不预判。
- **拿不准的框留着别删**——漏标比误检对模型伤害大得多。
- 这批素材实测的常见误检：红色招牌/灯箱标成标志牌、花坛边缘/路缘石标成路桩、
  店面招牌标成卡车、紧贴车辆行走的人标成骑行者。

## 回收结果

对方双击 `打包复核结果.bat` → 输入 1/2/3 → 生成 `复核结果_batch_NN_日期.zip`（只含 labels，几十 KB），
发回给负责人即可。

## 重新生成

```powershell
# 默认：datasets\videoset_02_label → 150 张一包，同时刷新 bat
"trafficlab-3d\env\Scripts\python.exe" tools\dataset_batches\make_batches.py --bats

# 只改 bat 模板、不重打 zip（快）
"trafficlab-3d\env\Scripts\python.exe" tools\dataset_batches\make_batches.py --bats --bats-only

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

每个包的 sha256 见分支上的 `dataset_batches/manifest.json`；
重跑 `make_batches.py` 产物是逐字节确定的（zip 内时间戳固定为 2026-01-01），
所以校验和不会因为重打而变。
