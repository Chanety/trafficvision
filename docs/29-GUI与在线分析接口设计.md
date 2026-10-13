# 29 · 图形界面与在线分析接口设计

日期：2026-10-13。对应赛道一「AI+软件创新」的硬性要求：
*作品须为已完成开发的软件产品*、*可提供在线演示地址或可安装的软件包*、
*提交软件说明文档 / 功能演示视频 / 核心算法说明 / 测试报告*。

---

## 一、为什么要做这两件事

在此之前，软件只有命令行入口（`python -m trafficvision.cli`）。这对算法验证足够，
但对"软件产品"这一验收口径不够：评审看到的不是产品，而是脚本。

缺口有两处，一并补齐：

| 缺口 | 对策 |
|---|---|
| 无图形界面 → "功能完整性"只能落在 7-13 档 | 新增桌面界面（PySide6） |
| 系统是无状态批处理器 → 无法在线分析 | 新增逐帧会话接口（FastAPI + WebSocket） |

---

## 二、图形界面（`trafficvision_gui/`）

### 2.1 布局

三栏 + 底部：

| 区域 | 内容 |
|---|---|
| 左栏 | 输入视频、机位名、四个预判梯度、分辨率/置信度、静态结构通道与遮挡续推开关、跳帧、输出目录 |
| 中栏 | 视频预览与播放控制（播放/暂停、进度条、时间） |
| 右栏 | 本次运行指标卡（每帧可预判目标 / 1 秒中位误差 / P90 / 每帧目标数）、四梯度说明、报告导出 |
| 底部 | 开始预判、取消、进度条、实时日志 |

### 2.2 关键设计决定

**用子进程而不是线程跑任务。** 一次预判动辄几十秒到几分钟，界面绝不能卡住。
子进程还有一个副作用是好的：管线崩溃不会带走界面，日志也完整可见。

**配色与技术报告图表同源。** 四个预判梯度固定配色
（0.1 蓝 `#2060cc`、0.5 绿 `#208040`、1 橙 `#cc7020`、2 紫 `#8040cc`），
与 `paper/competition/figures` 里的图表一致，使"界面—图表—报告"是同一套视觉语言。

**不用现成的 Fluent 组件库。** PyQt-Fluent-Widgets 虽有 8.1k 星，但免费版为 GPL-3.0，
会再叠一层传染性许可，并且要新增一条依赖进《资源使用清单》；自写 QSS 主题约 300 行，
零新增许可义务，且能做到与图表配色严格一致。

**PySide6 而非 PyQt。** PySide6 是 LGPL-3.0，PyQt 是 GPL-3.0 或商业授权。
本项目整体 AGPL-3.0，但保留"后续闭源交付时替换检测器"这条路，因此不让 GUI 层
再被 GPL 绑死。

### 2.3 启动

**必须用项目自己的虚拟环境，并且工作目录在项目根目录。** 用系统 base 的 Python
会报 `No module named 'trafficvision_gui'`（PySide6 只装在项目 venv 里），
在用户目录下运行同样会找不到包。

最省事的入口是双击 `tools\启动界面.bat`；等价的手动命令是：

```powershell
cd C:\Users\<你>\Desktop\AI+software-videoprer
.\trafficlab-3d\env\Scripts\python.exe -m trafficvision_gui.main
```

首次运行前如提示缺依赖：

```powershell
.\trafficlab-3d\env\Scripts\python.exe -m pip install -r requirements-gui-api.txt
```

---

## 三、在线分析接口（`trafficvision_api/`）

### 3.1 为什么必须会话化

原系统的固定成本有两项：模型加载、静态结构全片扫描。二者都是"每段视频付一次"。
在线分析的先决条件是把它们改成"每进程 / 每机位付一次"。`StreamSession` 正是做这件事：
模型只加载一次，之后逐帧推入。

### 3.2 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 权重是否存在、可用梯度、活跃会话数、已有场景先验列表 |
| POST | `/api/session` | 创建会话 → `session_id`（含相机名、梯度、分辨率、置信度、跳帧等） |
| GET | `/api/session/{sid}` | 会话状态 |
| GET | `/api/session/{sid}/report` | 累计报告（帧数、轨迹数、平均单帧延迟） |
| DELETE | `/api/session/{sid}` | 结束会话并返回报告 |
| WS | `/ws/analyze?session={sid}` | 逐帧分析主通道 |

### 3.3 为什么用 WebSocket 而不是逐帧 HTTP

实测每帧预算只有 29～48 ms（见 `docs/27` 与跳帧实测），一次 HTTP 往返的开销就能吃掉它。
WebSocket 建连一次、之后只传帧与结果。

### 3.4 协议

客户端 → 服务端

* 二进制帧：一帧图像（JPEG/PNG 字节）
* 文本帧（JSON）控制指令：
  * `{"cmd":"t","value":1.23}` 指定该帧时间戳（秒）
  * `{"cmd":"stride","value":2}` 跳帧：每 N 帧分析一次，其余帧只回占位结果
  * `{"cmd":"horizon","value":1.0}` 切换预判梯度

服务端 → 客户端：文本帧（JSON）结构化结果。**不返回画好的图**，渲染交给前端，
这样带宽小、前端可自由叠加。

```json
{
  "frame_index": 123, "t": 4.10, "analyzed": true, "latency_ms": 29.1,
  "objects": [
    {"id": 7, "cls": "汽车", "conf": 0.91, "bbox": [1121, 545, 1363, 619],
     "centroid": [1242, 582], "speed_px_s": 258.2, "state": "variable",
     "maneuver": false, "predictable": true, "coasting": false,
     "preds": {"0.1": [1146, 544, 1391, 619], "0.5": [], "1.0": [], "2.0": []}}
  ],
  "stats": {"tracked": 28, "predictable": 14, "coasting": 3, "memory": 10}
}
```

### 3.5 启动与自测

双击 `tools\启动在线接口.bat`，或手动：

```powershell
cd C:\Users\<你>\Desktop\AI+software-videoprer
.\trafficlab-3d\env\Scripts\python.exe -m trafficvision_api.server --port 8600
```

启动后：健康检查 `http://127.0.0.1:8600/api/health`，
交互式文档 `http://127.0.0.1:8600/docs`（FastAPI 自带，可直接在网页上试各个端点）。

### 3.6 边界（如实说明）

* 本接口**不含**部署运维：没有鉴权、没有并发限流、没有进程守护。
  若要对外提供在线演示地址，这三项必须补上。
* 静态结构通道的**首次全片扫描**不在本接口内（它需要整段视频）；
  接口只做逐帧注入，因此上线前应先用某个机位跑一次缓存，或在
  `outputs/scene_prior/<机位>.json` 预置。
* 场景先验命中与否可在 `/api/health` 的 `scene_priors` 字段直接看到。

---

## 四、验收方式（可核验，不靠声明）

| 项 | 怎么验 |
|---|---|
| 界面可构建 | `QT_QPA_PLATFORM=offscreen python -c "from trafficvision_gui.main import MainWindow; ..."` 构建并销毁主窗口 |
| 接口可用 | `GET /api/health` 返回 `ok: true` 且列出可用梯度与场景先验 |
| 逐帧结果正确 | 建会话后推一帧，返回 JSON 中的 `objects[].preds` 含四个梯度、`stats` 计数与画面一致 |
| 会话化生效 | 连续推 N 帧，`/report` 的 `frames_analyzed` 与推送帧数一致，且**模型只加载一次**（进程启动后无重复加载日志） |

---

## 五、后续可做

1. 界面接入在线接口：新增"在线分析"页，直接用 `/ws/analyze` 做实时预览
2. 四梯度同屏对照视图（界面上的第二个视图，把四个梯度叠在同一帧）
3. 一键打包为可安装包（PyInstaller / Inno Setup），对应赛道一"可安装的软件包"
4. 接口补鉴权与限流，用于对外在线演示
