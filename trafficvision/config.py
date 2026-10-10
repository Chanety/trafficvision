"""全局配置：类别、颜色、预判梯度与有效性阈值。"""

from __future__ import annotations

# 支持的预判时间梯度（秒）
HORIZONS: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0)

# 默认目标类型（基础类别表，12 类）
#
# ⚠️ 0~8 的顺序**绝对不能改动**：已有标注（videoset_01_label、videoset_02_label）
# 用的就是这套编号。新增类别只能往后追加。
#
# 9~11 是 2026-10-06 按用户要求新增的：
#   · roadlamp / trafficlight —— 路灯、信号灯单独成类（原来它们没有位置可放，
#     只能被误判成车辆或行人）
#   · rider —— 非机动车驾驶员：骑车人原来按"先载具后行人"规则被删除，
#     现在改为单独成类，与纯步行的 person 区分开
DEFAULT_CLASSES: tuple[str, ...] = (
    "car",          # 0
    "truck",        # 1
    "bus",          # 2
    "motorcycle",   # 3
    "bicycle",      # 4
    "person",       # 5
    "sign",         # 6
    "bollard",      # 7
    "tricycle",     # 8
    "roadlamp",     # 9
    "trafficlight", # 10
    "rider",        # 11
)

# ------------------------------------------------------------------ 合并方案
# 用户要求：摩托车 / 自行车 / 三轮车统一为"非机动车"，不再区分。
# 合并只发生在"输出与训练时"，原始 12 类标注一个字节都不改，随时可回退。
MERGED_CLASSES: tuple[str, ...] = (
    "car",          # 0
    "truck",        # 1
    "bus",          # 2
    "nonmotor",     # 3  ← 摩托车 + 自行车 + 三轮车
    "person",       # 4
    "sign",         # 5
    "bollard",      # 6
    "roadlamp",     # 7
    "trafficlight", # 8
    "rider",        # 9  非机动车驾驶员
)

# 基础类别编号 → 合并方案编号
MERGE_MAP: dict[int, int] = {
    0: 0, 1: 1, 2: 2,
    3: 3, 4: 3, 8: 3,      # motorcycle / bicycle / tricycle → nonmotor
    5: 4, 6: 5, 7: 6,
    9: 7, 10: 8, 11: 9,
}

# 静态目标类（相机不动时位置恒定，适合"整段视频回档确认 + 隐标记"）
STATIC_CLASSES: tuple[str, ...] = ("sign", "bollard", "roadlamp", "trafficlight")

# 两三轮载具：判定"骑乘状态"时的载具候选
NONMOTOR_IDS: frozenset[int] = frozenset({3, 4, 8})

# 开放词汇模型只能按英文单词匹配，没法直接输出 "nonmotor"；
# 所以走"先按摩托车/自行车/三轮车检测，再把类别名映射成 nonmotor"的路子，
# 保证无论用哪条权重路径，产品输出都是合并后的类别。
MERGE_NAME_MAP: dict[str, str] = {
    "motorcycle": "nonmotor",
    "bicycle": "nonmotor",
    "tricycle": "nonmotor",
}

# 开放词汇模型的提示词只覆盖 0~8（9 类）。9~11 由静态通道与骑乘判定补齐。
WORLD_PROMPT_CLASSES: tuple[str, ...] = DEFAULT_CLASSES[:9]


def _read_schema_tables():
    """从 schemas/classes_v3_10.yaml 读唯一权威类别表。

    该文件是类别的唯一来源；上面的内置常量只是"schema 读不到时的兜底"，
    保证任何环境都能 import 成功。读得到就以 schema 为准，避免两处真相打架。
    """
    try:
        import yaml

        path = Path(__file__).resolve().parents[1] / "schemas" / "classes_v3_10.yaml"
        d = yaml.safe_load(path.read_text(encoding="utf-8"))
        base = tuple(str(d["base_classes"][i]) for i in sorted(d["base_classes"]))
        merged = tuple(str(d["classes"][i]) for i in sorted(d["classes"]))
        mapping = {int(k): int(v) for k, v in d["map_from_base12"].items()}
        if base and merged and mapping:
            return base, merged, mapping
    except Exception:
        pass
    return None


_SCHEMA_TABLES = _read_schema_tables()
if _SCHEMA_TABLES:
    DEFAULT_CLASSES, MERGED_CLASSES, MERGE_MAP = _SCHEMA_TABLES
SCHEMA_VERSION = "v3"          # 与 schemas/classes_v3_10.yaml 保持一致

# 每个类别的绘制颜色（BGR）
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "car": (80, 220, 80),
    "truck": (0, 150, 255),
    "bus": (255, 130, 40),
    "motorcycle": (220, 60, 220),
    "bicycle": (220, 220, 60),
    "person": (60, 220, 255),
    "sign": (255, 255, 255),
    "bollard": (150, 150, 255),
    "tricycle": (0, 215, 255),
    "roadlamp": (0, 200, 200),      # 路灯：青
    "trafficlight": (60, 60, 255),  # 信号灯：红
    "rider": (255, 200, 0),         # 非机动车驾驶员：蓝绿
    # 合并类别：摩托车/自行车/三轮车统一按"非机动车"显示（v7 起的 7 类方案）
    "nonmotor": (220, 60, 220),
}
DEFAULT_COLOR: tuple[int, int, int] = (180, 180, 180)

# 画面显示名：内部类别名保持 ASCII（配置、文件名、JSON 都用它），
# 只有画到视频上的文字换成中文。（实测 OpenCV 5 可正常渲染中文字形）
CLASS_DISPLAY_NAMES: dict[str, str] = {
    "car": "汽车",
    "truck": "卡车",
    "bus": "公交",
    "nonmotor": "非机动车",
    "person": "行人",
    # 用户 2026-10-10：sign 这一类实际同时吃下交通标志、店招与广告牌，
    # 叫"标志牌"指代不清，改为把两类都写出来。**只改显示名**，
    # 类别键 sign 与编号 0~8 顺序都不动，已有标注与模型不受影响。
    "sign": "标志提示牌与广告牌",
    "bollard": "路桩",
    "motorcycle": "摩托",
    "bicycle": "自行车",
    "tricycle": "三轮车",
    "roadlamp": "路灯",
    "trafficlight": "信号灯",
    "rider": "非机动车驾驶员",
}


def display_name(cls_name: str) -> str:
    """类别显示名；未登记的直接返回原名。"""
    return CLASS_DISPLAY_NAMES.get(cls_name, cls_name)

# 预判有效性门槛
MIN_TRACK_AGE = 5          # 连续观测帧数不足则不给预判
MIN_SPEED_PX_S = 8.0       # 近乎静止的目标不给预判

# ---------------------------------------------------------------- 最小目标尺寸（按类独立）
# 用户 2026-10-08：
#   "对于行人和车辆的标准应该是独立的，行人和非机动车应当适当宽容，
#    最好有一种粗方法来决定以最小的综合开销进行这种筛选与过滤。"
#
# 粗方法（脚本 bench/min_size_sweep.py，可复现）：**用时序一致性当"是不是真目标"的代理**。
# 真目标在相邻帧会稳定复现，噪声不会。逐类统计各尺寸档的稳定率，
# 取稳定率达标的最小尺寸作为该类门槛。
# 实测（videoset_02 前 3 段 × 150 帧、v7@1280、conf 0.10，共 17119 框，整体稳定率 90.8%）：
#
#   类别        12px 以下   14~18px    32px 以上    数据说明
#   car          44~80%     85~88%     92.8%       10px 以上即可
#   person       83%        81~86%     93.8%       **小框大多是真人**，不该一刀切
#   nonmotor     0~60%      60~74%     94.6%       **18px 以下确实不可靠**
#   bollard      98~100%    96~100%    95.4%       静态结构小但极稳，不该收
#   sign         —          —          99.5%       全部落在 40px 以上
#
# 关键发现：**真正需要收紧的是非机动车，不是行人**——行人的小检测大多是远处真人，
# 而非机动车在 18px 以下是真实噪声。统一门槛会两头不讨好。
MIN_SIDE_BY_CLASS: dict[str, float] = {
    # 行人 / 骑乘者：最宽容（漏掉真人的代价高，且实测小框可信）
    "person": 8.0,
    "rider": 8.0,
    # 两三轮：实测 18px 以下时序稳定率最低
    "nonmotor": 14.0,
    "motorcycle": 14.0,
    "bicycle": 14.0,
    "tricycle": 14.0,
    # 机动车：车身大，几乎不受影响，设下限挡极端噪声
    "car": 10.0,
    "truck": 14.0,
    "bus": 14.0,
    # 静态结构：尺寸小但稳定率高（路桩 14~16px 档仍有 96~98%），不额外收紧
    "sign": 10.0,
    "bollard": 10.0,
    "roadlamp": 10.0,
    "trafficlight": 10.0,
}
DEFAULT_MIN_SIDE_PX = 10.0   # 未登记类别的兜底
# 面积兜底：只挡"很长但极薄"的退化框（如 60×2 px），不承担主判据
MIN_TARGET_AREA_PX = 120.0
MIN_TARGET_SIDE_PX = DEFAULT_MIN_SIDE_PX   # 兼容旧签名（早期只有一个全局门槛）

# NMS 阈值（同类框合并的松紧）。
# 实测 2026-10-07：ultralytics 默认 0.7 在 imgsz=1280 下压不住"同一辆车出两个框"——
# 6 段视频 1800 帧里 87% 的帧存在同类高重叠框（IoU>0.5），画面上表现为一辆车两个框。
# 收到 0.5 后重复框基本消失（见 docs\18）。
NMS_IOU = 0.5


def min_side_for(cls_name: str | None) -> float:
    """取该类别的短边门槛。"""
    if not cls_name:
        return DEFAULT_MIN_SIDE_PX
    return MIN_SIDE_BY_CLASS.get(cls_name, DEFAULT_MIN_SIDE_PX)


# ------------------------------------------------- 静态通道：细长结构豁免
# 灯杆 / 信号灯杆天然是细长条，短边经常不到 10px，会被"过小不识别"整根丢掉。
# 实测 VID_20261004_112416 前 150 帧：静态通道里被判过小直接丢掉的检测有 731 个，
# 最终只聚成 44 个簇——这就是用户反馈"图20 路灯未识别"的直接原因。
# 只豁免"确实细长"的形状（长边够长 + 高宽比够大 + 面积仍非噪声），其余照旧过滤。
STATIC_THIN_CLASSES: tuple[str, ...] = ("roadlamp", "trafficlight")
STATIC_THIN_MIN_ASPECT = 3.0        # 长边 / 短边
STATIC_THIN_MIN_LONG_SIDE = 40.0    # 长边下限（像素）
STATIC_THIN_MIN_AREA = 120.0        # 豁免后仍要满足的面积下限


def static_thin_exempt(box, cls_name: str | None) -> bool:
    """该静态检测是否属于"细长结构"（灯杆一类），从而豁免短边/面积门槛。"""
    if not cls_name or cls_name not in STATIC_THIN_CLASSES:
        return False
    w = float(box[2] - box[0])
    h = float(box[3] - box[1])
    long_side, short_side = max(w, h), min(w, h)
    if long_side < STATIC_THIN_MIN_LONG_SIDE:
        return False
    if short_side <= 0 or long_side / short_side < STATIC_THIN_MIN_ASPECT:
        return False
    return (w * h) >= STATIC_THIN_MIN_AREA


# 灯杆的"短边上限"：本机位（1080p 固定机位）实测真灯杆短边在 12~41px，
# 而画在楼体立面上的那条误检短边是 64px。零样本图文匹配分不开这两种
# （把框平移一格再问，建筑竖条照样得 0.66 分，真灯杆 0.54 分，见 docs/26），
# 所以补一条几何约束，取两者之间。只对 roadlamp 生效；
# 信号灯的横梁本来就宽，不适用这条。
ROADLAMP_MAX_SHORT_SIDE = 50.0


def target_too_small(box, cls_name: str | None = None,
                     min_side: float | None = None,
                     min_area: float | None = None) -> bool:
    """该检测框是否小到不值得识别（宽或高任一边为 0 也算）。

    优先级：显式 min_side（CLI 覆盖） > 按类门槛 > 兜底值；
    min_area 只作为退化框的兜底。
    """
    w = box[2] - box[0]
    h = box[3] - box[1]
    if w <= 0 or h <= 0:
        return True
    if min_side is not None:
        side = float(min_side)
    else:
        side = min_side_for(cls_name)
    area = MIN_TARGET_AREA_PX if min_area is None else float(min_area)
    return min(w, h) < side or (w * h) < area

# 片段时长约束（秒）
MIN_CLIP_SECONDS = 2.0
MAX_CLIP_SECONDS = 30.0

# 尾迹长度（秒）
TRAIL_SECONDS = 1.0

# 运动模型：cv = 图像匀速；ca = 匀加速；scale = 维度判定归一化匀速；auto = 按梯度分工
#
# 实测（videoset_02 6 段 × 300 帧，7 类模型，同一批目标配对比较，中位/均值 px）：
#   0.5 秒  cv 16.4/26.8  ｜ auto(旧,0.5 起用 scale) 17.8/31.9
#   1   秒  cv 34.2/56.9  ｜ auto(旧)                34.0/62.8
#   2   秒  cv 78.2/133.6 ｜ scale 75.2/131.4（只有 2 秒档维度判定略优）
#   ca 匀加速：0.5 秒 31.3/48.0、1 秒 88.2/134.2、2 秒 285.5/447.8 —— 差 3~8 倍，别用
# 因此 auto 只在 2 秒档启用维度判定，其余梯度用匀速；默认就用 auto。
MOTION_MODEL = "auto"

# 跟踪器：在 videoset_01 全集（14 段，排除手持段）实测后确定
# 结论：tracktrack 虽把轨迹条数降低 79%，但"每帧可预判目标数"下降 44.6%、
# 误差反而变差（≈58.9 → ≈81.5 px），因此不用它做默认。
# fasttrack 在 3 段横评中与 bytetrack 召回相当、误差略优，列为待全集验证的候选。
TRACKER = "bytetrack.yaml"

# 静止目标状态判定（constant / variable）
STILL_WINDOW = 6          # 连续静止帧数
STILL_THRESHOLD_PX = 3.0  # 位置波动阈值（像素）
STILL_BIAS_PX = 2.0       # bias 允许值（迟滞带宽，吸收检测噪声与视角误差）
# 净位移线索（用户 2026-10-10 抽检反馈"行人近点目标未识别"）：
# 只判"窗口内波动范围"会把走得很慢的目标当成静止——实测 112254 第 24 帧有
# 18 条行人轨迹里 14 条被藏掉，其中几条的估计速度是 25~40 px/s。
# 波动的毛病是"抖动也会撑大范围"，所以再加一条**净位移**：
# 窗口前 2 帧均值 → 后 2 帧均值，抖动会互相抵消，真移动不会。
# 实测（同一段素材，被隐藏的观测）：
#   行人 净位移中位 2.16px，13.9% 超过 5px；汽车 中位 0.79px，仅 0.2% 超过 5px。
# 所以取 5px：能救回约 14% 被误藏的行人，而停放车辆几乎不受影响。
STILL_NET_PX = 5.0        # 窗口内净位移超过该值 → 判定为在移动

# ---------------------------------------------------------------- 跟踪与遮挡
# 轨迹跳变防护：路口场景实测 7.8% 的样本是"轨迹突然跳到别的目标"（串轨），
# 它们吃掉 25% 的预判误差总量。判据要同时满足"位移离谱"与"框几乎不重叠"，
# 避免把真的急加速/急转弯误判成串轨。
JUMP_MIN_PX = 40.0        # 单帧位移绝对下限（像素）
JUMP_STEP_FACTOR = 4.0    # 相对该轨迹步长中位数的倍数
JUMP_IOU_MAX = 0.1        # 与上一帧框的 IoU 上限，低于它才算"换了个目标"

# 遮挡续推：目标被遮挡（跟踪器不再输出）后，继续用运动模型外推显示多少帧。
# 0 表示关闭。路口实测断档很常见（约 30% 的预判样本窗口内有过缺失），
# 续推能让画面上的框穿过去，而不是"消失再出现"。
COAST_FRAMES = 12

# 机动判定：用最近一小段轨迹估转角速度，判断目标在不在转弯/变道。
# 实测（新素材 1 秒梯度，移动目标）：直行中位误差 35 px，缓弯 42，转弯 49，急转 62 px，
# 所以转角速度是"该不该相信这条预判"的有效信号。
# 注意：实测把估计转角直接用于外推（圆弧/阻尼转角）反而更差——
# 转角估计的噪声比转弯本身的效应还大，因此这里只用它做置信度提示，不改外推。
MANEUVER_WINDOW_S = 0.6        # 估计转角用的历史窗口
MANEUVER_TURN_DEG_S = 15.0     # 转角速度门槛：超过即视为机动
MANEUVER_DEV_PX = 25.0         # 或"转弯在目标梯度内偏离直线"超过该像素数也算机动
# 窗口内位移低于该值时不给机动判定：实测门槛放到 20px 时，
# 被判"机动"的目标转角中位数会飙到 157°/s（P90 823°/s）——那是检测噪声不是转弯。
# 提到 60px（≈100px/s）后估计才稳定。
MANEUVER_MIN_MOVE_PX = 60.0


# ------------------------------------------------------- 静态目标地图（回档查看）
# 用户反馈：并排的几根路桩，每次只有一部分被检出；同一根路桩"这一帧有、
# 有人走过去之后就没有了"。原因是检测分数正好压在阈值附近，随光照/遮挡抖动。
#
# 机制（对应"回档查看 + 隐标记"）：
#   1. 整段视频先扫一遍静态目标（多提示词，低阈值，召回优先）；
#   2. 同一位置跨帧聚成一簇，统计"未被遮挡的帧里有多少帧被检出"；
#   3. 比例 ≥ STATIC_MAP_MIN_RATIO 且至少命中 STATIC_MAP_MIN_FRAMES 帧
#      → 该位置被确认为静态目标，写入"隐标记"；
#   4. 之后任何一帧只要没遮挡、但检测器没输出，就把隐标记补回画面
#      （用虚线框 + "·记忆"标注，且不做预判——它是静止的，预测没有意义）。
#
# 为什么用 0.8：用户给的经验值"非阻挡状态下八成多都能识别"。
STATIC_MAP_MIN_RATIO = 0.80
STATIC_MAP_MIN_FRAMES = 4
# 低于这个 IoU 视为"另一根柱子"（并排路桩间距小，阈值不能太高）
STATIC_MAP_MATCH_IOU = 0.25
# 动态目标盖住静态目标多少面积算"被遮挡"
STATIC_MAP_OCCLUDE_COVER = 0.50
# 同一目标的碎片合并：提示词打在"灯头 / 灯杆 / 灯箱"上会各自成簇，
# 需要按 IoU 或中心距把它们并回一个目标
STATIC_MAP_MERGE_IOU = 0.15
STATIC_MAP_MERGE_DIST = 0.50     # × 较大框的长边
# 小于这个面积的检测直接丢掉：实测 10×17px 这种量级的框全是噪声
STATIC_MAP_MIN_AREA = 400.0

# 静态通道的检测设置：召回优先，宁滥勿缺，由第 3 步的"跨帧投票"去伪。
# 0.05 是实测下来的点：112222 最左边那根被画面切掉一半的路桩，
# 开放词汇模型只有 0.02~0.04 分，用 0.08 会永远漏掉它、用 0.05 能稳定投出来；
# 而假阳性交给投票否决（conf 0.05 与 0.08 的被否决簇数量几乎一样多）。
STATIC_CONF = 0.05
STATIC_IMGSZ = 1280
# 每 N 帧扫一次静态通道。静态目标不随时间变化，抽样不影响正确性——
# 一次扫描的结果会覆盖它代表的 N 帧窗口；而本机实测该通道约 0.2 s/帧，
# 抽样 3 倍是把端到端耗时压回可接受范围的关键。
STATIC_STRIDE = 3

# 开放词汇提示词集成。
#
# 关键实测（bench/bollard_probe.py，VID_20261004_112222 / 112254）：
#   同一个混凝土路桩，提示词 "bollard" 只给 0.02~0.16 分，
#   换成 "concrete bollard" 给 0.55~0.91 分，"barrier post" 给 0.33~0.66 分。
#   标注流程用的是单词 "bollard" + conf=0.15 → 左边几根全部被丢弃，
#   于是微调模型从来没见过它们，永远学不会（自我强化的漏标）。
# 因此这里一个概念配多个提示词，同一概念内部再做一次 NMS 去重。
STATIC_PROMPTS: dict[str, tuple[str, ...]] = {
    "bollard": ("bollard", "concrete bollard", "barrier post", "traffic bollard"),
    "sign": ("traffic sign", "road sign", "sign"),
    "roadlamp": ("street light", "street lamp", "lamp post"),
    "trafficlight": ("traffic light", "traffic signal"),
}
