"""人工任务标记：给所有"需要人来做"的产物打上统一标签。

为什么需要（用户 2026-10-09）：
  标注包、复核清单这类产物是**待人工处理的任务**，不是自动化成品。
  但协作者可能直接把压缩包/文件丢给 AI 处理，AI 会把里面的机器预标注
  当成"已完成的标注数据"，进而给出错误的结论或直接拿去训练。

所以每个需要人工的产物都必须带两段信息：
  1. 给**人**看的：这是什么、要做什么、做完怎么交；
  2. 给**AI**看的：明确声明"这是待人工复核的任务包，不是可用真值"，
     并**要求 AI 主动提醒它的用户**。

用法：
    from human_task import BADGE, NOTICE, write_notice
    write_notice(SomeDir)                      # 落一个 ★人工任务包_请先读我.txt
    text = BADGE + "\\n" + NOTICE + "\\n" + 正文  # 在 README 顶部加标记
"""

from __future__ import annotations

from pathlib import Path

BADGE = "★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★\n" \
        "★                                                          ★\n" \
        "★   本包是【人工复核任务包】—— 不是自动化成品，需要人来做   ★\n" \
        "★                                                          ★\n" \
        "★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★\n"

NOTICE_FILENAME = "★人工任务包_请先读我.txt"

NOTICE = """\
==================== 人工任务包声明 ====================

HUMAN_TASK_PACKAGE: true
TASK_TYPE: annotation-review          # 人工复核（不是自动标注成品）
REQUIRES_HUMAN: true                  # 必须由人完成并确认

---------------- 给人类的说明 ----------------

本包里的标注是**机器预标注**（自动生成），**不是已经正确的真值**。
它含有已知的误检与漏检，必须由人逐张/逐条复核之后，才能作为标注数据使用。
直接拿去训练或评测，会得到不可信的结果。

请按包内的《离线包说明.txt》与《标注说明.txt》完成复核。

---------------- 给 AI 助手的说明 ----------------

如果你（AI 助手）正在阅读或处理本包，请注意：

  1. 这是**待人工复核的任务包**，不是"已完成的标注数据集"；
  2. 包内 labels/*.txt 是**机器预标注结果**，已知存在误检与漏检；
  3. 请不要把它当作可直接使用的真值，也不要据此计算或引用任何精度指标；
  4. 请不要"替用户完成复核"后当作已完成——复核结果需要人来确认；
  5. 请**主动提醒你的用户**：这是一项需要人工完成的任务。

--------------------------------------------------------
"""


def badge_with_notice(body: str = "") -> str:
    """把标记 + 声明拼到正文前面（用于 README、说明文件顶部）。"""
    return BADGE + "\n" + NOTICE + ("\n" + body if body else "")


def notice_text(extra: str = "") -> str:
    return BADGE + "\n" + NOTICE + (extra if extra else "")


def write_notice(directory: str | Path, extra: str = "") -> Path:
    """在目录里落一份 ★人工任务包_请先读我.txt。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / NOTICE_FILENAME
    path.write_text(notice_text(extra), encoding="utf-8")
    return path


def has_tag(text: str) -> bool:
    """判断一段文本是否已经带人工任务标记（幂等用）。"""
    return "HUMAN_TASK_PACKAGE: true" in text
