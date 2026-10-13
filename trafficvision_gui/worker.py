"""离线分析任务：以子进程方式调用 trafficvision.cli，避免长任务卡住界面。

用子进程而不是线程，有三个好处：
  1. 界面永远不会被 GIL 或 CUDA 初始化阻塞，交互始终流畅；
  2. 进度与日志可以通过标准输出实时拿到，出问题也看得见；
  3. 不动管线代码，风险最低。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, Signal

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


class PipelineJob(QObject):
    """跑一次预判出视频，产出 mp4 + 精度报告 json。"""

    log = Signal(str)
    started = Signal()
    finished = Signal(bool, str)      # (成功?, 说明)
    metrics = Signal(dict)            # 逐梯度指标

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.proc: QProcess | None = None
        self.out_video: Path | None = None
        self.out_json: Path | None = None

    # ------------------------------------------------------------------ 启动
    def start(self, video: Path, horizon: float, out_dir: Path,
              camera: str | None = None, imgsz: int = 1280, conf: float = 0.25,
              static_map: bool = True, coast: int = 12, guard: bool = True,
              max_frames: int | None = None, frame_stride: int = 1) -> None:
        if self.proc is not None and self.proc.state() != QProcess.NotRunning:
            self.log.emit("[跳过] 已有任务在运行")
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = video.stem
        self.out_video = out_dir / f"{stem}_h{horizon:g}.mp4"
        self.out_json = out_dir / f"{stem}_h{horizon:g}.json"

        args = ["-m", "trafficvision.cli",
                "--video", str(video), "--horizon", f"{horizon:g}",
                "--out", str(self.out_video), "--json", str(self.out_json),
                "--imgsz", str(imgsz), "--conf", str(conf),
                "--coast", str(coast), "--frame-stride", str(frame_stride)]
        if camera:
            args += ["--camera", camera]
        if not static_map:
            args.append("--no-static-map")
        if not guard:
            args.append("--no-guard")
        if max_frames:
            args += ["--max-frames", str(max_frames)]

        self.log.emit("$ " + " ".join(args[:6]) + " …")
        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(str(ROOT))
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_output)
        self.proc.finished.connect(self._on_finished)
        self.proc.errorOccurred.connect(
            lambda e: self.finished.emit(False, f"进程启动失败：{e}"))
        self.started.emit()
        self.proc.start(PY, args)

    def cancel(self) -> None:
        if self.proc is not None and self.proc.state() != QProcess.NotRunning:
            self.proc.kill()
            self.log.emit("[已取消]")

    def busy(self) -> bool:
        return self.proc is not None and self.proc.state() != QProcess.NotRunning

    # ------------------------------------------------------------------ 回调
    def _on_output(self) -> None:
        if self.proc is None:
            return
        raw = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        for line in raw.splitlines():
            if line.strip():
                self.log.emit(line.rstrip())

    def _on_finished(self, code: int, _status) -> None:
        if code != 0:
            self.finished.emit(False, f"任务失败（退出码 {code}）")
            return
        if self.out_json and self.out_json.exists():
            try:
                payload = json.loads(self.out_json.read_text(encoding="utf-8"))
                self.metrics.emit(payload.get("metrics", {}) or {})
            except Exception as exc:
                self.log.emit(f"[警告] 报告解析失败：{exc}")
        ok = bool(self.out_video and self.out_video.exists())
        self.finished.emit(ok, str(self.out_video) if ok else "未生成输出视频")
