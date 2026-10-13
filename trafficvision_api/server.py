"""在线分析 HTTP/WebSocket 服务。

启动::

    python -m trafficvision_api.server --port 8600

接口一览
--------
  GET    /api/health                 健康检查（权重、设备、场景先验）
  POST   /api/session                创建会话 → session_id
  GET    /api/session/{sid}          会话状态
  GET    /api/session/{sid}/report    累计报告
  DELETE /api/session/{sid}          结束会话
  WS     /ws/analyze?session={sid}   逐帧分析主通道

WebSocket 协议
--------------
客户端 → 服务端
  · 二进制帧   : 一帧图像（JPEG/PNG 字节，cv2.imdecode 解码）
  · 文本帧(JSON): 控制指令
      {"cmd":"t","value":1.23}      指定该帧时间戳（秒）
      {"cmd":"stride","value":2}    跳帧：每 N 帧分析一次，其余帧只回占位结果
      {"cmd":"horizon","value":1.0} 切换预判梯度
服务端 → 客户端
  · 文本帧(JSON): 该帧结构化结果（不返回画好的图，渲染交给前端）
      {"frame_index":…, "t":…, "objects":[…], "stats":{…}, "latency_ms":…}
  · 出错时: {"error":"…"}
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from trafficvision.config import HORIZONS

from .session import DEFAULT_WEIGHTS, SessionConfig, StreamSession

ROOT = Path(__file__).resolve().parents[1]
SCENE_PRIOR_DIR = ROOT / "outputs" / "scene_prior"

app = FastAPI(title="交通视频短时预判 · 在线分析接口", version="0.1.0")
SESSIONS: dict[str, StreamSession] = {}


class SessionRequest(BaseModel):
    camera: str | None = Field(default=None, description="机位名，用于复用场景先验")
    horizon: float = Field(default=1.0, description="预判梯度，须属于 0.1/0.5/1/2")
    imgsz: int = 1280
    conf: float = 0.25
    device: str = "0"
    coast: int = 12
    guard: bool = True
    fps: float = 30.0
    width: int = 1920
    height: int = 1080
    weights: str | None = None


def _cfg(req: SessionRequest) -> SessionConfig:
    if req.horizon not in HORIZONS:
        raise HTTPException(400, f"prejudge horizon must be one of {HORIZONS}")
    weights = req.weights or str(DEFAULT_WEIGHTS)
    if not Path(weights).exists():
        raise HTTPException(400, f"weights not found: {weights}")
    return SessionConfig(weights=weights, horizon=float(req.horizon),
                         imgsz=int(req.imgsz), conf=float(req.conf),
                         device=req.device, coast=int(req.coast), guard=bool(req.guard),
                         fps=float(req.fps), frame_w=int(req.width),
                         frame_h=int(req.height), camera=req.camera)


@app.get("/api/health")
def health() -> dict[str, Any]:
    priors = sorted(p.stem for p in SCENE_PRIOR_DIR.glob("*.json")) \
        if SCENE_PRIOR_DIR.exists() else []
    return {
        "ok": Path(DEFAULT_WEIGHTS).exists(),
        "default_weights": DEFAULT_WEIGHTS.name,
        "weights_exists": Path(DEFAULT_WEIGHTS).exists(),
        "horizons": list(HORIZONS),
        "active_sessions": len(SESSIONS),
        "scene_priors": priors,
    }


@app.post("/api/session")
def create_session(req: SessionRequest) -> dict[str, Any]:
    try:
        sess = StreamSession(_cfg(req))
    except HTTPException:
        raise
    except Exception as exc:  # 模型加载失败（缺权重/无 GPU）要如实报出
        raise HTTPException(500, f"failed to init session: {type(exc).__name__}: {exc}")
    sid = uuid.uuid4().hex[:12]
    SESSIONS[sid] = sess
    return {"session_id": sid, "config": sess.cfg.as_dict()}


@app.get("/api/session/{sid}")
def session_state(sid: str) -> dict[str, Any]:
    return _get(sid).report()


@app.get("/api/session/{sid}/report")
def session_report(sid: str) -> dict[str, Any]:
    return _get(sid).report()


@app.delete("/api/session/{sid}")
def close_session(sid: str) -> dict[str, Any]:
    sess = SESSIONS.pop(sid, None)
    if sess is None:
        raise HTTPException(404, "session not found")
    return {"closed": sid, "report": sess.report()}


def _get(sid: str) -> StreamSession:
    sess = SESSIONS.get(sid)
    if sess is None:
        raise HTTPException(404, "session not found")
    return sess


@app.websocket("/ws/analyze")
async def ws_analyze(ws: WebSocket, session: str) -> None:
    sess = SESSIONS.get(session)
    await ws.accept()
    if sess is None:
        await ws.send_text(json.dumps({"error": "session not found"}, ensure_ascii=False))
        await ws.close()
        return
    stride = 1
    counter = 0
    manual_t: float | None = None
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("text") is not None:
                try:
                    cmd = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await ws.send_text(json.dumps({"error": "bad json"}, ensure_ascii=False))
                    continue
                if cmd.get("cmd") == "stride":
                    stride = max(1, int(cmd.get("value", 1)))
                elif cmd.get("cmd") == "t":
                    manual_t = float(cmd.get("value"))
                elif cmd.get("cmd") == "horizon":
                    sess.cfg.horizon = float(cmd.get("value")) if cmd.get("value") in HORIZONS \
                        else sess.cfg.horizon
                await ws.send_text(json.dumps({"ack": cmd.get("cmd")}, ensure_ascii=False))
                continue
            data = msg.get("bytes")
            if data is None:
                continue
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                await ws.send_text(json.dumps({"error": "decode failed"}, ensure_ascii=False))
                continue
            analyze = (counter % stride == 0)
            t = manual_t if manual_t is not None else counter / max(sess.cfg.fps, 1e-6)
            result = sess.push(frame, t, analyze=analyze)
            counter += 1
            await ws.send_text(json.dumps(result, ensure_ascii=False))
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


@app.exception_handler(HTTPException)
async def _http_err(_request, exc: HTTPException) -> JSONResponse:
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


def main() -> None:
    ap = argparse.ArgumentParser(description="在线分析接口服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()
    import uvicorn

    uvicorn.run("trafficvision_api.server:app" if args.reload else app,
                host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
