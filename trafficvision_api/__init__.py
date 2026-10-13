"""在线分析接口层：把批处理管线包装成可常驻的逐帧会话。"""

from .session import StreamSession, SessionConfig  # noqa: F401

__all__ = ["StreamSession", "SessionConfig"]
