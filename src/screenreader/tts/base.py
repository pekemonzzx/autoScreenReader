"""TTS 后端抽象。

为什么要抽象：现在用 GPT-SoVITS 的 HTTP API（torch + CUDA 运行时，约 3GB），
P3 计划导出 ONNX 后换成进程内推理（约 700MB，且能用 DirectML 跑任意显卡）。
上层流水线不应该关心这件事。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class StreamFormat:
    samplerate: int
    channels: int
    sampwidth: int


class TtsBackend:
    """TTS 后端接口。

    两种用法：
      * synthesize(text) -> 完整 WAV 字节（简单、适合非实时）
      * synthesize_stream(text) -> 逐块 PCM（低首包延迟，实测 667ms vs 1786ms）
    """

    name: str = "base"

    def health(self) -> bool:
        raise NotImplementedError

    def synthesize(self, text: str, **kw) -> bytes:
        raise NotImplementedError

    def synthesize_stream(self, text: str, **kw) -> Iterator[bytes]:
        """产出裸 PCM 块。调用方需要先读 `self.stream_format` 拿到采样率。"""
        raise NotImplementedError

    @property
    def stream_format(self) -> Optional[StreamFormat]:
        return None

    def close(self) -> None:
        pass
