"""音频输出：设备枚举 + 播放。

直播场景要输出到 **VB-CABLE 的 `CABLE Input`**（在 OBS 里再把 `CABLE Output` 设为麦克风源），
所以设备是**按名字子串匹配**选的，不是硬编码索引——不同机器索引不一样。

流式播放是重点：每收到一块 PCM 就推进输出缓冲，**第一块到达就开始出声**。
这正是 TTS 流式（首包 667ms vs 整段 1786ms）能兑现价值的地方。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Iterable, List, Optional

log = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    index: int
    name: str
    channels: int
    default_samplerate: int

    def __str__(self) -> str:
        return f"[{self.index}] {self.name}  ({self.channels}ch @ {self.default_samplerate}Hz)"


def list_output_devices() -> List[DeviceInfo]:
    import sounddevice as sd

    out: List[DeviceInfo] = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_output_channels", 0) > 0:
            out.append(
                DeviceInfo(
                    index=i,
                    name=str(d["name"]),
                    channels=int(d["max_output_channels"]),
                    default_samplerate=int(d.get("default_samplerate") or 0),
                )
            )
    return out


def find_output_device(name_substr: str) -> Optional[DeviceInfo]:
    """按名字子串（大小写不敏感）找输出设备。

    直播用：``find_output_device("CABLE Input")``
    """
    if not name_substr:
        return None
    needle = name_substr.lower()
    candidates = list_output_devices()
    # 精确优先，其次前缀，最后子串
    for d in candidates:
        if d.name.lower() == needle:
            return d
    for d in candidates:
        if d.name.lower().startswith(needle):
            return d
    for d in candidates:
        if needle in d.name.lower():
            return d
    return None


def describe_default_device() -> str:
    import sounddevice as sd

    try:
        info = sd.query_devices(kind="output")
        return str(info.get("name", "?"))
    except Exception:
        return "（无默认输出设备）"


class Player:
    """把音频送到指定设备。

    线程模型：一个后台线程消费队列，主线程调用 `submit()` 即可非阻塞投递。
    `play_*` 系列是同步阻塞版本，方便测试与命令行使用。
    """

    def __init__(
        self,
        device: str = "",
        latency: str = "low",
        blocksize: int = 0,
        volume: int = 100,
    ) -> None:
        self.device_name = device
        self.latency = latency
        self.blocksize = blocksize
        # 0-300，走 setter 以拿到越界截断（直接赋 _volume 会绕过校验）
        self._volume = 100
        self.volume = volume
        self._device: Optional[DeviceInfo] = None

    # ------------------------------------------------------------------ #
    @property
    def volume(self) -> int:
        return self._volume

    @volume.setter
    def volume(self, v) -> None:
        try:
            self._volume = max(0, min(300, int(v)))      # 允许放大到 3 倍
        except Exception:
            self._volume = 100

    def _apply_volume(self, arr):
        """按当前音量缩放采样。

        100 = 原始输出；>100 是放大（用户反馈默认音量偏小）。
        放大后超过整数范围时用 **tanh 软限幅**，而不是硬切 ——
        硬切会把波形削平，听起来是刺啦的破音。

        注意 **不能按峰值归一化**：那样 `out/peak` 与增益无关，
        200% 和 300% 会输出得一模一样，放大就白调了。
        这里直接对归一化信号做 tanh：小信号几乎线性通过，
        大信号被压向 ±1，所以**增益越高听起来越响**（一直到饱和）。
        """
        import numpy as np

        v = self._volume
        if v == 100 or arr.size == 0:
            return arr
        if v <= 0:
            return np.zeros_like(arr)

        out = arr.astype(np.float32) * (v / 100.0)
        if not np.issubdtype(arr.dtype, np.integer):
            return out.astype(arr.dtype)

        info = np.iinfo(arr.dtype)
        peak = float(np.abs(out).max()) if out.size else 0.0
        if peak > info.max:
            x = out / info.max                 # 归一化（会超出 ±1）
            out = np.tanh(x) * info.max        # 软限幅
        out = np.clip(out, info.min, info.max)
        return out.astype(arr.dtype)

    # ------------------------------------------------------------------ #
    def resolve_device(self) -> Optional[DeviceInfo]:
        if self._device is None:
            if self.device_name:
                self._device = find_output_device(self.device_name)
                if self._device is None:
                    log.warning(
                        "找不到名字含 %r 的输出设备，回退到系统默认设备。可用设备：%s",
                        self.device_name,
                        [d.name for d in list_output_devices()],
                    )
        return self._device

    def _device_arg(self):
        d = self.resolve_device()
        return d.index if d else None

    # ------------------------------------------------------------------ #
    def play_wav(self, data: bytes) -> None:
        """播放一段完整 WAV（阻塞）。"""
        import io
        import wave

        import numpy as np
        import sounddevice as sd

        with wave.open(io.BytesIO(data), "rb") as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            sw = w.getsampwidth()
            nframes = w.getnframes()
            frames = w.readframes(nframes)

        # **空音频必须给出明确原因**。以前这里直接往下走，sd.play 抛一个
        # 空消息的异常，日志里只剩「播放失败：」什么都看不出来
        # （用户日志 02:36:52 就是这样，导致"没开始 TTS"的错觉）。
        if nframes <= 0 or not frames:
            raise RuntimeError("音频为空（合成结果没有内容）")

        dtype = {1: "int8", 2: "int16", 4: "int32"}.get(sw, "int16")
        arr = np.frombuffer(frames, dtype=dtype)
        if ch > 1:
            arr = arr.reshape(-1, ch)
        if arr.size == 0:
            raise RuntimeError("音频为空（解码后没有采样点）")
        arr = self._apply_volume(arr)          # 音量
        try:
            sd.play(arr, sr, device=self._device_arg(), blocking=True)
        except Exception:
            # 指定设备不可用时回退到系统默认设备，别让整句读不出来
            if self._device_arg() is not None:
                sd.play(arr, sr, device=None, blocking=True)
            else:
                raise

    def play_pcm_stream(
        self,
        chunks: Iterable[bytes],
        samplerate: int,
        channels: int = 1,
        sampwidth: int = 2,
        should_stop=None,
    ) -> int:
        """边收边播。返回写入的字节数。

        `should_stop` 是可选的零参回调，返回 True 时提前中止（用于打断上一句）。
        """
        import sounddevice as sd

        dtype = {1: "int8", 2: "int16", 4: "int32"}.get(sampwidth, "int16")
        written = 0
        stream = sd.RawOutputStream(
            samplerate=samplerate,
            channels=channels,
            dtype=dtype,
            device=self._device_arg(),
            latency=self.latency,
            blocksize=self.blocksize,
        )
        try:
            stream.start()
            for chunk in chunks:
                if should_stop is not None and should_stop():
                    break
                if not chunk:
                    continue
                stream.write(chunk)
                written += len(chunk)
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        return written

    def play_file(self, path: str) -> None:
        from pathlib import Path

        self.play_wav(Path(path).read_bytes())

    # ------------------------------------------------------------------ #
    def selfcheck(self, seconds: float = 0.4, freq: float = 440.0) -> bool:
        """放一小段提示音，用来确认输出设备真的出声了。"""
        import numpy as np
        import sounddevice as sd

        sr = 32000
        t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
        tone = (0.15 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
        try:
            sd.play(tone, sr, device=self._device_arg(), blocking=True)
            return True
        except Exception as e:
            log.error("播放自检失败：%s", e)
            return False
