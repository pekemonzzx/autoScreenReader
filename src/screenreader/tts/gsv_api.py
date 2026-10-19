"""GPT-SoVITS v2 推理 API 后端（HTTP）。

对应整合包的 `api_v2.py`：

    runtime\\python.exe -s api_v2.py -a 127.0.0.1 -p 9880 -c <tts_infer.yaml>

实测（4080S，v2ProPlus 微调模型，25 字文本）：
    非流式整段 p50 1786ms
    流式首包   p50  667ms   <- 默认走这条

流式返回的坑：响应仍是 RIFF/WAV 头，但 **size 字段是 0**（流式标准做法），
直接存盘后 `wave.open` 会认为时长 0。所以这里自己解析头、剥掉它，
只把后面的 PCM 块交给播放器——这也正是流式低延迟的意义（边收边播）。
"""

from __future__ import annotations

import logging
import struct
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

import requests

from .base import StreamFormat, TtsBackend

log = logging.getLogger(__name__)


def parse_wav_header(buf: bytes) -> Optional[tuple]:
    """从缓冲区里找 WAV 头，返回 (fmt信息, data开始偏移)。

    容忍 size 字段为 0 的流式 WAV。
    """
    if len(buf) < 12 or buf[:4] != b"RIFF" or buf[8:12] != b"WAVE":
        return None
    pos = 12
    fmt = None
    while pos + 8 <= len(buf):
        cid = buf[pos : pos + 4]
        (csize,) = struct.unpack_from("<I", buf, pos + 4)
        body = pos + 8
        if cid == b"fmt ":
            if body + 16 > len(buf):
                return None
            audio_fmt, channels, samplerate = struct.unpack_from("<HHI", buf, body)
            bits = struct.unpack_from("<H", buf, body + 14)[0]
            fmt = (samplerate, channels, max(1, bits // 8))
            pos = body + csize
            continue
        if cid == b"data":
            if fmt is None:
                return None
            return fmt, body
        # 其它块按声明长度跳过；长度为 0 时无法继续
        if csize == 0:
            return None
        pos = body + csize + (csize & 1)
    return None


class GsvApiBackend(TtsBackend):
    name = "gsv_api"

    def __init__(
        self,
        api_url: str = "http://127.0.0.1:9880",
        ref_audio: str = "",
        prompt_text: str = "",
        prompt_lang: str = "zh",
        text_lang: str = "zh",
        speed_factor: float = 1.0,
        split_method: str = "cut5",
        sample_steps: int = 32,
        timeout_s: float = 60.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.ref_audio = ref_audio
        self.prompt_text = prompt_text
        self.prompt_lang = prompt_lang
        self.text_lang = text_lang
        self.speed_factor = speed_factor
        self.split_method = split_method
        self.sample_steps = sample_steps
        self.timeout_s = timeout_s
        self._session = session or requests.Session()
        self._fmt: Optional[StreamFormat] = None
        self._health_ok = False
        self._health_at = 0.0

    # ------------------------------------------------------------------ #
    def _payload(self, text: str, streaming: bool) -> Dict[str, Any]:
        return {
            "text": text,
            "text_lang": self.text_lang,
            "ref_audio_path": self.ref_audio,
            "prompt_text": self.prompt_text,
            "prompt_lang": self.prompt_lang,
            "text_split_method": self.split_method,
            "batch_size": 1,
            "media_type": "wav",
            "speed_factor": self.speed_factor,
            "sample_steps": self.sample_steps,
            "streaming_mode": streaming,
        }

    def health(self, ttl_s: float = 30.0, timeout_s: float = 20.0) -> bool:
        """探活。带 TTL 缓存——每次真发一次合成请求要 3 秒左右，不该反复调。"""
        now = time.time()
        if self._health_at and (now - self._health_at) < ttl_s:
            return self._health_ok
        ok = False
        try:
            r = self._session.post(
                f"{self.api_url}/tts",
                json=self._payload("测", False),
                timeout=timeout_s,
            )
            ok = r.status_code == 200
            if not ok:
                log.warning("TTS 探活返回 HTTP %s", r.status_code)
        except Exception as e:
            log.warning("TTS 探活失败：%s", e)
        self._health_ok, self._health_at = ok, now
        return ok

    # ------------------------------------------------------------------ #
    def synthesize(self, text: str, **kw) -> bytes:
        payload = self._payload(text, streaming=False)
        payload.update({k: v for k, v in kw.items() if k in payload})
        if not self.ref_audio:
            raise RuntimeError(
                "没有配置参考音频（tts.ref_audio）。\n"
                "最常见的原因：**没有选中声库**。\n"
                "  · 检查程序目录下的 voices\\ 里有没有声库文件夹\n"
                "    （把 xxx.voicepack.zip 解压进去即可）\n"
                "  · 然后在工具栏右侧的「声库」下拉框里选一个\n"
                "  · 如果下拉框是空的，说明一个声库都没装"
            )
        r = self._session.post(f"{self.api_url}/tts", json=payload, timeout=self.timeout_s)
        if r.status_code != 200:
            # 把服务端返回的原因带出来，否则只有一句 "400 Bad Request" 完全没法排查
            detail = ""
            try:
                detail = r.text[:400]
            except Exception:
                pass
            raise RuntimeError(
                f"TTS 服务返回 HTTP {r.status_code}：{detail}\n"
                f"参考音频={(self.ref_audio or '（空）')}  参考文本={(self.prompt_text or '（空）')[:30]}"
            )
        return r.content

    # ------------------------------------------------------------------ #
    @property
    def stream_format(self) -> Optional[StreamFormat]:
        return self._fmt

    def synthesize_stream(self, text: str, **kw) -> Iterator[bytes]:
        """产出裸 PCM 块。第一块到达时已知采样率（见 stream_format）。"""
        payload = self._payload(text, streaming=True)
        payload.update({k: v for k, v in kw.items() if k in payload})

        buf = bytearray()
        header_done = False
        with self._session.post(
            f"{self.api_url}/tts", json=payload, timeout=self.timeout_s, stream=True
        ) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                if header_done:
                    yield chunk
                    continue

                buf.extend(chunk)
                parsed = parse_wav_header(buf)
                if parsed is None:
                    # 头还没收全；注意别无限缓存（万一对方不是 WAV）
                    if len(buf) > 1 << 20:
                        raise RuntimeError("响应不是可识别的 WAV，前 1MB 未能解析出头")
                    continue
                fmt, data_off = parsed
                samplerate, channels, sampwidth = fmt
                self._fmt = StreamFormat(samplerate, channels, sampwidth)
                header_done = True
                rest = bytes(buf[data_off:])
                buf.clear()
                if rest:
                    yield rest

        if not header_done:
            raise RuntimeError("流式响应结束但没收到完整的 WAV 头")

    # ------------------------------------------------------------------ #
    def set_refer_audio(self, path: str) -> None:
        """固定参考音频（服务端会缓存其特征，避免每次重算 HuBERT）。"""
        self.ref_audio = path

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass
