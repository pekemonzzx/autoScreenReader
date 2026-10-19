"""TTS 结果缓存（磁盘持久化 + LRU 淘汰）。

为什么必须有（方案 §3.6）：
    读屏内容重复率极高——同一句弹幕、固定的 UI 文案、反复滚动出现的同一段字。
    实测 TTS 整段 1786ms / 流式首包 667ms，而缓存命中是 **0 延迟**。
    这是把感知延迟拉回 500ms 以内的主要手段。

实现：按 (文本, 音色参数) 算 sha1 当文件名，索引存 JSON，超限时按 LRU 淘汰。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

log = logging.getLogger(__name__)


def make_key(
    text: str,
    *,
    ref_audio: str = "",
    prompt_text: str = "",
    speed_factor: float = 1.0,
    text_lang: str = "zh",
    extra: str = "",
) -> str:
    """缓存键。任何会影响音频输出的参数都要进 key，否则会串音色。"""
    payload = "\x1f".join(
        [
            text.strip(),
            str(Path(ref_audio).resolve()) if ref_audio else "",
            prompt_text.strip(),
            f"{speed_factor:.3f}",
            text_lang,
            extra,
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def wav_from_pcm(pcm: bytes, samplerate: int, channels: int = 1, sampwidth: int = 2) -> bytes:
    """把裸 PCM 包成合法 WAV。

    GPT-SoVITS 流式返回的 RIFF 头里 size 字段是 0（流式标准做法），
    直接落盘后 `wave.open` 会认为时长 0，所以这里自己重建头。
    """
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(samplerate)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_info(data: bytes) -> Tuple[int, int, float]:
    """返回 (采样率, 声道数, 时长秒)。失败返回 (0, 0, 0.0)。"""
    import io

    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            sr = w.getframerate()
            n = w.getnframes()
            return sr, w.getnchannels(), n / float(sr) if sr else 0.0
    except Exception:
        return 0, 0, 0.0


@dataclass
class CacheEntry:
    file: str
    size: int
    hits: int = 0
    created: float = 0.0
    used: float = 0.0


class AudioCache:
    """线程安全的音频 LRU 缓存。"""

    def __init__(
        self,
        directory: Path,
        max_entries: int = 500,
        max_mb: int = 512,
        enabled: bool = True,
        ttl_s: float = 0.0,
    ) -> None:
        self.enabled = enabled
        self.dir = Path(directory)
        self.max_entries = max_entries
        self.max_bytes = max_mb * 1024 * 1024
        # **条目存活时间（秒）。0 = 不自动过期。**
        #
        # 用户要求"语音生成 30 秒后自动清理"：读屏生成的音频绝大多数只播一次，
        # 留着只占磁盘。注意清的是**磁盘缓存**，卡片自己的 `c.audio` 还在内存里，
        # 所以播放在清理后依然正常；只有"过一会儿再点同一张卡"才会重新合成。
        self.ttl_s = float(ttl_s or 0.0)
        self._lock = threading.RLock()
        self._index: Dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0

        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._load_index()

    # ------------------------------------------------------------------ #
    @property
    def index_path(self) -> Path:
        return self.dir / "index.json"

    def _load_index(self) -> None:
        p = self.index_path
        if not p.exists():
            return
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            for k, v in raw.get("entries", {}).items():
                self._index[k] = CacheEntry(**v)
            self._hits = raw.get("hits", 0)
            self._misses = raw.get("misses", 0)
            # 清掉索引里有、磁盘上没有的条目
            for k in [k for k, e in self._index.items() if not (self.dir / e.file).exists()]:
                self._index.pop(k, None)
            log.info("音频缓存：%d 条，命中 %d / 未命中 %d", len(self._index), self._hits, self._misses)
        except Exception as e:
            log.warning("缓存索引读取失败，重建：%s", e)
            self._index = {}

    def _save_index(self) -> None:
        if not self.enabled:
            return
        try:
            payload = {
                "entries": {k: vars(v) for k, v in self._index.items()},
                "hits": self._hits,
                "misses": self._misses,
                "saved": time.time(),
            }
            tmp = self.index_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.index_path)
        except Exception as e:
            log.warning("缓存索引写入失败：%s", e)

    # ------------------------------------------------------------------ #
    def get(self, key: str) -> Optional[bytes]:
        if not self.enabled:
            return None
        with self._lock:
            e = self._index.get(key)
            if e is None:
                self._misses += 1
                return None
            p = self.dir / e.file
            try:
                data = p.read_bytes()
            except Exception:
                self._index.pop(key, None)
                self._misses += 1
                return None
            e.hits += 1
            e.used = time.time()
            self._hits += 1
            return data

    def put(self, key: str, data: bytes) -> None:
        if not self.enabled or not data:
            return
        with self._lock:
            name = f"{key}.wav"
            p = self.dir / name
            try:
                p.write_bytes(data)
            except Exception as e:
                log.warning("缓存写入失败：%s", e)
                return
            now = time.time()
            self._index[key] = CacheEntry(
                file=name, size=len(data), hits=0, created=now, used=now
            )
            self._evict()
            self._save_index()

    def _evict(self) -> None:
        total = sum(e.size for e in self._index.values())
        while self._index and (len(self._index) > self.max_entries or total > self.max_bytes):
            # 先淘汰最久未使用的
            victim = min(self._index.items(), key=lambda kv: kv[1].used)
            key, e = victim
            try:
                (self.dir / e.file).unlink(missing_ok=True)
            except Exception:
                pass
            total -= e.size
            self._index.pop(key, None)
            log.debug("缓存淘汰 %s", key[:8])

    def sweep(self, now: Optional[float] = None) -> int:
        """删掉超过 `ttl_s` 没被用过的条目，返回删了几个。

        用户要求"语音生成后 30 秒自动清理"。这里的判据用 `used`
        （最后一次被读取的时间）：音频刚生成时等于 `created`，
        如果这 30 秒里没人再用它，就删掉。

        **清的是磁盘缓存，不影响正在播的音频** —— 播放读的是卡片内存里的
        `c.audio` 字节。所以这里可以放心删。
        """
        if not self.enabled or self.ttl_s <= 0:
            return 0
        now = now if now is not None else time.time()
        with self._lock:
            dead = [k for k, e in self._index.items()
                    if now - (e.used or e.created) >= self.ttl_s]
            for key in dead:
                e = self._index.pop(key, None)
                if e is None:
                    continue
                try:
                    (self.dir / e.file).unlink(missing_ok=True)
                except Exception:
                    pass
            if dead:
                self._save_index()
                log.debug("缓存过期清理 %d 条（ttl=%.0fs）", len(dead), self.ttl_s)
            return len(dead)

    def clear(self) -> int:
        with self._lock:
            n = len(self._index)
            for e in self._index.values():
                try:
                    (self.dir / e.file).unlink(missing_ok=True)
                except Exception:
                    pass
            self._index.clear()
            self._hits = self._misses = 0
            self._save_index()
            return n

    def stats(self) -> Dict[str, object]:
        with self._lock:
            total = sum(e.size for e in self._index.values())
            calls = self._hits + self._misses
            return {
                "enabled": self.enabled,
                "entries": len(self._index),
                "total_mb": round(total / 1024 / 1024, 1),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / calls, 3) if calls else 0.0,
            }

    def flush(self) -> None:
        with self._lock:
            self._save_index()
