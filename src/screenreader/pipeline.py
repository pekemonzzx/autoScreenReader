"""主编排：截屏 → OCR → 分句 → TTS → 播放。

流水线设计（方案 §3.5）：

    截屏 ─► OCR ─► 清洗/分句 ─► [合成队列] ─► [播放队列]
                      │             │              │
               按标点切短句    逐句 TTS      第 1 句好了就播
                              (可并发)       后续边合成边播

两条路径：
  * **流式**（默认）：合成工作线程直接把 PCM 块推进音频设备，第一块到达就出声。
    实测首包 667ms，比非流式整段 1786ms 快 2.7 倍。代价是同一时刻只处理一句。
  * **非流式 + 多线程**：整句合成完再入播放队列，可以并行合成多句、吞吐更高，
    但首句要等整句合成完。适合"不需要极低延迟、但要一次读很长"的场景。
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable, Dict, List, Optional

from .audio import Player
from .cache import AudioCache, make_key, wav_from_pcm
from .capture import ScreenCapturer, enable_dpi_awareness
from .config import AppConfig
from .ocr import OcrEngine, tune_intra_threads
from .segment import SegmentOptions, normalize_text, split_text
from .tts import TtsBackend, create_backend
from .watchdog import GpuWatchdog, Tier

log = logging.getLogger(__name__)

EventFn = Callable[[Dict], None]


@dataclass
class ReadResult:
    text: str
    lines: int
    line_texts: List[str] = field(default_factory=list)
    capture_ms: float = 0.0
    ocr_ms: float = 0.0
    skipped: bool = False
    reason: str = ""
    new_text: str = ""


def line_key(s: str) -> str:
    """行去重用的归一化键：去掉空白/标点、转小写。

    OCR 对同一行两次识别可能给出略有差异的字符串（实测出现过
    "deepseek" / "dleepseek" / "dleepseekHARNESS"），所以不能直接字符串相等。
    """
    return re.sub(r"[^\w\u4e00-\u9fff]", "", s).lower()


def is_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    ka, kb = line_key(a), line_key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    if abs(len(ka) - len(kb)) > max(4, int(max(len(ka), len(kb)) * 0.5)):
        return False
    return SequenceMatcher(None, ka, kb).ratio() >= threshold


class ScreenReaderPipeline:
    def __init__(
        self,
        config: AppConfig,
        backend: Optional[TtsBackend] = None,
        player: Optional[Player] = None,
        on_event: Optional[EventFn] = None,
        autotune_ocr: bool = True,
    ) -> None:
        self.cfg = config
        self.on_event = on_event or (lambda e: None)
        self._stop = threading.Event()
        self._interrupt = threading.Event()

        # --- 截屏 / OCR ---
        enable_dpi_awareness()
        self.capturer = ScreenCapturer(
            monitor_index=config.capture.monitor_index,
            backend=config.capture.backend,
            region=tuple(config.capture.region) if config.capture.region else None,
        )
        self.ocr = self._build_ocr(autotune=autotune_ocr)

        # --- 显存分档 ---
        self.tier: Tier = Tier("A", "未探测", True, 32, 1, 25, "")
        self.watchdog: Optional[GpuWatchdog] = None

        # --- TTS / 播放 / 缓存 ---
        self.backend = backend or create_backend(config.tts)
        self.player = player or Player(
            device=config.audio.device,
            latency=config.audio.latency,
            blocksize=config.audio.blocksize,
        )
        self.cache = AudioCache(
            directory=config.cache_dir(),
            max_entries=config.cache.max_entries,
            max_mb=config.cache.max_mb,
            enabled=config.cache.enabled,
        )

        # --- 队列与线程 ---
        self._synth_q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._play_q: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._threads: List[threading.Thread] = []
        self._speaking = threading.Event()
        self._last_text = ""
        self._last_line_keys: List[str] = []
        self._pending: List[str] = []
        self._stats = {"reads": 0, "spoken": 0, "cache_hits": 0, "errors": 0}

    # ------------------------------------------------------------------ #
    def _build_ocr(self, autotune: bool) -> OcrEngine:
        c = self.cfg.ocr
        intra = c.intra_threads or None
        if autotune and not intra:
            # 用当前截屏区域的第一帧做标定图，最贴近真实负载
            frame = self.capturer.grab()
            if frame is not None:
                intra = tune_intra_threads(frame, c.det_limit_side_len)
        return OcrEngine(
            det_limit_side_len=c.det_limit_side_len,
            use_cls=c.use_cls,
            intra_threads=intra,
        )

    def _emit(self, kind: str, **kw) -> None:
        try:
            self.on_event({"type": kind, "ts": time.time(), **kw})
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._threads:
            return
        self.tier = self._watchdog_start()

        for i in range(max(1, self.cfg.pipeline.synth_workers)):
            t = threading.Thread(target=self._synth_loop, name=f"synth-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        if not self._streaming():
            t = threading.Thread(target=self._play_loop, name="play", daemon=True)
            t.start()
            self._threads.append(t)
        log.info(
            "流水线已启动：TTS=%s 流式=%s 合成线程=%d 缓存=%s",
            self.backend.name,
            self._streaming(),
            max(1, self.cfg.pipeline.synth_workers),
            "开" if self.cfg.cache.enabled else "关",
        )

    def _watchdog_start(self) -> Tier:
        try:
            self.watchdog = GpuWatchdog(on_change=self._on_tier_change)
            return self.watchdog.start()
        except Exception as e:
            log.warning("显存看门狗启动失败（不影响运行）：%s", e)
            return self.tier

    def _on_tier_change(self, tier: Tier, gpu) -> None:
        self.tier = tier
        self._emit("tier", tier=tier.key, label=tier.label, gpu=str(gpu) if gpu else "")
        # 降档后立刻生效：清空积压，避免继续抢资源
        self._drain_queues()

    def _streaming(self) -> bool:
        return bool(self.cfg.tts.streaming) and max(1, self.cfg.pipeline.synth_workers) == 1

    def stop(self) -> None:
        self._stop.set()
        self._interrupt.set()
        self._drain_queues()
        for _ in self._threads:
            self._synth_q.put(None)
        self._play_q.put(None)
        for t in self._threads:
            t.join(timeout=3)
        self._threads.clear()
        if self.watchdog:
            self.watchdog.stop()
        try:
            self.capturer.close()
        except Exception:
            pass
        self.cache.flush()
        log.info("流水线已停止：%s", self._stats)

    def _drain_queues(self) -> None:
        while True:
            try:
                self._synth_q.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                self._play_q.get_nowait()
            except queue.Empty:
                break

    # ------------------------------------------------------------------ #
    def read_once(self) -> ReadResult:
        """截屏 + OCR 一次，返回要朗读的文本（已去重/限行）。"""
        t0 = time.perf_counter()
        frame = self.capturer.grab()
        t1 = time.perf_counter()
        if frame is None:
            return ReadResult("", 0, reason="抓帧失败")
        res = self.ocr.read(frame)
        t2 = time.perf_counter()
        self._stats["reads"] += 1

        max_lines = min(self.cfg.ocr.max_lines, self.tier.max_lines)
        lines = [ln for ln in res.lines if ln.score >= self.cfg.ocr.min_score]
        truncated = len(lines) > max_lines
        if truncated:
            lines = lines[:max_lines]

        text = normalize_text("\n".join(ln.text for ln in lines))
        rr = ReadResult(
            text=text,
            lines=len(lines),
            line_texts=[ln.text for ln in lines],
            capture_ms=(t1 - t0) * 1000,
            ocr_ms=(t2 - t1) * 1000,
        )
        if truncated:
            rr.reason = f"已截断到 {max_lines} 行（原 {len(res.lines)} 行）"

        # 只朗读**新增**的行，而不是每轮把整块重读一遍。
        # 读屏屏幕上通常只有一行在变（弹幕滚动 / 数值跳动），
        # 整块重读会把 90% 的时间浪费在念旧内容上（靠缓存虽然不慢，但听起来很怪）。
        if self.cfg.pipeline.dedupe_lines:
            new_lines = self._select_new_lines(rr.line_texts)
            rr.new_text = normalize_text("\n".join(new_lines))
            if not new_lines:
                rr.skipped = True
                rr.reason = rr.reason or "没有新增行"
                self._last_text = text
                return rr
            self._last_text = text
            self._emit(
                "ocr",
                lines=rr.lines,
                new_lines=len(new_lines),
                chars=len(text),
                capture_ms=round(rr.capture_ms, 1),
                ocr_ms=round(rr.ocr_ms, 1),
                truncated=truncated,
            )
            return rr

        if self.cfg.pipeline.skip_unchanged and text and text == self._last_text:
            rr.skipped = True
            rr.reason = "内容与上次相同"
            return rr
        self._last_text = text
        rr.new_text = text

        self._emit(
            "ocr",
            lines=rr.lines,
            new_lines=len(rr.line_texts),
            chars=len(text),
            capture_ms=round(rr.capture_ms, 1),
            ocr_ms=round(rr.ocr_ms, 1),
            truncated=truncated,
        )
        return rr

    def _select_new_lines(self, lines: List[str]) -> List[str]:
        """挑出相对上一轮新增/变化的行。

        变化行占比过高（>= refresh_ratio）就当作整屏刷新，全部朗读——
        否则切场景时会只读出零散几行，听感很碎。
        """
        prev = self._last_line_keys
        cur_keys = [line_key(x) for x in lines]
        new_lines: List[str] = []
        changed = 0
        for raw, key in zip(lines, cur_keys):
            if not key:
                continue
            if any(is_similar(raw, p, self.cfg.pipeline.similarity) for p in prev):
                continue
            changed += 1
            new_lines.append(raw)

        self._last_line_keys = [k for k in cur_keys if k]
        if prev and changed >= max(3, int(len(cur_keys) * self.cfg.pipeline.refresh_ratio)):
            return [x for x, k in zip(lines, cur_keys) if k]
        return new_lines

    # ------------------------------------------------------------------ #
    def speak(self, text: str) -> int:
        """把文本分句后投入合成队列，返回句子数。

        关于"要不要打断"（实测教训）：
        最初每次新读屏都打断上一轮，结果 44 次读屏只合成出 2 句——因为
        第一句还在播，下一轮就把它掐了，永远读不到第二句。
        读屏场景的正确行为是**排队**：让合成追上文字出现的速度即可；
        只有当积压超过 `max_backlog` 时才丢弃最旧的，避免越读越落后。
        """
        opts = SegmentOptions(
            split_chars=self.cfg.segment.split_chars,
            soft_split_chars=self.cfg.segment.soft_split_chars,
            min_chars=self.cfg.segment.min_chars,
            max_chars=self.cfg.segment.max_chars,
        )
        segs = split_text(text, opts)
        if not segs:
            return 0

        if self.cfg.pipeline.interrupt_on_new:
            self._interrupt_current()
        else:
            dropped = self._trim_backlog(segs)
            if dropped:
                self._emit("backlog_drop", dropped=dropped)

        for s in segs:
            self._synth_q.put(s)
        self._emit("segment", count=len(segs), segments=segs)
        return len(segs)

    def _trim_backlog(self, current: List[str]) -> int:
        """积压过多时丢弃。**优先丢已经滚出屏幕的**，返回丢弃条数。

        为什么不直接丢最旧的：被丢掉的那句会被标记成"见过"，
        后续轮次不再重排，于是**永久不会被朗读**。
        实测第一次跑就丢了 10 句（13 句只念了 3 句）。
        屏幕上的弹幕是会滚走的——已经不在当前画面里的内容丢掉不可惜，
        还在屏幕上的则应该留着念完。
        """
        limit = max(1, self.cfg.pipeline.max_backlog)
        pending: List[str] = []
        while True:
            try:
                pending.append(self._synth_q.get_nowait())
            except queue.Empty:
                break
        if len(pending) <= limit:
            for s in pending:
                self._synth_q.put(s)
            return 0

        keep: List[str] = []
        dropped = 0
        for s in pending:
            still_visible = any(is_similar(s, c, self.cfg.pipeline.similarity) for c in current)
            if still_visible or len(keep) < limit:
                keep.append(s)
            else:
                dropped += 1
        # 仍然超限就从最旧的开始丢
        while len(keep) > limit:
            keep.pop(0)
            dropped += 1
        for s in keep:
            self._synth_q.put(s)
        return dropped

    def _interrupt_current(self) -> None:
        """打断当前朗读（只在 interrupt_on_new=True 时用）。"""
        if self._speaking.is_set():
            self._interrupt.set()
            self._drain_queues()
            time.sleep(0.05)
            self._interrupt.clear()

    # ------------------------------------------------------------------ #
    def _cache_key(self, text: str) -> str:
        return make_key(
            text,
            ref_audio=self.cfg.tts.ref_audio,
            prompt_text=self.cfg.tts.prompt_text,
            speed_factor=self.cfg.tts.speed_factor,
            text_lang=self.cfg.tts.text_lang,
            extra=f"{self.backend.name}|steps={self.tier.sample_steps}|stream={self._streaming()}",
        )

    def _synth_loop(self) -> None:
        while not self._stop.is_set():
            item = self._synth_q.get()
            if item is None:
                break
            text = item.strip()
            if not text:
                continue
            try:
                self._speaking.set()
                if self._streaming():
                    self._synth_and_play_stream(text)
                else:
                    data = self._synth_full(text)
                    if data and not self._stop.is_set():
                        self._play_q.put(data)
            except Exception as e:
                self._stats["errors"] += 1
                log.warning("合成/播放失败（%s）：%s", text[:20], e)
                self._emit("error", where="synth", text=text[:40], error=str(e))
            finally:
                if self._synth_q.empty():
                    self._speaking.clear()

    def _synth_full(self, text: str) -> Optional[bytes]:
        key = self._cache_key(text)
        hit = self.cache.get(key)
        if hit is not None:
            self._stats["cache_hits"] += 1
            self._emit("cache_hit", text=text[:40], bytes=len(hit))
            return hit
        t0 = time.perf_counter()
        data = self.backend.synthesize(text, sample_steps=self.tier.sample_steps)
        dt = (time.perf_counter() - t0) * 1000
        if data:
            self.cache.put(key, data)
            self._stats["spoken"] += 1
            self._emit("tts", text=text[:40], ms=round(dt, 1), bytes=len(data))
        return data

    def _synth_and_play_stream(self, text: str) -> None:
        key = self._cache_key(text)
        hit = self.cache.get(key)
        if hit is not None:
            self._stats["cache_hits"] += 1
            self._emit("cache_hit", text=text[:40], bytes=len(hit))
            t0 = time.perf_counter()
            self.player.play_wav(hit)
            self._emit("play", text=text[:40], ms=round((time.perf_counter() - t0) * 1000, 1))
            return

        t0 = time.perf_counter()
        chunks = self.backend.synthesize_stream(text, sample_steps=self.tier.sample_steps)
        first = {"at": None}
        collected = bytearray()

        def spy():
            for c in chunks:
                if first["at"] is None:
                    first["at"] = (time.perf_counter() - t0) * 1000
                collected.extend(c)
                yield c

        self.player.play_pcm_stream(
            spy(),
            samplerate=(self.backend.stream_format.samplerate if self.backend.stream_format else 32000),
            channels=(self.backend.stream_format.channels if self.backend.stream_format else 1),
            sampwidth=(self.backend.stream_format.sampwidth if self.backend.stream_format else 2),
            should_stop=self._interrupt.is_set,
        )
        total = (time.perf_counter() - t0) * 1000
        if collected and not self._interrupt.is_set():
            fmt = self.backend.stream_format
            wav = wav_from_pcm(
                bytes(collected),
                samplerate=fmt.samplerate if fmt else 32000,
                channels=fmt.channels if fmt else 1,
                sampwidth=fmt.sampwidth if fmt else 2,
            )
            self.cache.put(key, wav)
            self._stats["spoken"] += 1
        self._emit(
            "tts",
            text=text[:40],
            first_packet_ms=round(first["at"], 1) if first["at"] else None,
            total_ms=round(total, 1),
            bytes=len(collected),
            streamed=True,
        )

    def _play_loop(self) -> None:
        while not self._stop.is_set():
            data = self._play_q.get()
            if data is None:
                break
            try:
                t0 = time.perf_counter()
                self.player.play_wav(data)
                self._emit("play", ms=round((time.perf_counter() - t0) * 1000, 1))
            except Exception as e:
                log.warning("播放失败：%s", e)
                self._emit("error", where="play", error=str(e))

    # ------------------------------------------------------------------ #
    def run_forever(self, interval_s: Optional[float] = None) -> None:
        """循环读屏。Ctrl+C 退出。"""
        interval = interval_s if interval_s is not None else self.cfg.pipeline.min_interval_s
        self.start()
        try:
            while not self._stop.is_set():
                rr = self.read_once()
                if rr.reason:
                    log.debug("读屏：%s", rr.reason)
                if rr.new_text and not rr.skipped:
                    self.speak(rr.new_text)
                time.sleep(max(0.05, interval))
        except KeyboardInterrupt:
            log.info("收到中断，正在退出…")
        finally:
            self.stop()

    def stats(self) -> Dict:
        return {
            **self._stats,
            "cache": self.cache.stats(),
            "tier": self.tier.key,
            "queue": {"synth": self._synth_q.qsize(), "play": self._play_q.qsize()},
        }
