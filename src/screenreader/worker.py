"""读屏台的引擎层：识别轮询 + 后台预合成 + 单实例播放。

**不依赖 Qt、不依赖任何界面**，只用 threading + 回调。
这样窗口模式和悬浮字幕条模式都能挂同一套引擎，也方便单测。

数据流：

    DetectorWorker ──observe──► CardStore ──new/rearmed──► SynthPool
         ▲                          │                        │
    定时轮询各区域              （生灭规则）              后台预合成
         │                          │                        ▼
    capture + OCR             died -> 销毁卡片          card.audio = wav
                                                          │
                                            PlaybackService ◄── 用户点播放
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .audio import Player
from .board import BoardConfig, Card, CardStore, ObserveResult, Region
from .cache import AudioCache, make_key
from .capture import ScreenCapturer
from .config import AppConfig
from .ocr import OcrEngine, tune_intra_threads
from .segment import (
    SegmentOptions,
    clean_for_tts,
    has_speakable,
    merge_lines_to_sentences,
    paragraph_key,
    split_text,
)
from .tts import TtsBackend, create_backend
from .watchdog import GpuWatchdog, Tier

log = logging.getLogger(__name__)

EventFn = Callable[[Dict], None]


@dataclass
class EngineConfig:
    poll_interval_s: float = 0.4
    synth_workers: int = 1
    # **自动朗读**：勾选后每张卡片生成就自动排进播放队列，顺序读一遍。
    #
    # 效果等价于"自动化的「全部播放」"：不用手点，新出现的句子自动播。
    # 每张卡只播一次（靠 Card.auto_done 保证，不重复）。
    auto_read_all: bool = False
    # **画面没变就跳过 OCR**（CPU 优化里收益最大的一条）。
    #
    # OCR 是唯一的大头：实测 2560x1440 一帧 825ms，截屏只要 5.9ms。
    # 而读屏时屏幕绝大部分时间是静止的 —— 对同一幅画面反复跑 OCR 纯属浪费。
    # 打开后静态画面下的 CPU 占用接近零；画面一变立刻恢复识别。
    skip_unchanged_frame: bool = True
    # 识别出的行是否再按标点切句（granularity="sentence" 时用）
    segment_enabled: bool = False
    max_lines_per_region: int = 30
    min_score: float = 0.5
    # 送进 TTS 之前清洗文字（去行首编号/排版装饰等）。
    # 实测：不清洗时 "1语音输出前…" 会被念出一个孤立的"一"，听起来像多了个怪音。
    clean_text: bool = True
    # 丢掉不值得念的行（纯符号、纯数字、太短）
    drop_unspeakable: bool = True
    # 整段模式：内容要**稳定这么久**才建卡去合成。
    #
    # 为什么需要：滚动/翻页时区域内容每 0.4 秒就变一次，如果每次都建卡合成，
    # 就会不停播报"刚出现的前几个字"，等页面停下来想完整读的时候，
    # 卡片已经被这些碎片搅乱，没法顺序朗读了。
    paragraph_settle_s: float = 0.75


class BoardEngine:
    """识别 + 合成 + 播放的完整引擎（无界面）。"""

    def __init__(
        self,
        app_cfg: AppConfig,
        engine_cfg: Optional[EngineConfig] = None,
        backend: Optional[TtsBackend] = None,
        player: Optional[Player] = None,
        on_event: Optional[EventFn] = None,
        board_cfg: Optional[BoardConfig] = None,
    ) -> None:
        self.cfg = app_cfg
        self.ecfg = engine_cfg or EngineConfig()
        self.on_event = on_event or (lambda e: None)
        self.store = CardStore(board_cfg)
        self.backend = backend or create_backend(app_cfg.tts)
        self.player = player or Player(
            device=app_cfg.audio.device,
            latency=app_cfg.audio.latency,
            blocksize=app_cfg.audio.blocksize,
            volume=getattr(app_cfg.audio, "volume", 100),
        )
        self.cache = AudioCache(
            directory=app_cfg.cache_dir(),
            max_entries=app_cfg.cache.max_entries,
            max_mb=app_cfg.cache.max_mb,
            enabled=app_cfg.cache.enabled,
            ttl_s=getattr(app_cfg.cache, "ttl_s", 0.0),
        )

        self._stop = threading.Event()
        self._synth_q: "queue.Queue[str]" = queue.Queue()
        self._claimed: set = set()
        # 卡片排队等合成的时刻：{card_id: ts}。用来发现"卡在合成中"的卡片。
        self._queued_at: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._threads: List[threading.Thread] = []
        self._capturers: Dict[str, ScreenCapturer] = {}
        self._ocr: Optional[OcrEngine] = None
        self._stop_play = threading.Event()
        self._play_thread: Optional[threading.Thread] = None
        self._running = False
        self.stats = {"polls": 0, "ocr_ms": 0.0, "synth_ok": 0, "synth_fail": 0, "cache_hits": 0, "skipped_dead": 0}
        self.watchdog: Optional[GpuWatchdog] = None
        self.tier: Optional[Tier] = None
        # 整段模式：暂停自动朗读（用户手动点播时临时让路）
        self._play_paused = False
        # 整段模式每区域的内容稳定性状态：{rid: {key, since, committed}}
        self._para_state: Dict[str, dict] = {}
        # 正在"整段朗读中"的区域 —— 新生成的句子会自动接进播放队列
        self._playing_regions: set = set()
        # 每个区域的连续失败次数（用于日志和界面提示，不参与逻辑）
        self._region_errors: Dict[str, int] = {}
        # 每个区域上一帧的廉价签名，用来判断"画面有没有变"
        self._frame_sig: Dict[str, object] = {}
        # 因为"画面没变"而跳过 OCR 的次数（性能统计）
        self.stats_skipped_frames = 0
        # 要求**绕过缓存**重新合成的卡片（"重新生成"按钮用）
        self._force_synth: set = set()
        # **OCR 互斥锁**：检测线程和「截图朗读」都会调 `ocr.read()`，
        # 同一个 ONNX 会话不能并发跑，否则会崩或给出错乱结果。
        self._ocr_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def _emit(self, kind: str, **kw) -> None:
        try:
            self.on_event({"type": kind, "ts": time.time(), **kw})
        except Exception:
            pass

    # -------------------------------------------------------- 区域管理 --
    def add_region(self, name: str, rect, monitor_index: int = 0,
                   mode: str = "line", temp: bool = False) -> Region:
        r = self.store.add_region(name, rect, monitor_index)
        if mode and mode != r.mode:
            self.store.update_region(r.id, mode=mode)
        if temp:
            # **临时区域（「截图朗读」）**：读完就删，且不写进配置文件。
            r.temp = True
        self._drop_capturer(r.id)  # 下次轮询按新区域重建
        self._emit("region_added", region=r)
        return r

    def region_enabled_map(self) -> Dict[str, bool]:
        """记下每个**常驻**区域当前的开启/停用状态（截图朗读前存快照用）。"""
        return {r.id: r.enabled for r in self.store.regions if not getattr(r, "temp", False)}

    def apply_region_enabled_map(self, m: Dict[str, bool]) -> None:
        """把快照里的开启/停用状态恢复回去（截图朗读结束后用）。"""
        for rid, en in (m or {}).items():
            if self.store.region(rid) is not None:
                self.store.update_region(rid, enabled=bool(en))

    def region_playback_done(self, rid: str) -> bool:
        """这个区域的句子是不是**全部读完了**。

        「截图朗读」读完要自动清场，所以必须能准确判断"读完了"。

        **关键：要求 `played_once`，只看 `auto_done` 会误判。**
        `_autoplay_loop` 是先把卡片出队（`queued=False; auto_done=True`），
        再调 `play_card()`（里面才 `mark_playing`）。中间那一小段时间里
        "没排队 + auto_done + 没有在播"三条全成立 —— 以前就会在这里误判成
        "读完了"，把区域删掉、卡片销毁，紧接着的 `play_card` 拿不到卡片，
        于是**完全没声音**。
        （用户实测：OCR 有结果、TTS 返回 200、就是不读。）
        """
        cards = self.store.cards_of(rid)
        if not cards:
            return False
        ids = {c.id for c in cards}
        if self.store.playing_id in ids:
            return False
        # autoplay 队列里还有该区域的卡在排队 -> 没读完
        try:
            nxt = self.store.next_auto_card()
        except Exception:
            nxt = None
        if nxt is not None and nxt.region_id == rid:
            return False
        for c in cards:
            if c.queued and not c.auto_done:
                return False
            if c.audio_state == "failed":
                continue                    # 失败的不阻塞清场
            if not c.auto_done:
                return False
            if not c.played_once:
                return False                # **必须真的播过**（见上面说明）
        return True

    def observe_text(self, rid: str, sentences) -> int:
        """把一段**已经识别好的文字**变成卡片，并**排进合成队列**。

        「截图朗读」专用（文字来自对冻结图的一次 OCR，不走轮询）。

        **为什么不能用 `store.observe()` 直接建卡**：
        平时卡片是 `_poll_region()` 建出来的，建完会走 `_after_observe()`，
        而**只有 `_after_observe()` 会调 `_enqueue()` 把卡片排进合成队列**。
        截图朗读如果绕过它直接 `store.observe()`，卡片会出现在界面上、
        状态是"合成中"，但**永远不会有任何句子去合成** ——
        autoplay 也会一直等 `c.ready`，于是"不读也不清理"。
        （用户实测：OCR 很快出结果，然后就不动了。）

        返回建了几张卡。
        """
        out = self.store.observe(rid, sentences)
        for c in out.added:
            self._enqueue(c)                 # ← 关键：排进合成队列
            self._emit("card_added", card_id=c.id, region_id=rid, text=c.text[:60])
        for c in out.rearmed:
            self._enqueue(c)
            self._emit("card_rearmed", card_id=c.id, region_id=rid, text=c.text[:60])
        for c in out.died:
            self._emit("card_died", card_id=c.id, region_id=rid, text=c.text[:40])
        return len(out.added)

    def retry_synth(self, cid: str) -> bool:
        """**重新生成**某张卡片的音频（用户要求）。

        动机：用户反馈"个别语音会少半句话"，怀疑是合成偶发截断。
        重试会**绕过缓存**：坏音频如果已经缓存，命中缓存只会把同一份坏音频
        拿回来；GPT-SoVITS 是采样生成的，重新合成有机会得到完整的一份。

        重试成功也**不写回缓存**（见 `_synth_one`），避免把坏音频固化。
        """
        c = self.store.card(cid)
        if c is None:
            return False
        c.audio_state = "pending"
        c.audio = None
        c.audio_error = ""
        c.from_cache = False
        c.auto_done = False          # 允许自动朗读再播一遍这一句
        with self._lock:
            self._claimed.discard(c.id)
            self._force_synth.add(c.id)
        self._emit("card_retry", card_id=c.id, region_id=c.region_id,
                   text=c.text[:40])
        self._enqueue(c)
        return True

    def set_speed(self, factor: float) -> float:
        """设置合成语速（0.5~2.0）。

        走的是 GPT-SoVITS 的 `speed_factor`（**合成侧变速**，音色和音高不变）。
        不用播放时重采样，因为那会把声音变调（快放=尖、慢放=闷）。

        注意：缓存键里包含 speed_factor，所以改了速率之后**不会命中旧音频**，
        新合成的才按新速率。已经生成的卡片要听新速率得点"重新生成"。
        """
        try:
            f = float(factor)
        except Exception:
            return float(getattr(self.cfg.tts, "speed_factor", 1.0))
        f = max(0.5, min(2.0, f))
        self.cfg.tts.speed_factor = f
        try:
            self.backend.speed_factor = f
        except Exception:
            pass
        return f

    def clear_region_cards(self, rid: str) -> int:
        """清掉某个区域的卡片（换识别模式时用）。

        单行模式建的是碎句、整段模式建的是合并后的句子，两者语义不同，
        换模式后必须重来，否则新旧混在一起读起来是乱的。

        **只停这个区域相关的播放**，不要 `stop_playback()` 一刀切 ——
        以前那样做，你在单行区域正播着的时候去切整段区域的模式，
        单行那边会被无声掐断，而且播放状态没同步，看起来就像"卡住了"。
        """
        # 这个区域如果在整段朗读中，一并退出朗读态
        self._playing_regions.discard(rid)
        # 正在播的这张卡属于本区域才真的停
        cur = self.store.playing_id
        if cur:
            c = self.store.card(cur)
            if c is not None and c.region_id == rid:
                self.stop_playback()
        self._para_state.pop(rid, None)   # 稳定性状态也清掉，换模式后重新判定
        n = self.store.clear_region(rid)
        self._emit("region_cards_cleared", region_id=rid, cards=n)
        return n

    def remove_region(self, rid: str) -> int:
        n = self.store.remove_region(rid)
        self._drop_capturer(rid)
        self._emit("region_removed", region_id=rid, cards=n)
        return n

    def update_region(self, rid: str, **kw) -> Optional[Region]:
        r = self.store.update_region(rid, **kw)
        if r and any(k in kw for k in ("rect", "monitor_index", "enabled")):
            self._drop_capturer(rid)
        self._emit("region_updated", region=r)
        return r

    def _drop_capturer(self, rid: str) -> None:
        cap = self._capturers.pop(rid, None)
        if cap:
            try:
                cap.close()
            except Exception:
                pass

    # -------------------------------------------------------- 生命周期 --
    def start(self) -> None:
        if self._running:
            return
        # 这类配置缺失会让每个请求都失败，不如启动时就喊出来
        if not self.cfg.tts.ref_audio:
            msg = "tts.ref_audio 为空 —— 所有合成都将失败。请在 data/config.json 里填参考音频。"
            log.error(msg)
            self._emit("error", where="config", error=msg)
        if not self.cfg.tts.prompt_text:
            log.warning("tts.prompt_text 为空，克隆效果会明显变差")

        self._running = True
        self._stop.clear()
        # 注意：**不要在这里初始化 OCR**。
        # _ensure_ocr() 会跑 OCR 线程数标定（实测约 10 秒），在界面线程上调用
        # 会让窗口"未响应"。检测线程的 _poll_region() 本来就会懒加载它，
        # 所以这里交给后台线程做。
        # 用户实测反馈过："点击启动时有明显的未响应等待"。

        # 显存看门狗：档位决定采样步数（GPU 紧张时降步数换速度），只降不升
        try:
            self.watchdog = GpuWatchdog(on_change=self._on_tier_change)
            self.tier = self.watchdog.start()
            self._emit("tier", tier=self.tier.key, label=self.tier.label)
        except Exception as e:
            log.warning("显存看门狗启动失败（不影响运行）：%s", e)

        for i in range(max(1, self.ecfg.synth_workers)):
            t = threading.Thread(target=self._synth_loop, name=f"synth-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        t = threading.Thread(target=self._detect_loop, name="detect", daemon=True)
        t.start()
        self._threads.append(t)
        # 整段模式的自动连续朗读
        t = threading.Thread(target=self._autoplay_loop, name="autoplay", daemon=True)
        t.start()
        self._threads.append(t)

        log.info(
            "读屏台引擎已启动：区域 %d 个，轮询 %.2fs，合成线程 %d",
            len(self.store.regions), self.ecfg.poll_interval_s, self.ecfg.synth_workers,
        )
        self._emit("started")

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._stop.set()
        self.stop_playback()
        for _ in self._threads:
            self._synth_q.put("")
        for t in self._threads:
            t.join(timeout=3)
        self._threads.clear()
        if self.watchdog:
            self.watchdog.stop()
            self.watchdog = None
        for rid in list(self._capturers):
            self._drop_capturer(rid)
        self.cache.flush()
        log.info("读屏台引擎已停止：%s", self.stats)
        self._emit("stopped")

    def ensure_threads(self) -> bool:
        """确保 detect / autoplay / **synth** 线程都还活着；死了就补一个。

        为什么需要：线程是 daemon，一旦因为意外退出，`running` 仍然是 True、
        界面还显示"识别中"，但**再也不会有新卡片**。用户实测到的现象是
        "识别一直卡住，必须手动停止再开始"。这里让引擎自己能救回来。

        **synth 必须一起管**：以前只补 detect/autoplay，合成线程一旦不在，
        卡片就永远停在"合成中"，而且没有任何东西会去修它
        （用户实测：截图朗读之后卡片显示"合成中"再也不动、TTS 根本不开始）。
        返回是否做了修复。
        """
        if not self._running:
            return False
        alive = {t.name for t in self._threads if t.is_alive()}
        fixed = False
        for name, fn in (("detect", self._detect_loop),
                         ("autoplay", self._autoplay_loop),
                         ("synth", self._synth_loop)):
            if name in alive:
                continue
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)
            log.warning("%s 线程已不在，自动重启（自愈）", name)
            fixed = True
        return fixed

    def recover_stuck_synth(self, max_s: float = 45.0) -> int:
        """把长时间卡在"合成中"的卡片救回来。

        卡片状态是 `pending` 就表示"排队等合成"。如果合成线程不在、或者
        卡在某个请求上，这些卡片会**永远**显示"合成中" —— 用户看到的就是
        "没开始 TTS、一直转"。这里给一个上界：

          * 第一次超时 -> 重新排队（可能是队列里丢了）
          * 第二次还超时 -> 判失败并写明原因（TTS 服务没起来 / 卡住）

        返回处理了几张。
        """
        now = time.time()
        n = 0
        for c in list(self.store.cards):
            if c.audio_state != "pending":
                continue
            t0 = self._queued_at.get(c.id)
            if t0 is None or now - t0 < max_s:
                continue
            n += 1
            if c.synth_retries <= 0:
                c.synth_retries = 1
                log.warning("「%s」合成卡住 %.0fs，重新排队", c.text[:20], now - t0)
                with self._lock:
                    self._claimed.discard(c.id)
                self._enqueue(c)
            else:
                c.audio_state = "failed"
                c.audio_error = "合成超时（TTS 服务可能没在运行）"
                self.stats["synth_fail"] += 1
                log.warning("「%s」合成两次超时，判失败", c.text[:20])
                self._release(c.id)
                self._emit("card_failed", card_id=c.id, region_id=c.region_id,
                           error=c.audio_error)
        return n

    @property
    def running(self) -> bool:
        return self._running

    # ---------------------------------------------------------- OCR 准备 --
    def _ensure_ocr(self) -> Optional[OcrEngine]:
        if self._ocr is not None:
            return self._ocr
        c = self.cfg.ocr
        intra = c.intra_threads or None
        if not intra and c.autotune and self.store.regions:
            frame = self._grab(self.store.regions[0])
            if frame is not None:
                intra = tune_intra_threads(frame, c.det_limit_side_len)
        try:
            self._ocr = OcrEngine(
                det_limit_side_len=c.det_limit_side_len, use_cls=c.use_cls, intra_threads=intra
            )
        except Exception as e:
            log.error("OCR 初始化失败：%s", e)
            self._emit("error", where="ocr_init", error=str(e))
            self._ocr = None
        return self._ocr

    def _capturer(self, r: Region) -> Optional[ScreenCapturer]:
        cap = self._capturers.get(r.id)
        if cap is None:
            try:
                cap = ScreenCapturer(
                    monitor_index=r.monitor_index, backend="auto", region=tuple(r.rect)
                )
                self._capturers[r.id] = cap
            except Exception as e:
                log.warning("区域 %s 截屏器创建失败：%s", r.name, e)
                self._emit("error", where="capture_init", region=r.id, error=str(e))
                return None
        return cap

    def _grab(self, r: Region):
        cap = self._capturer(r)
        return cap.grab() if cap else None

    # ---------------------------------------------------------- 轮询线程 --
    def _detect_loop(self) -> None:
        """识别轮询线程。

        **这里必须层层兜住异常。**
        以前整个循环没有任何 try/except —— 只要某一个区域的一次轮询抛异常
        （两个区域不同模式时更容易触发），这个线程就**静默死亡**，
        界面上表现成"识别卡住、卡片不再更新"，必须清空区域重开才恢复。
        现在：单个区域出错只跳过那一轮并报错，线程本身永不退出。
        """
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                for r in self.store.regions:
                    if self._stop.is_set():
                        break
                    if not r.enabled:
                        continue
                    # **临时区域（截图朗读）永远不参与轮询**。
                    # 它的文字是对冻结图做一次 OCR 得来的，如果又被轮询，
                    # 就会走进"整段稳定性判定"，和 play_paragraph 抢状态
                    # （用户日志里的 KeyError: 'key' 就是这么触发的）。
                    if getattr(r, "temp", False):
                        continue
                    try:
                        self._poll_region(r)
                    except Exception as e:
                        # 单个区域失败不影响其它区域
                        self._region_errors[r.id] = self._region_errors.get(r.id, 0) + 1
                        log.exception("区域 %s 轮询失败（第 %d 次）：%s",
                                      r.name, self._region_errors[r.id], e)
                        self._emit("error", where="poll", region=r.id,
                                   error=f"{type(e).__name__}: {e}"[:200])
            except Exception as e:
                # 连遍历都出事（比如 store 被并发改坏）也不能让线程死
                log.exception("识别轮询异常：%s", e)
                self._emit("error", where="detect_loop", error=str(e)[:200])
            self.stats["polls"] += 1
            self.stats["ocr_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            self._stop.wait(self.ecfg.poll_interval_s)

    @staticmethod
    def _frame_signature(frame):
        """把一帧压成 48x48 灰度小图，用来判断"画面有没有变"。

        **为什么要有这个**：OCR 是唯一的 CPU 大头（实测 2560x1440 要 825ms，
        而截屏只要 5.9ms，所以瓶颈完全在 OCR）。但读屏的场景里，屏幕
        **绝大部分时间是静止的** —— 每一轮都对同一幅画面重跑 OCR 纯属浪费。

        这个签名把任意尺寸的帧下采样成 48x48 灰度（固定大小，所以区域改了
        尺寸也能和历史签名比较），耗时在 0.1ms 量级，比 OCR 便宜三四个数量级。
        """
        import numpy as np

        h, w = frame.shape[:2]
        if h < 2 or w < 2:
            return None
        ys = np.linspace(0, h - 1, 48).astype(np.int32)
        xs = np.linspace(0, w - 1, 48).astype(np.int32)
        small = frame[ys][:, xs]
        # RGB -> 灰度（整数近似，比 np.dot 快且够用）
        return (small[..., 0].astype(np.int32) * 77
                + small[..., 1].astype(np.int32) * 150
                + small[..., 2].astype(np.int32) * 29) >> 8

    def _frame_unchanged(self, rid: str, sig) -> bool:
        """和上一帧比，画面是否基本没变。"""
        if sig is None:
            return False
        prev = self._frame_sig.get(rid)
        self._frame_sig[rid] = sig
        if prev is None or getattr(prev, "shape", None) != sig.shape:
            return False
        import numpy as np

        diff = np.abs(sig - prev)
        # 判据：明显变化的像素占比极低就算"没变"。
        # 阈值 12 灰度、0.3% 面积 —— 光标闪烁、抗锯齿微动都会被忽略，
        # 而真的多出一行字/翻页一定会超过。
        changed_ratio = float((diff > 12).mean())
        return changed_ratio < 0.003

    def _clean_raw(self, ocr_lines):
        """OCR 原始行 -> 清洗后的 `[(text, top, left), ...]`。

        轮询和「截图朗读」共用这一套规则，保证"界面显示的文字 == 会念的文字"，
        也保证两条路径的过滤行为完全一致（否则同屏内容两条路读出来不一样）。
        """
        raw = [(ln.text, ln.top, ln.left)
               for ln in ocr_lines if ln.score >= self.ecfg.min_score]
        if len(raw) > self.ecfg.max_lines_per_region:
            raw = raw[: self.ecfg.max_lines_per_region]
        # 清洗：去掉行首编号、带圈数字这类"屏幕上看没问题、念出来很怪"的东西
        if self.ecfg.clean_text:
            raw = [(clean_for_tts(t), top, left) for t, top, left in raw]
        if self.ecfg.drop_unspeakable:
            raw = [x for x in raw if has_speakable(x[0])]
        return [x for x in raw if x[0].strip()]

    def read_frame_once(self, frame):
        """对**一张已经抓好的图**做一次 OCR，返回 `[(text, top, left), ...]`。

        「截图朗读」专用：**不走检测轮询**。
        这样就不会有"OCR 每轮重跑 -> 内容抖动 -> 整段卡片重建 -> 又读一遍"
        的重复朗读，也不依赖整段模式的"内容稳定 0.75 秒"（静态图必然稳定）。

        必须拿 `_ocr_lock`：检测线程可能正在用同一个 ONNX 会话。
        **带超时**，拿不到就直接返回空 —— 不然后台 OCR 会把检测线程卡死。
        """
        got = self._ocr_lock.acquire(timeout=6.0)
        if not got:
            log.warning("截图朗读拿不到 OCR 锁，放弃这次识别")
            return []
        try:
            ocr = self._ocr or self._ensure_ocr()
            if ocr is None:
                return []
            try:
                res = ocr.read(frame)
            except Exception as e:
                log.warning("截图朗读 OCR 失败：%s", e)
                return []
        finally:
            self._ocr_lock.release()
        return self._clean_raw(res.lines)

    def _poll_region(self, r: Region) -> None:
        ocr = self._ocr or self._ensure_ocr()
        if ocr is None:
            return
        frame = self._grab(r)
        if frame is None:
            self._emit("error", where="grab", region=r.id, error="抓帧失败")
            return

        # ---- 帧不变就直接跳过 OCR（CPU 优化，收益最大的一处）----
        #
        # 注意：跳过 OCR 就没有 observe()，卡片会因为累计 misses 被误删，
        # 所以必须调 touch_region() 给它们续命。
        if self.ecfg.skip_unchanged_frame:
            sig = self._frame_signature(frame)
            if self._frame_unchanged(r.id, sig):
                self.store.touch_region(r.id)
                self.stats_skipped_frames += 1
                return

        # **OCR 必须带超时地拿锁**，不能无限等。
        # 「截图朗读」也在用同一个 ONNX 会话；万一它那边卡住（大图、驱动问题），
        # 检测线程如果在这里死等，整块识别就永久停摆 ——
        # 用户实测就是"截图朗读之后识别一直卡住"。宁可这一轮跳过。
        got = self._ocr_lock.acquire(timeout=3.0)
        if not got:
            log.warning("OCR 被别的任务占着，本轮跳过（区域 %s）", r.name)
            self._emit("warning", where="ocr_busy", region=r.id)
            return
        try:
            res = ocr.read(frame)
        finally:
            self._ocr_lock.release()
        raw = self._clean_raw(res.lines)

        lines = [x[0] for x in raw]
        spans = [(x[1], x[2]) for x in raw]

        # ---- 整段模式 ----
        #
        # 关键：OCR 是按**视觉行**给出的。一段话折成几行时，每行都是"半截话"，
        # 逐行合成听起来就是一顿一顿、语义破碎。
        # 处理顺序（用户明确要求）：
        #   1. 按图上位置从上到下排序
        #   2. 合并折行（行尾没标点 = 自动折行）
        #   3. 按优先级断句：先句号，再逗号，绝不在句子中间切
        if r.mode == "paragraph":
            sentences = merge_lines_to_sentences(
                lines,
                max_chars=self.cfg.segment.max_chars or 80,
                spans=spans,
            )
            if not self._paragraph_settled(r, sentences):
                # 内容还在变（滚动/翻页中）——**不建卡、不合成**。
                # 否则会不停播报"刚出现的前几个字"，等页面停下来想完整读时，
                # 卡片已经被这些碎片搅乱，顺序也乱了。
                return
            lines = sentences
        elif self.ecfg.segment_enabled or self.store.cfg.granularity == "sentence":
            opts = SegmentOptions(
                split_chars=self.cfg.segment.split_chars,
                soft_split_chars=self.cfg.segment.soft_split_chars,
                min_chars=self.cfg.segment.min_chars,
                max_chars=self.cfg.segment.max_chars,
            )
            expanded: List[str] = []
            for ln in lines:
                segs = split_text(ln, opts)
                expanded.extend(segs or [ln])
            lines = expanded

        out = self.store.observe(r.id, lines)
        self._after_observe(r, out)

    def _paragraph_settled(self, r: Region, sentences: List[str]) -> bool:
        """整段模式：内容稳定了吗？稳定才允许建卡。

        返回 True 表示"可以建卡了"，False 表示"还在变，先别动"。

        判定很直白：内容指纹变了就重新计时；连续 `paragraph_settle_s` 秒
        指纹不变才算稳定。稳定之后如果指纹没再变过，就**不再重复建卡**
        （否则每一轮轮询都会 observe 一次）。
        """
        st = self._para_state.setdefault(r.id, {"key": "", "since": 0.0, "committed": ""})
        # **防御性补齐**：`_para_state` 里的条目可能是别处（play_paragraph）
        # 先建出来的，字段不一定齐。缺字段时用 setdefault 补，而不是 KeyError。
        st.setdefault("key", "")
        st.setdefault("since", 0.0)
        st.setdefault("committed", "")
        key = paragraph_key("".join(sentences))
        now = time.time()

        if key != st["key"]:
            st["key"] = key
            st["since"] = now
            st["committed"] = ""       # 内容变了，之前确认过的作废

            # **内容一变，旧卡片立刻清掉**（用户要求）。
            #
            # 注意必须**逐张发 card_died**：CardStore._destroy() 本身不发事件，
            # 而界面只在收到 card_added/rearmed/died/cleared 时才刷新列表 ——
            # 以前这里只调 clear_region()，卡片在 store 里没了，
            # 界面上却一直留着，看起来就是"缓存的卡片卡住列表"。
            doomed = [c for c in self.store.sorted_cards() if c.region_id == r.id]
            self.store.clear_region(r.id)
            self._drop_pending(r.id)          # 还没合成的也一并作废
            for c in doomed:
                self._emit("card_died", card_id=c.id, region_id=r.id, text=c.text[:40])

            self._playing_regions.discard(r.id)
            if doomed:
                self._emit("paragraph_cleared", region_id=r.id, cards=len(doomed))
            self._emit("paragraph_settling", region_id=r.id,
                       chars=len("".join(sentences)))
            return False

        if not key:
            return False               # 空白，没什么可建的

        if key == st["committed"]:
            return False               # 这一份已经建过卡了，别重复

        if now - st["since"] < self.ecfg.paragraph_settle_s:
            return False               # 还没稳够

        st["committed"] = key
        self._emit("paragraph_settled", region_id=r.id, sentences=len(sentences))
        return True

    def _after_observe(self, r: Region, out: ObserveResult) -> None:
        if not (out.added or out.changed or out.died):
            return
        if out.died:
            for c in out.died:
                self._emit("card_died", card_id=c.id, region_id=r.id, text=c.text[:40])
        for c in out.added:
            self._enqueue(c)
            # 正在整段朗读中产生的新句子，**直接接进播放队列**（用户要求）。
            # 这样"后面的句子合成好了"和"前面正在播"可以并行，
            # 不用等整段全部就绪。
            if r.id in self._playing_regions:
                c.queued = True
                self._emit("paragraph_appended", region_id=r.id, text=c.text[:40])
            # **自动朗读**：勾选后每张新卡都排进队列，自动顺序播一遍。
            # 注意放在 `_playing_regions` 判断之外 —— 那个只针对"整段朗读中"，
            # 而自动朗读对**单行模式**同样要生效（用户明确要求"类似于自动点全部阅读"）。
            elif self.ecfg.auto_read_all:
                c.queued = True
                self._emit("autoread_queued", region_id=r.id, text=c.text[:40])
            self._emit("card_added", card_id=c.id, region_id=r.id, text=c.text[:60])
        for c in out.rearmed:
            self._enqueue(c)
            self._emit("card_rearmed", card_id=c.id, region_id=r.id, text=c.text[:60])

    def _enqueue(self, c: Card) -> None:
        with self._lock:
            if c.id in self._claimed:
                return
            self._claimed.add(c.id)
        c.audio_state = "pending"
        with self._lock:
            self._queued_at[c.id] = time.time()   # 记录排队时刻（卡住恢复用）
        self._synth_q.put(c.id)

    def _drop_pending(self, rid: str) -> int:
        """把已经不存在（被清掉）的卡片从合成认领表里摘掉。

        内容一变，之前排队的句子再合成出来也没用了。
        合成线程本来就会 `store.card(cid) is None` 时跳过（计入 skipped_dead），
        这里只是顺手把认领表清干净，免得它一直涨。
        """
        with self._lock:
            gone = {cid for cid in self._claimed if self.store.card(cid) is None}
            self._claimed -= gone
        return len(gone)

    # ---------------------------------------------------------- 合成线程 --
    def _synth_loop(self) -> None:
        _last_sweep = 0.0
        while not self._stop.is_set():
            try:
                cid = self._synth_q.get(timeout=0.3)
            except queue.Empty:
                # 空闲时顺手清理过期缓存（用户要求"生成 30 秒后自动清理"）
                now = time.time()
                if self.cache.ttl_s > 0 and now - _last_sweep >= 5.0:
                    _last_sweep = now
                    try:
                        self.cache.sweep()
                    except Exception as e:
                        log.debug("缓存清理失败：%s", e)
                continue
            if not cid or self._stop.is_set():
                if not cid:
                    continue
                break
            c = self.store.card(cid)
            if c is None:
                with self._lock:
                    self._claimed.discard(cid)
                continue
            # **不能让合成线程因为一张卡失败就整体死掉** ——
            # 那会导致后面所有句子永远停在"合成中"，界面看起来就是卡住。
            try:
                self._synth_one(c)
            except Exception as e:
                log.exception("合成失败 card=%s：%s", cid, e)
                c.audio_state = "failed"
                c.audio_error = f"{type(e).__name__}: {e}"[:300]
                self._emit("card_failed", card_id=cid,
                           region_id=c.region_id, error=c.audio_error)
                with self._lock:
                    self._claimed.discard(cid)

    def _on_tier_change(self, tier: Tier, gpu) -> None:
        self.tier = tier
        log.warning("显存降档 -> %s（%s），采样步数调整为 %d", tier.key, tier.label, tier.sample_steps)
        self._emit("tier", tier=tier.key, label=tier.label, sample_steps=tier.sample_steps)

    @property
    def sample_steps(self) -> int:
        """采样步数：显存充裕用配置值，紧张的档位用更少步数换速度。"""
        base = self.cfg.tts.sample_steps
        if self.tier is not None:
            return min(base, self.tier.sample_steps)
        return base

    def _synth_one(self, c: Card) -> None:
        key = make_key(
            c.text,
            ref_audio=self.cfg.tts.ref_audio,
            prompt_text=self.cfg.tts.prompt_text,
            speed_factor=self.cfg.tts.speed_factor,
            text_lang=self.cfg.tts.text_lang,
            extra=f"{self.backend.name}",
        )
        # **"重新生成"必须绕过缓存。**
        # 用户反馈"个别语音会少半句话"——如果那份坏音频已经在缓存里，
        # 重试时命中缓存就会把同一份坏音频拿回来，看起来像"重试没用"。
        # GPT-SoVITS 是采样生成的，重新合成有机会得到正常的一份。
        bypass = c.id in self._force_synth
        if bypass:
            with self._lock:
                self._force_synth.discard(c.id)
        hit = None if bypass else self.cache.get(key)
        if hit is not None:
            c.audio = hit
            c.audio_state = "ready"
            c.from_cache = True
            c.synth_ms = 0.0
            self.stats["cache_hits"] += 1
            self._emit("card_ready", card_id=c.id, from_cache=True, ms=0.0)
            self._release(c.id)
            return

        t0 = time.perf_counter()
        try:
            data = self.backend.synthesize(c.text, sample_steps=self.sample_steps)
        except Exception as e:
            # **TTS 服务正在启动时不要判失败。**
            # 首次加载模型实测要 28 秒，而「截图朗读」往往在这之前就发出了请求。
            # 以前直接判失败 -> 卡片上写"失败"，用户看起来就是"OCR 有结果但不读"。
            # 这里把"连不上服务"单独识别出来，等一会儿再重排（最多约 60 秒）。
            msg = repr(e)
            warmup = any(k in msg for k in (
                "ConnectionError", "Connection refused", "NewConnectionError",
                "Max retries exceeded", "Failed to establish", "timed out"))
            if warmup and c.warmup_retries < 60:
                c.warmup_retries += 1
                c.audio_state = "pending"
                c.audio_error = "TTS 服务正在启动…"
                self.stats["synth_wait_service"] = self.stats.get("synth_wait_service", 0) + 1
                if c.warmup_retries in (1, 5, 15, 30, 50):
                    log.info("TTS 服务还没就绪，等它启动（第 %d 次重试）", c.warmup_retries)
                self._release(c.id)
                time.sleep(1.0)
                if self.store.card(c.id) is not None:
                    self._enqueue(c)
                return
            c.audio_state = "failed"
            c.audio_error = str(e) or type(e).__name__
            self.stats["synth_fail"] += 1
            log.warning("合成失败「%s」：%s", c.text[:20], e)
            self._emit("card_failed", card_id=c.id, text=c.text[:40], error=c.audio_error)
            self._release(c.id)
            return

        # 排队期间这句话可能已经从屏幕上消失了。合成结果照样进缓存
        # （这句话再出现就是 0 延迟），但不再往界面推状态。
        alive = self.store.card(c.id) is not None
        if not alive:
            self.stats["skipped_dead"] += 1
            if data:
                self.cache.put(key, data)
            self._release(c.id)
            return

        c.synth_ms = (time.perf_counter() - t0) * 1000
        if not data:
            c.audio_state = "failed"
            c.audio_error = "空音频"
            self.stats["synth_fail"] += 1
            self._release(c.id)
            return
        # **校验 WAV 里真的有采样点。**
        # 有时后端返回的是一个"合法的 WAV 头 + 0 帧数据"，`if not data` 拦不住，
        # 到播放阶段才炸（而且错误消息是空的）。这里提前判掉，给出明确原因。
        try:
            import io as _io
            import wave as _wave

            with _wave.open(_io.BytesIO(data), "rb") as _w:
                if _w.getnframes() <= 0:
                    raise ValueError("0 帧")
        except Exception as e:
            c.audio_state = "failed"
            c.audio_error = f"合成结果为空音频（{e}）"
            self.stats["synth_fail"] += 1
            log.warning("「%s」合成结果是空音频：%s", c.text[:20], e)
            self._emit("card_failed", card_id=c.id, region_id=c.region_id,
                       error=c.audio_error)
            self._release(c.id)
            return

        c.audio = data
        c.audio_state = "ready"
        c.from_cache = False
        # 重试时**不写回缓存**：如果这一份还是坏的，别把坏音频固化下来
        if not bypass:
            self.cache.put(key, data)
        self.stats["synth_ok"] += 1
        self._emit("card_ready", card_id=c.id, from_cache=False, ms=round(c.synth_ms, 1))
        self._release(c.id)

    def _release(self, cid: str) -> None:
        with self._lock:
            self._claimed.discard(cid)
            self._queued_at.pop(cid, None)   # 处理完了，不再算"卡住"

    # ------------------------------------------------------------ 播放 --
    def _autoplay_loop(self) -> None:
        """整段朗读：把**用户点过播放**的句子按顺序连续读出去。

        用户的要求：
          * 页面滚动、内容还在变的时候**不合成、不播放**
          * 页面稳定后，由用户**手动点播放按钮**，从上到下顺序读

        所以这里不再自动开播，只消费被 `play_paragraph()` 排进队列的卡片：
          * 严格按 `seq`（=从上到下的出现顺序）取下一句，
            **前面一句没好就等它**，绝不跳到后面去，否则听起来是乱的
          * 合成是后台线程并行做的，所以后面几句往往已经准备好了
        """
        while not self._stop.is_set():
            try:
                if self._play_paused:
                    self._stop.wait(0.2)
                    continue
                c = self.store.next_auto_card()
                if c is None:
                    self._stop.wait(0.2)
                    continue
                if c.audio_state == "failed":
                    # 合成失败就跳过，别把整段卡死
                    c.auto_done = True
                    c.queued = False
                    self._emit("autoplay_skipped", card_id=c.id, text=c.text[:40])
                    continue
                if not c.ready:
                    # 还没合成好 —— 原地等它（保持顺序）。
                    # **顺手补一次入队**：如果这张卡是"建出来了但没人排进合成队列"
                    # （截图朗读早期就是这么绕过 _after_observe 的），
                    # 这里兜一下，避免永远等下去。`_enqueue` 对已在队列里的 id 是幂等的。
                    if c.audio_state == "pending":
                        self._enqueue(c)
                    self._stop.wait(0.15)
                    continue
                # 已经在播这一句了就等它播完
                if self.store.playing_id == c.id:
                    self._stop.wait(0.1)
                    continue

                c.auto_done = True
                c.queued = False
                self._emit("autoplay_next", card_id=c.id, text=c.text[:60], seq=c.seq)
                try:
                    self.play_card(c.id)
                    self._wait_play_done(c.id)
                except Exception as e:
                    log.warning("整段朗读播放失败：%s", e)
                # 句子之间的自然停顿（中文朗读约 300ms 比较自然）
                self._stop.wait(0.35)
            except Exception as e:
                # **外层兜底：任何意外都不能让朗读线程死掉**，
                # 死了以后"全部播放"就再也不响，看起来就是卡住。
                log.exception("整段朗读线程异常：%s", e)
                self._stop.wait(0.3)

    def play_paragraph(self, rid: str) -> int:
        """顺序朗读某个区域：从**第一项**开始，从上到下读完。

        用户手动触发（不再自动播）。返回排进队列的句子数。

        进入"朗读中"状态后，**这个区域后来新生成的句子会自动接进队列**，
        所以第一句一好就能开播，后面边播边补。

        单行模式也能用 —— 队列本身不关心模式，只是单行模式的卡片是逐行的。
        """
        r = self.store.region(rid)
        if r is None:
            return 0
        self._playing_regions.add(rid)
        n = self.store.queue_region(rid)
        self._stop_play.clear()
        # **这里必须用完整结构，不能是空 dict。**
        # 以前写的是 `setdefault(rid, {})["queued"] = True` —— 而
        # `_paragraph_settled` 用的是 `setdefault(rid, {"key": ...})`。
        # `setdefault` **不会合并**：只要这里先塞了 `{}`，那边就永远拿回 `{}`，
        # 于是 `st["key"]` 直接 KeyError: 'key'，整个区域的整段识别全部失败
        # （用户日志里就是这个错，反复出现）。
        st = self._para_state.setdefault(
            rid, {"key": "", "since": 0.0, "committed": ""})
        st["queued"] = True
        self._emit("paragraph_queued", region_id=rid, sentences=n)
        return n

    def play_paragraphs(self, rids: Optional[List[str]] = None) -> int:
        """一次播多个区域（悬浮窗的「全部播放」用）。

        默认把**所有区域**（不管单行还是整段）都按顺序排进队列 ——
        用户要求单行模式的「全部播放」和整段一样能顺序播放。
        队列全局按 seq 出队，所以出来的顺序就是界面上从上到下的顺序。
        """
        if rids is None:
            rids = [r.id for r in self.store.regions if r.enabled]
        total = 0
        for rid in rids:
            total += self.play_paragraph(rid)
        return total

    def set_volume(self, volume: int) -> int:
        """调播放音量 0-100（滑动条实时生效，不用重启）。"""
        try:
            self.player.volume = int(volume)
        except Exception as e:
            log.warning("设置音量失败：%s", e)
            return 0
        return getattr(self.player, "volume", 100)

    def stop_paragraph(self, rid: str) -> int:
        """停止某个区域的整段朗读。"""
        self._playing_regions.discard(rid)
        n = self.store.unqueue_region(rid)
        self.stop_playback()
        if n:
            self._emit("paragraph_stopped", region_id=rid, sentences=n)
        return n

    def stop_all_paragraphs(self) -> int:
        n = 0
        for rid in list(self._playing_regions):
            n += self.stop_paragraph(rid)
        return n

    def _wait_play_done(self, cid: str, timeout_s: float = 180.0) -> None:
        """等某张卡片播完。

        **不能因为"卡片没了"就提前返回**：那样朗读循环会立刻去播下一句，
        `play_card()` 里的 `sd.stop()` 就把上一句掐断了 ——
        用户听到的是"最后几个字没了"，很容易误判成合成被截断。

        所以判据只有一个：`playing_id` 不再是这张卡（播完了，或被新播放顶掉）。
        """
        t0 = time.time()
        while not self._stop.is_set() and time.time() - t0 < timeout_s:
            if self.store.playing_id != cid:
                return
            self._stop.wait(0.1)

    def play_card(self, cid: str) -> bool:
        """播放某张卡片。同一时刻只播一张，新的会打断旧的。"""
        c = self.store.card(cid)
        if c is None:
            return False
        if c.audio is None:
            # 还没合成好：等它就绪后自动播（避免用户点了没反应）
            self._enqueue(c)
            self._emit("play_pending", card_id=cid)
            self._pending_play = cid
            return False
        self.stop_playback()
        self.store.mark_playing(cid)
        # **标记"真的开始播了"**。`region_playback_done` 会要求这个标记，
        # 否则清理定时器会在 autoplay 的"已出队但还没开播"窗口里误判完成，
        # 把卡片删掉导致静默不播（实测过的 bug）。
        c.played_once = True
        self._stop_play.clear()
        self._emit("play_started", card_id=cid, text=c.text[:40])

        def _run():
            try:
                self.player.play_wav(c.audio)
            except Exception as e:
                # **被 stop_playback() 打断是正常现象，不是错误。**
                # `sd.stop()` 会中断正在阻塞的 `sd.play()`，抛出来的异常
                # 消息经常是空的 —— 以前直接记成"播放失败："，日志里什么都看不出来。
                if self._stop_play.is_set():
                    log.info("播放被停止（预期）：%s", repr(e))
                else:
                    log.warning("播放失败：%r", e)
                    self._emit("error", where="play", error=repr(e)[:300])
            finally:
                if self.store.playing_id == cid:
                    self.store.mark_playing(None)
                self._emit("play_finished", card_id=cid)

        self._play_thread = threading.Thread(target=_run, name="play", daemon=True)
        self._play_thread.start()
        return True

    def stop_playback(self) -> None:
        self._stop_play.set()
        cid = self.store.playing_id
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass
        if cid:
            self.store.mark_playing(None)
            self._emit("play_stopped", card_id=cid)

    @property
    def playing_id(self) -> Optional[str]:
        return self.store.playing_id

    # ------------------------------------------------------------ 其他 --
    def clear_cards(self) -> int:
        self.stop_playback()
        n = self.store.clear()
        self._emit("cleared", cards=n)
        return n

    def stats_snapshot(self) -> Dict[str, object]:
        return {**self.stats, **self.store.stats()}
