"""读屏台的数据模型：区域 + 卡片 + 生命周期。

**这一层是纯逻辑，不依赖 Qt、不依赖音频设备**，所以可以直接单测，
而且窗口模式和悬浮字幕条模式可以共用同一套规则。

核心语义（用户需求）：
    * 先框选 1~N 个屏幕区域，然后持续识别
    * 识别出的文字按「行」或「句」变成**卡片**
    * 当某段文字从框选区域里消失时，**自动销毁**对应的卡片、语音、播放按钮

两个必须处理的现实问题：
  1. **OCR 会抖**：同一行两次识别结果可能略有差异（实测出现过
     `deepseek` / `dleepseek` / `dleepseekHARNESS`）。所以匹配用
     「去标点+小写」归一化 + 相似度，不能直接字符串相等。
  2. **OCR 会漏帧**：游戏画面、遮挡、字体变化都可能让某行某一帧识别不到。
     所以消失不是立刻销毁，要连续漏 `destroy_after_misses` 次才销毁。
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Tuple

log = logging.getLogger(__name__)

Rect = Tuple[int, int, int, int]  # (left, top, width, height) 相对显示器左上角


# --------------------------------------------------------------------------- #
def line_key(s: str) -> str:
    """匹配用归一化键：去掉空白与标点、转小写。"""
    return re.sub(r"[^\w\u4e00-\u9fff]", "", s or "").lower()


def similarity(a: str, b: str) -> float:
    ka, kb = line_key(a), line_key(b)
    if not ka or not kb:
        return 0.0
    if ka == kb:
        return 1.0
    return SequenceMatcher(None, ka, kb).ratio()


# --------------------------------------------------------------------------- #
@dataclass
class Region:
    """一个框选区域。"""

    id: str
    name: str
    rect: Rect
    monitor_index: int = 0
    enabled: bool = True
    color: str = "#4ea1ff"
    # 识别模式：
    #   "line"      单行模式 —— 每行一张卡，手动点播（原来的行为）
    #   "paragraph" 整段模式 —— 把折行的多行还原成句子，自动按顺序连续朗读
    mode: str = "line"
    # **临时区域**（「截图朗读」用）。
    #
    # 截图朗读的流程是"框一段 -> 立刻读一遍 -> 读完自动删掉"，所以这种区域
    # **绝不能进持久化**（否则每次用完都在 config.json 里留一条垃圾）。
    # 用这个标记把它和用户手框的常驻区域区分开。
    temp: bool = False

    @classmethod
    def create(cls, name: str, rect: Rect, monitor_index: int = 0, color: str = "#4ea1ff",
               mode: str = "line", temp: bool = False) -> "Region":
        return cls(id=uuid.uuid4().hex[:8], name=name, rect=tuple(rect),
                   monitor_index=monitor_index, color=color, mode=mode, temp=temp)

    def label(self) -> str:
        l, t, w, h = self.rect
        tag = "整段" if self.mode == "paragraph" else "单行"
        if self.temp:
            tag = "截图朗读"
        return f"{self.name}  [{tag}]  ({w}×{h} @ {l},{t})"


@dataclass
class Card:
    """一句待朗读的文字，以及它自己的语音与播放状态。"""

    id: str
    region_id: str
    text: str
    key: str
    first_seen: float
    last_seen: float
    # 单调递增的创建序号 —— **整段模式按它决定朗读顺序**。
    # 不能用 first_seen：同一批 observe() 建出来的卡片时间戳是一样的，
    # 再按文本排序就变成字典序了，读出来语无伦次。
    seq: int = 0
    misses: int = 0
    revision: int = 0
    audio: Optional[bytes] = None
    audio_state: str = "pending"   # pending | ready | failed | none
    audio_error: str = ""
    play_state: str = "idle"       # idle | playing
    # 整段模式：轮到它时要自动朗读，不需要用户点
    auto_play: bool = False
    auto_done: bool = False        # 自动朗读队列里是否已处理过
    # 整段模式：用户点了「整段朗读」之后，这张卡被排进队列
    queued: bool = False
    # 已经该销毁，但因为正在播放而暂时留着（播完立刻收掉）
    doomed: bool = False
    synth_ms: float = 0.0
    from_cache: bool = False
    # 这张卡被"救"过几次（合成长时间没进展时重新排队）。
    # 超过 1 次就直接判失败 —— 否则会在"重排-卡住-重排"之间死循环。
    synth_retries: int = 0
    # **真正开始播过没有**。
    #
    # 为什么必须有这个标记：`_autoplay_loop` 的顺序是
    #   c.auto_done = True; c.queued = False; ... self.play_card(c.id)
    # 而 `mark_playing()` 要等进到 `play_card` 里面才发生。
    # 中间有一个窗口期：卡片"没排队、auto_done=True、playing_id 不是它"。
    # 「截图朗读」的清理定时器（150ms 一次）如果正好在这个窗口里检查，
    # 就会误判"已经读完了"，把区域删掉、卡片销毁 ——
    # 于是紧接着的 `play_card` 找不到卡片，**静默不播**。
    # 这就是用户看到的"OCR 有结果、TTS 也成功、但就是没声音"。
    played_once: bool = False
    # 因为"TTS 服务还在启动"而重试的次数（首次加载模型实测要 28 秒）。
    # 单独一个计数：不要和 `synth_retries`（合成卡住的救援计数）混用，
    # 否则服务预热的重试会被"卡住救援"当成"已经救过一次"直接判失败。
    warmup_retries: int = 0

    @property
    def ready(self) -> bool:
        return self.audio is not None and self.audio_state == "ready"

    def display(self, limit: int = 80) -> str:
        t = self.text.strip()
        return t if len(t) <= limit else t[: limit - 1] + "…"


@dataclass
class BoardConfig:
    # 连续多少轮看不到就销毁（OCR 会漏帧，不能一帧没看到就删）
    destroy_after_misses: int = 3
    # 匹配阈值：相似度 >= 这个值视为"同一句"
    match_threshold: float = 0.85
    # 文本变化但相似度 >= 这个值时，只更新显示文字，**沿用已有语音**（不重合成）
    resynth_threshold: float = 0.92
    # line = 每个 OCR 行一张卡片（弹幕场景）；sentence = 行内再按标点切句
    granularity: str = "line"
    # 卡片总数上限，超出时丢弃最久没见到的
    max_cards: int = 300


@dataclass
class ObserveResult:
    added: List[Card] = field(default_factory=list)
    kept: List[Card] = field(default_factory=list)
    changed: List[Card] = field(default_factory=list)   # 文本变了
    rearmed: List[Card] = field(default_factory=list)   # 文本变了且需要重新合成
    died: List[Card] = field(default_factory=list)      # 已销毁


class CardStore:
    """区域与卡片的集合，以及"随屏幕内容生灭"的全部规则。"""

    def __init__(self, config: Optional[BoardConfig] = None) -> None:
        self.cfg = config or BoardConfig()
        self._regions: Dict[str, Region] = {}
        self._cards: Dict[str, Card] = {}
        self._playing: Optional[str] = None
        self._seq = 0
        self._palette = ["#4ea1ff", "#ffb454", "#7ee787", "#ff7b72", "#d2a8ff", "#79c0ff"]

    # ------------------------------------------------------------ 区域 --
    @property
    def regions(self) -> List[Region]:
        return list(self._regions.values())

    def region(self, rid: str) -> Optional[Region]:
        return self._regions.get(rid)

    def add_region(self, name: str, rect: Rect, monitor_index: int = 0,
                   color: Optional[str] = None, mode: str = "line",
                   rid: Optional[str] = None) -> Region:
        color = color or self._palette[len(self._regions) % len(self._palette)]
        r = Region.create(name or f"区域{len(self._regions) + 1}", rect, monitor_index, color, mode)
        if rid:
            r.id = rid          # 恢复上次保存的区域，保留 id 让卡片/角标能对上
        self._regions[r.id] = r
        return r

    def restore_regions(self, items: List[dict]) -> int:
        """从配置里恢复区域（含单行/整段模式）。"""
        n = 0
        for it in items:
            try:
                # **防御历史数据**：0.7.6 及更早会把「截图朗读」的临时区域
                # 写进配置。那种区域恢复出来只是个没用的常驻区域，
                # 还会被轮询（进而和整段朗读抢状态）。直接跳过。
                if str(it.get("name") or "") == "截图朗读":
                    log.info("跳过历史遗留的临时区域（截图朗读）")
                    continue
                r = self.add_region(
                    it.get("name") or "区域",
                    tuple(it["rect"]),
                    int(it.get("monitor_index") or 0),
                    it.get("color"),
                    it.get("mode") or "line",
                    it.get("id") or None,
                )
                r.enabled = bool(it.get("enabled", True))
                r.temp = bool(it.get("temp", False))
                n += 1
            except Exception as e:
                log.warning("恢复区域失败（%s）：%s", it, e)
        return n

    def update_region(self, rid: str, **kw) -> Optional[Region]:
        r = self._regions.get(rid)
        if not r:
            return None
        for k, v in kw.items():
            if hasattr(r, k):
                setattr(r, k, v)
        return r

    def remove_region(self, rid: str) -> int:
        """删区域时连带销毁它的全部卡片，返回销毁数。"""
        self._regions.pop(rid, None)
        doomed = [c for c in self._cards.values() if c.region_id == rid]
        for c in doomed:
            self._destroy(c)
        return len(doomed)

    # ------------------------------------------------------------ 卡片 --
    @property
    def cards(self) -> List[Card]:
        return list(self._cards.values())

    def card(self, cid: str) -> Optional[Card]:
        return self._cards.get(cid)

    def cards_of(self, rid: str) -> List[Card]:
        return [c for c in self._cards.values() if c.region_id == rid]

    def sorted_cards(self, region_order: Optional[List[str]] = None) -> List[Card]:
        """按区域分组、组内按**创建顺序**排序。

        组内必须用 seq（创建序号）而不是 first_seen：同一批 observe() 建出来的
        卡片时间戳相同，退回按文本排序会变成字典序 —— 整段模式读出来语无伦次。
        """
        order = region_order or [r.id for r in self._regions.values()]
        idx = {rid: i for i, rid in enumerate(order)}
        return sorted(
            self._cards.values(),
            key=lambda c: (idx.get(c.region_id, 999), c.seq, c.first_seen),
        )

    def next_auto_card(self) -> Optional[Card]:
        """整段模式：取下一句该朗读的卡片（按 seq 顺序，不跳过后面的）。

        **只有被用户点过「整段朗读」排进队列的卡片才会返回** ——
        不再自动播放。原因是滚动时内容一直在变，自动播会不停念"刚出现的
        前几个字"，等页面停下来反而因为卡片乱序没法顺序读。

        返回 None 表示队列空了。
        """
        pending = [
            c for c in self._cards.values()
            if c.queued and not c.auto_done
        ]
        if not pending:
            return None
        return min(pending, key=lambda c: c.seq)

    def queue_region(self, rid: str) -> int:
        """把某个区域的卡片按 seq（=从上到下的出现顺序）排进朗读队列。

        返回排入的卡片数。已经在播的那张不重排。
        """
        cards = sorted(
            (c for c in self._cards.values() if c.region_id == rid),
            key=lambda c: c.seq,
        )
        n = 0
        for c in cards:
            c.auto_done = False
            c.queued = True
            n += 1
        return n

    def unqueue_region(self, rid: str) -> int:
        """取消某区域的朗读队列（内容变了 / 用户停止时用）。"""
        n = 0
        for c in self._cards.values():
            if c.region_id == rid and c.queued:
                c.queued = False
                c.auto_done = True
                n += 1
        return n

    def _new_card(self, rid: str, text: str, now: float) -> Card:
        r = self._regions.get(rid)
        self._seq += 1
        c = Card(
            id=uuid.uuid4().hex[:10],
            region_id=rid,
            text=text,
            key=line_key(text),
            first_seen=now,
            last_seen=now,
            # 整段模式的句子要按出现顺序朗读，所以给个单调序号
            seq=self._seq,
            auto_play=bool(r and r.mode == "paragraph"),
        )
        self._cards[c.id] = c
        self._evict_if_needed()
        return c

    def _evict_if_needed(self) -> None:
        while len(self._cards) > self.cfg.max_cards:
            # 别把正在播的那张淘汰掉，否则播放会被掐断
            pool = [c for c in self._cards.values() if c.id != self._playing]
            if not pool:
                break
            victim = min(pool, key=lambda c: c.last_seen)
            log.debug("卡片超限，销毁最旧的：%s", victim.text[:20])
            self._destroy(victim)

    def _destroy(self, c: Card, force: bool = False) -> bool:
        """销毁一张卡片。返回是否真的删掉了。

        **正在播放的卡片默认不删**，只打个 `doomed` 标记，等它播完再删。

        为什么：单行模式下卡片在文字离开屏幕约 1.2 秒后就会被销毁
        （destroy_after_misses=3 × 轮询 0.4s），而一句话要念 4~5 秒。
        卡片先死 -> 朗读循环立刻去播下一句 -> `sd.stop()` 把上一句掐断，
        用户听到的就是"最后几个字没了"。这个坑很容易被误判成"合成被截断"。

        `force=True`（用户主动清空/删区域）时不保护，立即删。
        """
        if not force and self._playing == c.id:
            c.doomed = True          # 先记着，播完再收
            return False
        if self._playing == c.id:
            self._playing = None
        c.play_state = "idle"
        c.audio = None
        self._cards.pop(c.id, None)
        return True

    # ------------------------------------------------------------ 观察 --
    def observe(self, rid: str, lines: Iterable[str], now: Optional[float] = None) -> ObserveResult:
        """喂入某个区域这一轮识别到的所有行，返回本轮的生灭结果。"""
        now = now if now is not None else time.time()
        res = ObserveResult()
        if rid not in self._regions:
            return res

        existing = self.cards_of(rid)
        used: set = set()

        for raw in lines:
            text = (raw or "").strip()
            if not text or not line_key(text):
                continue

            best, best_score = None, 0.0
            for c in existing:
                if c.id in used:
                    continue
                s = similarity(c.text, text)
                if s > best_score:
                    best, best_score = c, s

            if best is not None and best_score >= self.cfg.match_threshold:
                used.add(best.id)
                best.misses = 0
                best.last_seen = now
                if line_key(best.text) != line_key(text):
                    best.text = text
                    best.revision += 1
                    if best_score >= self.cfg.resynth_threshold:
                        # 只是 OCR 抖动，沿用已有语音，避免白合成
                        res.changed.append(best)
                    else:
                        best.audio = None
                        best.audio_state = "pending"
                        res.changed.append(best)
                        res.rearmed.append(best)
                res.kept.append(best)
            else:
                c = self._new_card(rid, text, now)
                used.add(c.id)
                res.added.append(c)

        # 没匹配上的：累计漏帧次数，够了就销毁
        for c in existing:
            if c.id in used:
                continue
            c.misses += 1
            if c.misses >= self.cfg.destroy_after_misses:
                # 只有**真的删掉了**才报 died。
                # 正在播放的卡片会被 _destroy 保护（只打 doomed 标记），
                # 那种情况不能对外说"它死了" —— 界面会把它从列表里删掉，
                # 但音频还在响，看起来就是"卡片没了声音还在"。
                if self._destroy(c):
                    res.died.append(c)

        return res

    def touch_region(self, rid: str, now: Optional[float] = None) -> int:
        """给某个区域的卡片"续命"，不改变内容。

        **为什么需要**：引擎里加了一道廉价帧签名比较 —— 画面没变化就跳过
        OCR（OCR 是唯一的 CPU 大头，单次几百毫秒）。但跳过 OCR 就没有
        `observe()`，卡片会一直累计 `misses`，过了 `destroy_after_misses`
        就被当成"文字消失了"删掉。

        所以"画面没变"时要调这里：把 `misses` 清零、`last_seen` 刷新，
        效果等同"这些文字还在屏幕上"。返回受影响的卡片数。
        """
        now = now if now is not None else time.time()
        n = 0
        for c in self._cards.values():
            if c.region_id != rid:
                continue
            c.misses = 0
            c.last_seen = now
            n += 1
        return n

    # ------------------------------------------------------------ 播放 --
    @property
    def playing_id(self) -> Optional[str]:
        return self._playing

    def mark_playing(self, cid: Optional[str]) -> Optional[Card]:
        """同一时刻只允许一张卡片在播（否则声音会叠在一起）。

        播完（cid=None）时顺手把"该删但因为在播而留着的"卡片收掉。
        """
        prev = self._playing
        if prev and prev != cid:
            p = self._cards.get(prev)
            if p:
                p.play_state = "idle"
        self._playing = cid
        c = self._cards.get(cid) if cid else None
        if c:
            c.play_state = "playing"
        # 上一张播完了 -> 如果它之前被标记为待销毁，现在真的删掉
        if cid is None and prev:
            old = self._cards.get(prev)
            if old is not None and getattr(old, "doomed", False):
                self._destroy(old)
        return c

    # ------------------------------------------------------------ 其他 --
    def clear(self) -> int:
        """清空全部卡片。**用户主动操作，立即生效**（不等播放结束）。"""
        n = len(self._cards)
        self._playing = None
        for c in list(self._cards.values()):
            self._destroy(c, force=True)
        return n

    def clear_region(self, rid: str) -> int:
        """清掉某个区域的卡片（换识别模式用）。**用户主动操作，立即生效**。"""
        doomed = [c for c in self._cards.values() if c.region_id == rid]
        for c in doomed:
            self._destroy(c, force=True)
        if self._playing and self._cards.get(self._playing) is None:
            self._playing = None
        return len(doomed)

    def stats(self) -> Dict[str, object]:
        cards = list(self._cards.values())
        return {
            "regions": len(self._regions),
            "cards": len(cards),
            "pending": sum(1 for c in cards if c.audio_state == "pending"),
            "ready": sum(1 for c in cards if c.ready),
            "failed": sum(1 for c in cards if c.audio_state == "failed"),
            "playing": self._playing,
        }
