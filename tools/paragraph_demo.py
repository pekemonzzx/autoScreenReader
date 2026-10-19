"""整段模式端到端演示：滚动 -> 清卡 -> 稳定 -> 手动播放。

对应用户提的 4 条要求：
  1. 滚动（OCR 变化）时**立刻清掉**旧卡片
  2. 先按**图上位置**从上到下排序，再考虑合并
  3. 断句：先句号，再逗号，**不切在句子中间**
  4. 手动点播放，从上到下顺序读；播放中新增的句子自动进队列

不碰真麦克风/声卡，用假的合成与播放，随时可跑。
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from screenreader.board import BoardConfig, CardStore  # noqa: E402
from screenreader.segment import merge_lines_to_sentences  # noqa: E402
from screenreader.worker import BoardEngine, EngineConfig  # noqa: E402

STABLE = [
    ("他那个写了很多规则，", 10.0, 0.0),
    ("主要是为了让玩家不要乱想。", 30.0, 0.0),
    ("还有啊，我觉得这个本子", 50.0, 0.0),
    ("写得挺有意思的，", 70.0, 0.0),
    ("值得再玩一次。", 90.0, 0.0),
]

SCROLL = [
    [("案发现场有三个人。", 10.0, 0.0), ("第一个是……", 30.0, 0.0)],
    [("第一个是管家，他当", 5.0, 0.0), ("时正在厨房。", 25.0, 0.0)],
    [("时正在厨房。第二", 8.0, 0.0), ("个人是司机。", 28.0, 0.0)],
]

OK, BAD = [], []


def check(name, cond, detail=""):
    (OK if cond else BAD).append(name)
    print(f"    [{'OK' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


class Harness:
    """借用 BoardEngine 的稳定性判定，不启动线程。"""

    def __init__(self, settle: float = 0.4):
        self.ecfg = EngineConfig(paragraph_settle_s=settle)
        self._para_state: dict = {}
        self._playing_regions: set = set()
        self.store = CardStore(BoardConfig())
        self.events: list = []
        self._settled = BoardEngine._paragraph_settled

    def _emit(self, kind, **kw):
        self.events.append(kind)

    def feed(self, region, lines_with_pos):
        texts = [x[0] for x in lines_with_pos]
        spans = [(x[1], x[2]) for x in lines_with_pos]
        sentences = merge_lines_to_sentences(texts, spans=spans)
        ok = self._settled(self, region, sentences)
        return ok, sentences


def main() -> int:
    print("=" * 78)
    print("整段模式端到端演示（按用户 4 条要求）")
    print("=" * 78)

    h = Harness(settle=0.4)
    r = h.store.add_region("正文区", (0, 0, 800, 400))
    h.store.update_region(r.id, mode="paragraph")

    # ---------- 1. 滚动：不建卡 + 立刻清卡 ----------
    print("\n[1] 滚动中：不建卡；内容一变立刻清旧卡")
    built = 0
    for i, snap in enumerate(SCROLL, 1):
        ok, sents = h.feed(r, snap)
        if ok:
            out = h.store.observe(r.id, sents)
            built += len(out.added)
        time.sleep(0.1)
    check("滚动期间不建卡", built == 0, f"建了 {built} 张")

    # 先弄出一批卡片，再模拟"页面滑走"。
    # 注意要喂两次同样的内容：第一次是"内容刚变"，第二次才是"稳定够了"。
    h.feed(r, STABLE)
    time.sleep(0.5)
    ok, sents = h.feed(r, STABLE)
    check("稳定后可以建卡", ok)
    out = h.store.observe(r.id, sents)
    before = len([c for c in h.store.sorted_cards() if c.region_id == r.id])
    check("建出了卡片", before > 0, f"{before} 张")

    ok, _ = h.feed(r, [("滑走之后的完全不同的内容。", 10.0, 0.0)])
    after = len([c for c in h.store.sorted_cards() if c.region_id == r.id])
    check("内容一变旧卡立刻清光", after == 0, f"还剩 {after} 张")
    check("发了 cleared 事件", "paragraph_cleared" in h.events)

    # ---------- 2. 按位置排序 ----------
    print("\n[2] 先按图上位置排序，再合并")
    shuffled = [
        ("第二段到此结束。", 30.0, 0.0),
        ("第一段还没说完，", 10.0, 0.0),
        ("接着说。", 20.0, 0.0),
    ]
    sents = merge_lines_to_sentences([x[0] for x in shuffled],
                                     spans=[(x[1], x[2]) for x in shuffled])
    check("乱序输入按位置排好",
          sents == ["第一段还没说完，接着说。", "第二段到此结束。"], str(sents))

    # ---------- 3. 断句优先级 ----------
    print("\n[3] 断句：先句号，再逗号，不切在句子中间")
    long_comma = "这是一句很长的话，中间有逗号，所以可以在这里切，但是整句还是有点长啊。"
    got = merge_lines_to_sentences([long_comma], max_chars=20)
    check("超长时按逗号切", len(got) >= 2, str(got))
    no_comma = "这句话有三十多个字但是一个逗号都没有所以就绝对不能硬切因为它是一个完整的句子啊"
    got = merge_lines_to_sentences([no_comma], max_chars=20)
    check("没逗号的长句不硬切", len(got) == 1, f"{len(got)} 段")

    # ---------- 4. 手动播放 + 播放中追加 ----------
    print("\n[4] 手动点播放，从上到下顺序读；播放中新增自动进队列")
    h.store.clear_region(r.id)
    h._para_state.clear()
    ok, sents = h.feed(r, STABLE)
    time.sleep(0.5)
    ok, sents = h.feed(r, STABLE)
    out = h.store.observe(r.id, sents)
    print(f"    合并后 {len(sents)} 句：")
    for s in sents:
        print(f"      · {s}")

    check("没点播放时队列为空", h.store.next_auto_card() is None)

    # 模拟合成：最快的一句先就绪
    for c in out.added:
        def mk(card):
            def run():
                time.sleep(0.15)
                card.audio = card.text.encode("utf-8")
                card.audio_state = "ready"
            return run
        threading.Thread(target=mk(c), daemon=True).start()

    time.sleep(0.35)
    h._playing_regions.add(r.id)
    n = h.store.queue_region(r.id)
    check("点播放后开始排队", n == len(sents), str(n))

    played = []
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            c = h.store.next_auto_card()
            if c is None:
                stop.wait(0.03)
                continue
            if not c.ready:
                stop.wait(0.02)
                continue
            c.auto_done = True
            c.queued = False
            h.store.mark_playing(c.id)
            played.append(c.text)
            time.sleep(len(c.text) * 0.005)
            h.store.mark_playing(None)
            stop.wait(0.03)

    th = threading.Thread(target=loop, daemon=True)
    th.start()

    # 播放途中又识别出新的一句 —— 应自动接进队列
    time.sleep(0.1)
    more = h.store.observe(r.id, sents + ["后来又补了一句。"])
    for c in more.added:
        if r.id in h._playing_regions:
            c.queued = True

        # 新句子也要"合成"一下，否则播放循环会一直等它 ready。
        # 注意是 mk_new() —— 要调用它拿到内部的 run，不能直接传 mk_new
        def mk_new(card=c):
            def run():
                time.sleep(0.15)
                card.audio = card.text.encode("utf-8")
                card.audio_state = "ready"
            return run

        threading.Thread(target=mk_new(), daemon=True).start()
    check("播放中新增的句子进了队列",
          any(c.text == "后来又补了一句。" and c.queued for c in more.added))

    expect_count = len(sents) + 1
    deadline = time.time() + 15
    while time.time() < deadline:
        # 必须等**期望的句数全播完**，不能一看到空队列就收工 ——
        # 中途队列恰好空一下（正在换句）会被误判成播完了
        if len(played) >= expect_count and h.store.next_auto_card() is None \
                and h.store.playing_id is None:
            break
        time.sleep(0.03)
    stop.set()
    th.join(timeout=2)

    print(f"\n    实际朗读顺序：")
    for i, p in enumerate(played, 1):
        print(f"      {i}. {p}")
    expect = sents + ["后来又补了一句。"]
    check("顺序与页面从上到下一致", played == expect,
          "" if played == expect else f"期望 {expect}")

    print("\n" + "=" * 78)
    print(f"通过 {len(OK)} 项，失败 {len(BAD)} 项")
    for b in BAD:
        print(f"  FAIL: {b}")
    print("=" * 78)
    return 1 if BAD else 0


if __name__ == "__main__":
    raise SystemExit(main())
