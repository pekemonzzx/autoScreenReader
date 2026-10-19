"""board.py 的生命周期测试（纯逻辑，不需要界面/声卡/TTS）。

运行：
    .\\.venv\\Scripts\\python tests\\test_board.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from screenreader.board import (  # noqa: E402
    BoardConfig,
    CardStore,
    line_key,
    similarity,
)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def expect_eq(name, got, want):
    check(name, got == want, f"得到 {got!r}，期望 {want!r}")


# --------------------------------------------------------------------------- #
def test_similarity():
    print("\n[相似度]")
    check("完全相同", similarity("来了来了。", "来了来了。") == 1.0)
    check("标点差异视为相同", similarity("来了来了", "来了来了。") == 1.0)
    check("空格差异视为相同", similarity("deepseek HARNESS", "deepseekHARNESS") == 1.0)
    jit = similarity("deepseek HARNESS", "dleepseek HARNESS")
    check("OCR 抖动仍算相似", jit >= 0.85, f"相似度 {jit:.3f}")
    low = similarity("今天播好早", "明天再说吧")
    check("不同内容相似度低", low < 0.5, f"相似度 {low:.3f}")
    expect_eq("空串键", line_key(" 。、 "), "")


def test_region_crud():
    print("\n[区域增删]")
    s = CardStore()
    r1 = s.add_region("弹幕", (0, 0, 400, 200))
    r2 = s.add_region("聊天", (0, 300, 400, 200))
    expect_eq("两个区域", len(s.regions), 2)
    check("自动配色不同", r1.color != r2.color, f"{r1.color} / {r2.color}")

    s.observe(r1.id, ["第一句", "第二句"])
    s.observe(r2.id, ["另一个区域的话"])
    expect_eq("卡片数", len(s.cards), 3)

    n = s.remove_region(r1.id)
    expect_eq("删区域连带销毁卡片", n, 2)
    expect_eq("剩余卡片", len(s.cards), 1)
    expect_eq("剩余区域", len(s.regions), 1)


def test_lifecycle():
    print("\n[生灭生命周期]")
    s = CardStore(BoardConfig(destroy_after_misses=3))
    r = s.add_region("弹幕", (0, 0, 400, 200))

    out = s.observe(r.id, ["你好啊", "在吗"])
    expect_eq("新出现 2 张", len(out.added), 2)
    expect_eq("没有消失", len(out.died), 0)
    expect_eq("卡片数", len(s.cards), 2)

    # 内容不变 -> 都算 kept
    out = s.observe(r.id, ["你好啊", "在吗"])
    expect_eq("内容不变：keep", len(out.kept), 2)
    expect_eq("内容不变：不新增", len(out.added), 0)

    # "在吗" 消失一次
    out = s.observe(r.id, ["你好啊"])
    expect_eq("漏 1 次还不销毁", len(out.died), 0)
    expect_eq("卡片仍有 2 张", len(s.cards), 2)

    # 连续漏 3 次才销毁
    out = s.observe(r.id, ["你好啊"])
    out = s.observe(r.id, ["你好啊"])
    expect_eq("漏 3 次后销毁", len(out.died), 1)
    expect_eq("卡片剩 1 张", len(s.cards), 1)
    expect_eq("销毁的是那句", out.died[0].text, "在吗")

    # 中间重新出现则重新计数
    s.observe(r.id, ["你好啊", "在吗"])
    s.observe(r.id, ["你好啊"])
    out = s.observe(r.id, ["你好啊", "在吗"])   # 又回来了
    expect_eq("回来后重新计数", s.card(s.cards[0].id).misses if s.cards else -1, 0)
    out = s.observe(r.id, ["你好啊"])
    expect_eq("计数已重置，不会立刻销毁", len(out.died), 0)


def test_ocr_jitter():
    print("\n[OCR 抖动]")
    s = CardStore()
    r = s.add_region("弹幕", (0, 0, 400, 200))
    s.observe(r.id, ["deepseek HARNESS"])
    out = s.observe(r.id, ["dleepseek HARNESS"])
    expect_eq("抖动不新建卡片", len(out.added), 0)
    expect_eq("抖动不销毁", len(out.died), 0)
    expect_eq("卡片仍是 1 张", len(s.cards), 1)
    expect_eq("文本更新为新的", s.cards[0].text, "dleepseek HARNESS")
    expect_eq("高相似度不触发重新合成", len(out.rearmed), 0)

    # 变化"中等"（相似度介于 match_threshold 与 resynth_threshold 之间）
    # -> 视为同一句、更新文字，但必须重新合成
    s2 = CardStore()
    r2 = s2.add_region("文本", (0, 0, 200, 100))
    s2.observe(r2.id, ["欢迎来到直播间"])
    sim = similarity("欢迎来到直播间", "欢迎来到直播问")
    check("构造出的相似度在阈值区间内", 0.85 <= sim < 0.92, f"相似度 {sim:.3f}")
    out = s2.observe(r2.id, ["欢迎来到直播问"])
    expect_eq("中等变化不新建卡片", len(out.added), 0)
    expect_eq("中等变化触发重新合成", len(out.rearmed), 1)
    expect_eq("旧语音已清空", s2.cards[0].audio is None, True)
    expect_eq("状态回到 pending", s2.cards[0].audio_state, "pending")

    # 变化"很大"（相似度低于 match_threshold）-> 视为新卡片，旧卡随后自然消亡
    s3 = CardStore(BoardConfig(destroy_after_misses=1))
    r3 = s3.add_region("数值", (0, 0, 200, 100))
    s3.observe(r3.id, ["金币 12345"])
    out = s3.observe(r3.id, ["金币 99999"])
    expect_eq("数值大变 -> 新建卡片", len(out.added), 1)
    expect_eq("旧卡片同轮销毁", len(out.died), 1)


def test_multiregion_match_isolation():
    print("\n[区域隔离]")
    s = CardStore()
    a = s.add_region("A", (0, 0, 200, 100))
    b = s.add_region("B", (0, 200, 200, 100))
    s.observe(a.id, ["相同的句子"])
    out = s.observe(b.id, ["相同的句子"])
    expect_eq("同内容在不同区域不互相匹配", len(out.added), 1)
    expect_eq("总共 2 张卡片", len(s.cards), 2)
    check("卡片分属不同区域", s.cards[0].region_id != s.cards[1].region_id)


def test_max_cards():
    print("\n[卡片上限]")
    s = CardStore(BoardConfig(max_cards=5))
    r = s.add_region("x", (0, 0, 100, 100))
    for i in range(10):
        s.observe(r.id, [f"句子{i}"])
    check("不超过上限", len(s.cards) <= 5, f"实际 {len(s.cards)}")


def test_playback_single():
    print("\n[播放单实例]")
    s = CardStore()
    r = s.add_region("x", (0, 0, 100, 100))
    s.observe(r.id, ["第一句", "第二句"])
    c1, c2 = s.cards
    s.mark_playing(c1.id)
    expect_eq("第一句在播", c1.play_state, "playing")
    s.mark_playing(c2.id)
    expect_eq("切到第二句后第一句复位", c1.play_state, "idle")
    expect_eq("第二句在播", c2.play_state, "playing")
    expect_eq("playing_id", s.playing_id, c2.id)

    # 正在播放的卡片**不会**被自动销毁（否则音频会被掐断），
    # 所以漏帧再多次也只销毁没在播的那一张
    s.observe(r.id, [""])
    s.observe(r.id, [""])
    out = s.observe(r.id, [""])
    expect_eq("只销毁没在播的那张", len(out.died), 1)
    expect_eq("正在播的还留着", s.card(c2.id) is not None, True)
    expect_eq("playing 保持（没被误判成播完）", s.playing_id, c2.id)
    expect_eq("被保护的卡打了 doomed 标记", getattr(c2, "doomed", False), True)

    # 播完之后才真的收掉
    s.mark_playing(None)
    expect_eq("播完后卡片被收掉", s.card(c2.id) is None, True)
    expect_eq("playing 已清空", s.playing_id, None)


def test_ordering_and_stats():
    print("\n[排序与统计]")
    s = CardStore()
    a = s.add_region("A", (0, 0, 100, 100))
    b = s.add_region("B", (0, 100, 100, 100))
    s.observe(b.id, ["B区的话"])
    s.observe(a.id, ["A区第一句", "A区第二句"])
    order = [c.region_id for c in s.sorted_cards()]
    expect_eq("按区域顺序分组", order, [a.id, a.id, b.id])

    st = s.stats()
    expect_eq("统计-区域", st["regions"], 2)
    expect_eq("统计-卡片", st["cards"], 3)
    expect_eq("统计-待合成", st["pending"], 3)

    s.cards[0].audio = b"x"
    s.cards[0].audio_state = "ready"
    expect_eq("统计-就绪", s.stats()["ready"], 1)

    expect_eq("清空", s.clear(), 3)
    expect_eq("清空后为 0", len(s.cards), 0)


# --------------------------------------------------------------------------- #
def main() -> int:
    for fn in (
        test_similarity,
        test_region_crud,
        test_lifecycle,
        test_ocr_jitter,
        test_multiregion_match_isolation,
        test_max_cards,
        test_playback_single,
        test_ordering_and_stats,
    ):
        fn()
    print("\n" + "=" * 60)
    print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    for f in FAIL:
        print(f"  FAIL: {f}")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
