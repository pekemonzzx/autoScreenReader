"""两个区域（一个单行 + 一个整段）同时开着时的行为。

用户反馈：单行和整段两个区域同时开，会"卡住"，得清空区域重开才恢复。

根因（已修）：
  1. `_detect_loop` / `_synth_loop` / `_autoplay_loop` 都没有兜住异常 ——
     任何一个区域的一次轮询抛异常，**线程就静默死亡**，识别/合成/朗读
     永久停摆，表现就是"卡住"。
  2. `clear_region_cards()` 里调了 `stop_playback()`，是**全局**停播 ——
     切整段区域的模式会把正在播的单行区域无声掐断。
  3. `_refresh_cards()` 让悬浮窗也只显示"当前选中区域"的卡片，
     另一个区域的卡片凭空消失。

这个测试覆盖 1 和 2（3 在 test_ui_smoke 里）。
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 测试隔离：别碰真实配置
os.environ["SCREENREADER_CONFIG"] = str(ROOT / "data" / "_tworegion_cfg.json")

from screenreader.board import BoardConfig, CardStore  # noqa: E402
from screenreader.config import AppConfig  # noqa: E402
from screenreader.worker import BoardEngine, EngineConfig  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


class FakeBackend:
    """合成后端：立刻返回一段假 WAV。"""

    def synthesize(self, text, **kw):
        return b"RIFF" + b"\x00" * 60

    def health(self, *a, **k):
        return True


class FakePlayer:
    """播放器：记录被停了几次，不真的出声。"""

    def __init__(self):
        self.stops = 0
        self.played = []

    def play_wav(self, data):
        self.played.append(data)

    def volume(self):
        return 100


def make_engine():
    cfg = AppConfig()
    eng = BoardEngine(
        cfg,
        backend=FakeBackend(),
        player=FakePlayer(),
        engine_cfg=EngineConfig(poll_interval_s=0.05, paragraph_settle_s=0.2),
    )
    return eng


def test_detect_loop_survives():
    """一个区域的轮询抛异常，不能让检测线程死掉，也不能拖累别的区域。"""
    print("\n[检测线程：单区域异常不能拖垮整体]")
    eng = make_engine()
    a = eng.store.add_region("单行区", (0, 0, 10, 10))
    b = eng.store.add_region("整段区", (20, 0, 10, 10))
    eng.store.update_region(b.id, mode="paragraph")

    calls = {"a": 0, "b": 0}

    def fake_poll(r):
        if r.id == a.id:
            calls["a"] += 1
            raise RuntimeError("模拟：单行区域这一帧 OCR 炸了")
        calls["b"] += 1

    eng._poll_region = fake_poll  # noqa: SLF001
    t = threading.Thread(target=eng._detect_loop, daemon=True)  # noqa: SLF001
    t.start()
    time.sleep(0.6)
    eng._stop.set()  # noqa: SLF001
    t.join(timeout=2)

    check("检测线程没有死", not t.is_alive(), "线程已退出" if not t.is_alive() else "仍在跑")
    check("出错区域被反复尝试（不是放弃）", calls["a"] >= 2, f"{calls['a']} 次")
    check("另一个区域一直在正常轮询", calls["b"] >= 2, f"{calls['b']} 次")
    check("错误记到了 _region_errors", eng._region_errors.get(a.id, 0) >= 1,  # noqa: SLF001
          str(eng._region_errors))  # noqa: SLF001


def test_clear_region_does_not_stop_other():
    """切 A 区域的模式，不能把 B 区域正在播的声音掐掉。"""
    print("\n[切模式只影响本区域]")
    eng = make_engine()
    a = eng.store.add_region("单行区", (0, 0, 10, 10))
    b = eng.store.add_region("整段区", (20, 0, 10, 10))

    # B 区域有一张卡在播
    out_b = eng.store.observe(b.id, ["整段区域的一句话。"])
    card_b = out_b.added[0]
    eng.store.mark_playing(card_b.id)
    check("B 正在播", eng.store.playing_id == card_b.id)

    # A 区域也有卡
    eng.store.observe(a.id, ["单行的一句话。"])

    eng.clear_region_cards(a.id)
    check("清 A 的卡之后 B 还在播", eng.store.playing_id == card_b.id,
          str(eng.store.playing_id))
    check("A 的卡被清空",
          not any(c.region_id == a.id for c in eng.store.sorted_cards()))
    check("B 的卡还在",
          any(c.region_id == b.id for c in eng.store.sorted_cards()))

    # 再清 B 自己，这时才该停
    eng.clear_region_cards(b.id)
    check("清自己区域时才停播", eng.store.playing_id is None)


def test_clear_region_clears_para_state():
    """换模式后稳定性状态要重置，否则会一直"还在变"不建卡。"""
    print("\n[换模式时清掉整段稳定性状态]")
    eng = make_engine()
    a = eng.store.add_region("整段区", (0, 0, 10, 10))
    eng.store.update_region(a.id, mode="paragraph")
    eng._para_state[a.id] = {"key": "abc", "since": 0.0, "committed": "abc"}  # noqa: SLF001

    eng.clear_region_cards(a.id)
    check("_para_state 已清空", a.id not in eng._para_state,  # noqa: SLF001
          str(list(eng._para_state)))  # noqa: SLF001


def test_synth_loop_survives():
    """合成失败不能让合成线程死掉。"""
    print("\n[合成线程：单张失败不能拖垮整体]")
    eng = make_engine()
    a = eng.store.add_region("t", (0, 0, 10, 10))
    out = eng.store.observe(a.id, ["一句要合成的话。"])
    cid = out.added[0].id

    calls = {"n": 0}

    def boom(c):
        calls["n"] += 1
        raise RuntimeError("模拟：合成炸了")

    eng._synth_one = boom  # noqa: SLF001
    t = threading.Thread(target=eng._synth_loop, daemon=True)  # noqa: SLF001
    t.start()

    eng._enqueue(eng.store.card(cid))  # noqa: SLF001
    time.sleep(0.5)
    # 再塞一张，线程还活着就应该继续处理
    out2 = eng.store.observe(a.id, ["第二句话。"])
    if out2.added:
        eng._enqueue(out2.added[0])  # noqa: SLF001
    time.sleep(0.5)

    eng._stop.set()  # noqa: SLF001
    t.join(timeout=2)
    check("合成线程没有死", not t.is_alive())
    check("失败的那张被标记 failed", eng.store.card(cid) is None or
          eng.store.card(cid).audio_state == "failed",
          str(getattr(eng.store.card(cid), "audio_state", "卡片已回收")))
    check("后续卡片仍被处理", calls["n"] >= 2, f"处理了 {calls['n']} 张")


def main() -> int:
    for fn in (test_detect_loop_survives, test_clear_region_does_not_stop_other,
               test_clear_region_clears_para_state, test_synth_loop_survives):
        try:
            fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            check(f"{fn.__name__} 未抛异常", False, f"{type(e).__name__}: {e}")

    try:
        Path(os.environ["SCREENREADER_CONFIG"]).unlink(missing_ok=True)
    except Exception:
        pass

    print("\n" + "=" * 60)
    print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    for f in FAIL:
        print(f"  FAIL: {f}")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
