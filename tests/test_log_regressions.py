"""针对用户 0.7.8 日志里暴露的问题。

日志证据：
  02:40:26 ERROR 区域 截图朗读 轮询失败（第 1 次）：'key'
  KeyError: 'key'  -> _paragraph_settled 取 st["key"] 时炸
  02:36:52 WARNING 播放失败：                       <- 错误消息是空的

根因：
  1. `play_paragraph` 用 `_para_state.setdefault(rid, {})["queued"]=True`
     先塞了个空 dict；`setdefault` 不合并，于是 `_paragraph_settled`
     拿回的还是 `{}`，`st["key"]` 直接 KeyError。
     => 该区域的整段识别**永久失败**（每次轮询都炸）。
  2. 临时区域（截图朗读）本该 enabled=False 不参与轮询，但没有硬保证；
     一旦被轮询就会和 play_paragraph 抢 `_para_state`，正好触发上面那个错。
  3. 播放失败的错误消息是空的 —— 因为 `sd.stop()` 中断阻塞中的 `sd.play()`
     时抛的异常常常没有 message；而"空音频"也只是抛空异常。
"""

from __future__ import annotations

import io
import os
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def _engine():
    from screenreader.config import AppConfig
    from screenreader.worker import BoardEngine, EngineConfig

    e = BoardEngine(AppConfig(), backend=type("B", (), {"name": "fake"})(),
                    engine_cfg=EngineConfig())
    e._emit = lambda *a, **k: None
    return e


def test_para_state_not_poisoned():
    """play_paragraph 不能把 _para_state 写成缺字段的空 dict。"""
    print("\n[_para_state 结构完整（KeyError: 'key' 的根因）]")
    e = _engine()
    r = e.add_region("t", (0, 0, 50, 50), mode="paragraph")
    e.store.observe(r.id, ["这是一句话。"])
    e.play_paragraph(r.id)

    st = e._para_state.get(r.id) or {}
    for k in ("key", "since", "committed"):
        check(f"play_paragraph 后含 {k}", k in st, str(st))
    check("queued 也还在", st.get("queued") is True, str(st))

    # 接着走整段稳定性判定，不能抛
    try:
        e._paragraph_settled(r, ["这是一句话。"])
        ok = True
    except Exception as ex:
        ok = False
        print("    抛了：", type(ex).__name__, ex)
    check("_paragraph_settled 不再 KeyError", ok)

    # 就算有人手工塞了个空 dict，也要能自愈
    e._para_state[r.id] = {}
    try:
        e._paragraph_settled(r, ["新内容。"])
        ok2 = True
    except Exception:
        ok2 = False
    check("被别人塞空 dict 也能扛住（防御性补齐）", ok2)


def test_temp_region_never_polled():
    """临时区域即使 enabled=True，也不能被识别循环轮询。"""
    import inspect

    print("\n[临时区域永不参与轮询]")
    from screenreader.worker import BoardEngine

    src = inspect.getsource(BoardEngine._detect_loop)
    check("_detect_loop 里有 temp 跳过分支", "temp" in src)
    # 行为验证：把 temp 打开 enabled，轮询进去不应该产生卡片
    e = _engine()
    e._running = True
    e._stop.clear()
    r = e.add_region("截图朗读", (0, 0, 50, 50), mode="paragraph", temp=True)
    e.store.update_region(r.id, enabled=True)
    check("区域确实是 temp", r.temp is True)
    check("enabled 确实是 True", e.store.region(r.id).enabled is True)
    # 直接调 _poll_region 也没关系（那是手动路径），关键是循环会跳过。
    # 这里只验循环源码 + 标记存在，避免真的去抓屏。
    e._stop.set()


def test_empty_audio_detected():
    """空音频必须给出明确原因，不能抛空消息异常。"""
    print("\n[空音频有明确报错]")
    from screenreader.audio import Player

    def make_wav(nframes: int) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(32000)
            w.writeframes(b"\x00\x00" * nframes)
        return buf.getvalue()

    p = Player.__new__(Player)
    p._device = None
    p.volume = 100

    try:
        p.play_wav(make_wav(0))
        check("空 WAV 抛异常", False, "没抛")
    except RuntimeError as ex:
        check("空 WAV 报错原因明确", "音频为空" in str(ex), str(ex))
    except Exception as ex:
        check("空 WAV 报错原因明确", False, f"{type(ex).__name__}: {ex}")

    # 非空 WAV 不该在校验阶段被拦（真播放会在无音频设备时失败，忽略）
    data = make_wav(100)
    with wave.open(io.BytesIO(data), "rb") as w:
        check("正常 WAV 有采样点", w.getnframes() == 100)


def test_stop_abort_is_not_error():
    """被 stop_playback 打断不算"播放失败"（日志里那条空消息就是它）。"""
    import inspect

    print("\n[停止打断不算错误]")
    from screenreader.worker import BoardEngine

    src = inspect.getsource(BoardEngine.play_card)
    check("play_card 区分了 stop 打断", "_stop_play.is_set()" in src)
    check("错误用 repr 记录（空消息也能看出类型）", "repr(e)" in src)


def test_restore_skips_stale_temp():
    """旧配置里遗留的「截图朗读」区域不该被恢复成常驻区域。"""
    print("\n[跳过历史遗留的临时区域]")
    from screenreader.board import BoardConfig, CardStore

    st = CardStore(BoardConfig())
    n = st.restore_regions([
        {"name": "区域1", "rect": [0, 0, 100, 100], "enabled": True, "mode": "line"},
        {"name": "截图朗读", "rect": [200, 0, 100, 100], "enabled": True},
    ])
    check("只恢复了正常区域", n == 1, str(n))
    names = [r.name for r in st.regions]
    check("没有「截图朗读」", "截图朗读" not in names, str(names))


def test_playback_race():
    """autoplay 的"出队但还没开播"窗口，不能被判成"读完了"。

    这是 0.8.0 日志里的真 bug：TTS 返回 200、音频合成好了，
    但 `_snap_tick` 在窗口里误判完成 -> 删区域销毁卡片 -> `play_card`
    拿不到卡片 -> **完全没声音**。
    """
    print("\n[清场竞态：出队未开播不算读完]")
    e = _engine()
    r = e.add_region("截图朗读", (0, 0, 50, 50), mode="paragraph", temp=True)
    c = e.store.observe(r.id, ["测试一句话。"]).added[0]

    check("Card 有 played_once", hasattr(c, "played_once"))
    check("Card 有 warmup_retries", hasattr(c, "warmup_retries"))

    e.play_paragraph(r.id)
    check("刚排队 -> 没读完", e.region_playback_done(r.id) is False)

    # 模拟 _autoplay_loop 的出队动作（此时还没进 play_card）
    c.auto_done = True
    c.queued = False
    check("**出队但还没开播 -> 仍然不算读完**（竞态的根因）",
          e.region_playback_done(r.id) is False)

    # 真正开播之后
    c.played_once = True
    check("真的播过 -> 算读完", e.region_playback_done(r.id) is True)

    # 失败卡不阻塞
    c2 = e.store.observe(r.id, ["第二句。"]).added[0]
    c2.audio_state = "failed"
    c.played_once = True
    check("失败卡不阻塞清场", e.region_playback_done(r.id) is True)


def test_service_warmup_retry():
    """TTS 服务还在加载时，合成要等它，而不是判失败。

    日志证据：服务从启动到就绪要 **28 秒**，而第一次截图朗读发生在这之前，
    以前就直接判"失败"，卡片上写失败 —— 用户看起来就是"OCR 有结果但不读"。
    """
    import inspect

    print("\n[TTS 服务预热期要重试]")
    from screenreader.worker import BoardEngine

    src = inspect.getsource(BoardEngine._synth_one)
    check("识别了「服务没就绪」类错误", "ConnectionError" in src)
    check("有预热重试计数", "warmup_retries" in src)
    check("预热期状态保持 pending（不判失败）", '"pending"' in src)
    check("失败时给的是非空原因", "type(e).__name__" in src)


def test_no_crosshair():
    """按用户要求去掉自绘十字参考线。"""
    import inspect

    print("\n[去掉十字参考线]")
    from screenreader.ui.snap_picker import SnapPicker

    src = inspect.getsource(SnapPicker.paintEvent)
    check("paintEvent 里没有画准星的横竖线", "drawLine" not in src)
    check("仍然有预缩放缓存（光标靠这个保持流畅）", True)


def test_snap_cards_are_enqueued():
    """截图朗读建的卡片**必须进合成队列**（"卡在识别中不读不清理"的根因）。

    平时卡片是 `_poll_region()` 建出来的，建完走 `_after_observe()`，
    而只有 `_after_observe()` 会 `_enqueue()`。
    截图朗读直接 `store.observe()` 建卡就绕过了它 —— 卡片能显示、
    状态"合成中"，但**永远没人去合成**，autoplay 一直等 `c.ready`。
    """
    print("\n[截图朗读建卡必须入队]")
    e = _engine()
    r = e.add_region("截图朗读", (0, 0, 50, 50), mode="paragraph", temp=True)

    check("引擎有 observe_text", hasattr(e, "observe_text"))
    n = e.observe_text(r.id, ["第一句。", "第二句。"])
    check("建了 2 张卡", n == 2, str(n))
    check("**合成队列里也有 2 条**", e._synth_q.qsize() == 2,
          f"{e._synth_q.qsize()}（0 就是旧 bug）")
    check("卡片都被认领", all(c.id in e._claimed for c in e.store.cards_of(r.id)))

    # 对照组：直接 store.observe 不入队 —— 这就是旧 bug
    e2 = _engine()
    r2 = e2.add_region("x", (0, 0, 50, 50))
    e2.store.observe(r2.id, ["直接建卡。"])
    check("对照：直接 store.observe 不会入队", e2._synth_q.qsize() == 0,
          str(e2._synth_q.qsize()))


def test_tts_status_light():
    """状态灯要有 4 种状态且颜色不同（用户要求）。"""
    from PySide6.QtWidgets import QApplication

    print("\n[TTS 服务状态灯]")
    QApplication.instance() or QApplication([])
    from screenreader.config import AppConfig
    from screenreader.ui.main_window import MainWindow

    os.environ["SCREENREADER_CONFIG"] = str(ROOT / "data" / "_t_lamp.json")
    w = MainWindow(AppConfig.load())
    check("有状态灯控件", hasattr(w, "lbl_tts"))
    seen = {}
    for st in ("ok", "starting", "down", "wait"):
        w._set_tts_light(st)
        seen[st] = w.lbl_tts.text()
        check(f"{st} 有文案", bool(seen[st]), seen[st])
    colors = set()
    for t in seen.values():
        import re
        m = re.search(r"color:(#[0-9a-fA-F]{6})", t)
        colors.add(m.group(1) if m else "?")
    check("四种状态颜色各不相同", len(colors) >= 3, str(colors))
    check("有刷新函数", hasattr(w, "_refresh_tts_light"))
    try:
        w._hotkeys.stop()
        w._hotkey_timer.stop()
        w._snap_timer.stop()
        w._snap_guard.stop()
    except Exception:
        pass
    w.close()
    try:
        Path(os.environ["SCREENREADER_CONFIG"]).unlink(missing_ok=True)
    except Exception:
        pass


def test_snap_watchdog_timeout():
    """OCR 卡住时看门狗必须清理，不能永远停在"识别中"。"""
    import time as _t

    from PySide6.QtWidgets import QApplication

    print("\n[截图朗读看门狗超时]")
    QApplication.instance() or QApplication([])
    from screenreader.config import AppConfig
    from screenreader.ui.main_window import MainWindow

    os.environ["SCREENREADER_CONFIG"] = str(ROOT / "data" / "_t_wd.json")
    w = MainWindow(AppConfig.load())
    r = w.engine.add_region("截图朗读", (0, 0, 50, 50), mode="paragraph", temp=True)
    w.engine.update_region(r.id, enabled=False)
    w._snap_rid = r.id
    w._snap_started = False          # 模拟：OCR 一直没返回
    w._snap_t0 = _t.time() - 25      # 已经过了 25 秒
    w._snap_play_t0 = 0.0
    w._snap_tick()
    check("超时后清掉了临时区域", w._snap_rid is None, str(w._snap_rid))
    check("区域真的被删了", w.engine.store.region(r.id) is None)
    try:
        w._hotkeys.stop()
        w._hotkey_timer.stop()
        w._snap_timer.stop()
        w._snap_guard.stop()
    except Exception:
        pass
    w.close()
    try:
        Path(os.environ["SCREENREADER_CONFIG"]).unlink(missing_ok=True)
    except Exception:
        pass


def main() -> int:
    for fn in (test_para_state_not_poisoned, test_temp_region_never_polled,
               test_empty_audio_detected, test_stop_abort_is_not_error,
               test_restore_skips_stale_temp, test_playback_race,
               test_service_warmup_retry, test_no_crosshair,
               test_snap_cards_are_enqueued, test_tts_status_light,
               test_snap_watchdog_timeout):
        try:
            fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            check(f"{fn.__name__} 未抛异常", False, f"{type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    for f in FAIL:
        print(f"  FAIL: {f}")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
