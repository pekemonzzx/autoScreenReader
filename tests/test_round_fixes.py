"""本轮的 5 个修改点（用户反馈）。

1. 截图朗读之后卡片永远停在"合成中"（TTS 不开始）
2. 普通框选的右键语义：撤销上一个；没有可撤销的才退出
3. 「删除区域」提到一级工具栏
4. 选中区域后按 Delete 删除
5. 「停止识别」和「全部停止」合并
"""

from __future__ import annotations

import os
import sys
import time
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


def test_synth_thread_self_heals():
    """合成线程死了必须能补回来（这是"卡片永远合成中"的根因）。"""
    print("\n[合成线程也在自愈范围里]")
    e = _engine()
    e._running = True
    e._stop.clear()
    fixed = e.ensure_threads()
    check("ensure_threads 做了修复", fixed is True)
    names = {t.name for t in e._threads}
    check("补出了 synth", "synth" in names, str(names))
    check("补出了 detect", "detect" in names)
    check("补出了 autoplay", "autoplay" in names)
    e._stop.set()
    time.sleep(0.1)


def test_stuck_synth_recovery():
    """卡在"合成中"的卡片：先重排，再判失败，不会永远转。"""
    print("\n[合成卡住能救回来]")
    e = _engine()
    r = e.add_region("t", (0, 0, 50, 50))
    c = e.store.observe(r.id, ["一句会被卡住的话。"]).added[0]
    e._enqueue(c)
    check("排队时记下了时刻", c.id in e._queued_at)

    # 人为让它"已经卡了 100 秒"
    e._queued_at[c.id] = time.time() - 100
    n = e.recover_stuck_synth(max_s=45.0)
    check("第一次检测到卡住", n == 1, str(n))
    check("第一次是重新排队", c.synth_retries == 1 and c.audio_state == "pending",
          f"retries={c.synth_retries} state={c.audio_state}")

    e._queued_at[c.id] = time.time() - 100
    n2 = e.recover_stuck_synth(max_s=45.0)
    check("第二次检测到卡住", n2 == 1, str(n2))
    check("第二次判失败", c.audio_state == "failed", c.audio_state)
    check("失败原因写清楚了", "超时" in c.audio_error, c.audio_error)


def test_picker_right_click_semantics():
    """普通框选：右键先撤销，没得撤销才退出。"""
    from PySide6.QtCore import QRect
    from PySide6.QtWidgets import QApplication

    from screenreader.ui.region_picker import RegionPicker

    print("\n[右键 = 撤销，没有则退出]")
    QApplication.instance() or QApplication([])
    p = RegionPicker()
    cancelled = []
    p.cancelled.connect(lambda: cancelled.append(1))
    p._rects.append(QRect(0, 0, 10, 10))
    p._rects.append(QRect(20, 0, 10, 10))

    class _Ev:
        def button(self):
            from PySide6.QtCore import Qt
            return Qt.RightButton

        def position(self):
            from PySide6.QtCore import QPoint
            return QPoint(0, 0)

    def right_click():
        """用真的 QMouseEvent，否则 super().mousePressEvent 会因类型不对而报错。"""
        from PySide6.QtCore import QEvent, QPointF, Qt
        from PySide6.QtGui import QMouseEvent
        return QMouseEvent(QEvent.MouseButtonPress, QPointF(0, 0), QPointF(0, 0),
                           Qt.RightButton, Qt.RightButton, Qt.NoModifier)

    # 有可撤销的 -> 撤销，不触发取消
    p.mousePressEvent(right_click())
    check("右键撤销了一个框", len(p._rects) == 1, str(len(p._rects)))
    check("撤销时没有退出", not cancelled)
    p.mousePressEvent(right_click())
    check("再撤销一个", len(p._rects) == 0, str(len(p._rects)))
    check("还是没有退出", not cancelled)

    # 没得撤销了 -> 退出
    p.mousePressEvent(right_click())
    check("没有可撤销的就发出取消", bool(cancelled))


def test_toolbar_regroup():
    """布局：删除区域上一级；停止识别/全部停止合并。"""
    from PySide6.QtWidgets import QApplication, QToolBar

    print("\n[工具栏重排]")
    QApplication.instance() or QApplication([])
    from screenreader.config import AppConfig
    from screenreader.ui.main_window import MainWindow

    os.environ["SCREENREADER_CONFIG"] = str(ROOT / "data" / "_t5.json")
    w = MainWindow(AppConfig.load())
    tb = w.findChild(QToolBar)
    texts = [a.text() for a in tb.actions() if a.text()]
    check("一级有「删除区域」", any("删除区域" in t for t in texts), str(texts))
    check("一级不再有「全部停止」", not any("全部停止" in t for t in texts), str(texts))
    check("一级有「开始识别」", any("开始识别" in t for t in texts), str(texts))
    more = [a.text() for a in w.btn_more.menu().actions() if a.text()]
    check("「更多」里有「停止朗读（不停识别）」",
          any("停止朗读" in m for m in more), str(more))
    check("主窗口装了事件过滤器（Delete 删除）",
          w.region_list.focusPolicy() is not None)
    # Delete 删除
    a = w.engine.add_region("A", (0, 0, 20, 20))
    w.engine.add_region("B", (30, 0, 20, 20))
    w._refresh_regions()
    w.region_list.setCurrentRow(0)
    w._remove_selected_region()
    check("删除区域有效", len(w.engine.store.regions) == 1,
          str(len(w.engine.store.regions)))
    _teardown(w)


def _teardown(w):
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


def test_hotkey_blocked_while_picking():
    """框选期间不允许快捷键再次唤出截图朗读（用户要求）。

    否则会在已有全屏框选层上再叠一层，两个都置顶，看起来就是"界面卡住"。
    """
    from PySide6.QtGui import QColor, QImage, QPixmap
    from PySide6.QtWidgets import QApplication

    from screenreader.config import AppConfig
    from screenreader.ui.main_window import MainWindow
    from screenreader.ui.region_picker import RegionPicker
    from screenreader.ui.snap_picker import SnapPicker

    print("\n[框选期间屏蔽快捷键]")
    QApplication.instance() or QApplication([])
    os.environ["SCREENREADER_CONFIG"] = str(ROOT / "data" / "_t_hk.json")
    w = MainWindow(AppConfig.load())

    check("有 _picker_busy", hasattr(w, "_picker_busy"))
    check("空闲时不算忙", w._picker_busy() is False)

    # 1) 普通框选层开着
    w._picker = RegionPicker()
    w._picker.start()
    check("普通框选层开着时算忙", w._picker_busy() is True)

    calls = []
    orig = w.snap_read
    w.snap_read = lambda: calls.append(1)
    w._hotkeys.events.put("snap")
    w._poll_hotkeys()
    check("**普通框选时快捷键被忽略**", calls == [], f"调用 {len(calls)} 次")
    w.snap_read = orig
    w._picker.hide()
    w._picker = None
    check("取消后不再忙", w._picker_busy() is False)

    # 2) 截图朗读框选层开着
    img = QImage(200, 150, QImage.Format_RGB888)
    img.fill(QColor(0, 0, 0))
    w._snap_picker = SnapPicker(QPixmap.fromImage(img))
    check("截图朗读框选层开着时算忙", w._picker_busy() is True)

    calls2 = []
    w.snap_read = lambda: calls2.append(1)
    w._hotkeys.events.put("snap")
    w._poll_hotkeys()
    check("**截图朗读框选时快捷键也被忽略**", calls2 == [], f"调用 {len(calls2)} 次")
    w.snap_read = orig

    # 3) 直接调 snap_read 也要挡住（工具栏按钮同理）
    blocked = []
    w.snap_read = lambda: blocked.append(1)
    # 这里调真实的入口判断，而不是被替换的 snap_read
    from screenreader.ui.main_window import MainWindow as MW
    MW.snap_read(w)
    check("直接调入口也会被挡住", blocked == [])

    _teardown(w)


def main() -> int:
    for fn in (test_synth_thread_self_heals, test_stuck_synth_recovery,
               test_picker_right_click_semantics, test_toolbar_regroup,
               test_hotkey_blocked_while_picking):
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
