"""验收「截图朗读」的安全性与自愈能力（这轮用户报的三个严重问题）。

用户原话：
  2. 截图朗读成功之后显示在识别一直卡住，需要手动重新识别才会识别并朗读
  3. 截图朗读时取消可能会失败，能听到报错音，无法退出截图界面，
     甚至无法关闭程序，只能重启电脑（过于严重）
  1. 个别快捷键无法设置，比如鼠标前进侧键

对应要验的东西：
  * 框选层的取消**永远不会抛异常**（抛了就会把用户困在全屏窗口里）
  * 有全局 Esc 兜底 + 硬超时保险丝
  * `_end_snap` 即使中途出错也一定恢复其他区域的开关
  * OCR 锁**带超时**，检测线程不会被后台任务永久卡死
  * 检测线程能自愈
  * 快捷键捕获走全局钩子，不再依赖 Qt 的鼠标事件
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from screenreader.hotkey import HotkeyListener  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def test_cancel_never_raises():
    """取消路径必须绝对安全：槽函数抛异常也不能把窗口留在屏幕上。"""
    from PySide6.QtGui import QColor, QImage, QPixmap
    from PySide6.QtWidgets import QApplication

    from screenreader.ui.snap_picker import SnapPicker

    print("\n[取消永远不会抛异常]")
    QApplication.instance() or QApplication([])

    img = QImage(60, 40, QImage.Format_RGB888)
    img.fill(QColor(0, 0, 0))
    p = SnapPicker(QPixmap.fromImage(img))

    cancelled = []
    # 故意让接收方抛异常 —— 模拟"处理取消时出错"
    p.cancelled.connect(lambda: (_ for _ in ()).throw(RuntimeError("故意爆炸")))
    p.cancelled.connect(lambda: cancelled.append(1))
    try:
        p._cancel()
        ok = True
    except Exception as e:
        ok = False
        print("    抛出来了：", e)
    check("取消不抛异常", ok)
    check("取消信号仍然发出去了", bool(cancelled))
    check("窗口已关闭", not p.isVisible())

    # 确定路径同理
    from PySide6.QtCore import QRect
    p2 = SnapPicker(QPixmap.fromImage(img))
    p2.picked.connect(lambda r, i: (_ for _ in ()).throw(RuntimeError("又炸")))
    p2._rect = QRect(5, 5, 20, 20)
    try:
        p2._finish()
        ok2 = True
    except Exception:
        ok2 = False
    check("确定也不抛异常", ok2)
    check("确定之后窗口也关了", not p2.isVisible())


def test_hotkey_capture_mode():
    """捕获模式：能录按键，也能录鼠标侧键，且不触发正式绑定。"""
    print("\n[快捷键捕获模式（含鼠标侧键）]")
    h = HotkeyListener()
    if not h.supported:
        check("非 Windows，跳过", True)
        return
    ok = h.start()
    check("钩子能启动", ok)
    time.sleep(0.5)

    # 模拟钩子回调直接喂事件（不真的按键）
    h.set_bindings({"snap": "ctrl+alt+s"})
    h.begin_capture()
    h._match(-2, is_mouse=True)        # 侧键2（前进）
    got = None
    try:
        got = h.captured.get_nowait()
    except Exception:
        pass
    check("捕获到鼠标侧键2", got == "mouse5", str(got))
    check("捕获后自动退出捕获模式", h._capturing is False)
    check("没有误触发正式绑定", h.events.empty())

    h.begin_capture()
    h._match(-1, is_mouse=True)
    try:
        got2 = h.captured.get_nowait()
    except Exception:
        got2 = None
    check("捕获到鼠标侧键1", got2 == "mouse4", str(got2))

    # 只按修饰键不应该被录成快捷键
    h.begin_capture()
    h._match(0x11, is_mouse=False)     # VK_CONTROL
    check("只按修饰键不录", h.captured.empty())
    h.end_capture()

    # 逃生键
    h.set_escape(True)
    h._match(0x1B, is_mouse=False)     # Esc
    esc = None
    try:
        esc = h.events.get_nowait()
    except Exception:
        pass
    check("全局 Esc 能发出逃生事件", esc == "__escape__", str(esc))
    h.set_escape(False)
    h._match(0x1B, is_mouse=False)
    check("关掉逃生后 Esc 不再发事件", h.events.empty())

    h.stop()


def test_ocr_lock_timeout():
    """OCR 锁必须带超时：后台任务占着锁时，检测线程要跳过而不是死等。"""
    from screenreader.config import AppConfig
    from screenreader.worker import BoardEngine, EngineConfig

    print("\n[OCR 锁带超时]")
    e = BoardEngine(AppConfig(), backend=type("B", (), {"name": "fake"})(),
                    engine_cfg=EngineConfig())
    e._emit = lambda *a, **k: None

    # 人为把锁占住
    e._ocr_lock.acquire()
    t0 = time.time()
    out = e.read_frame_once(__import__("numpy").zeros((10, 10, 3), dtype="uint8"))
    dt = time.time() - t0
    check("拿不到锁时返回空而不是死等", out == [])
    check("确实等了（不是立刻返回）", dt > 0.5, f"{dt:.1f}s")
    check("没有卡到 8 秒以上", dt < 8, f"{dt:.1f}s")
    e._ocr_lock.release()


def test_ensure_threads():
    """检测线程死了要能自愈（这是"识别一直卡住"的根治）。"""
    from screenreader.config import AppConfig
    from screenreader.worker import BoardEngine, EngineConfig

    print("\n[检测线程自愈]")
    e = BoardEngine(AppConfig(), backend=type("B", (), {"name": "fake"})(),
                    engine_cfg=EngineConfig(poll_interval_s=0.05))
    e._emit = lambda *a, **k: None
    e.store.add_region("t", (0, 0, 10, 10))
    # 不启动真实循环，只手动放一个占位线程来验证名字判断
    e._running = True
    e._stop.clear()
    check("有 ensure_threads", hasattr(e, "ensure_threads"))
    fixed = e.ensure_threads()
    check("线程缺失时会被补上", fixed is True)
    alive = {t.name for t in e._threads if t.is_alive()}
    check("补出了 detect", "detect" in alive, str(alive))
    check("补出了 autoplay", "autoplay" in alive, str(alive))
    # 再调一次不应重复补
    fixed2 = e.ensure_threads()
    check("已经活着就不会重复补", fixed2 is False)
    e._stop.set()
    time.sleep(0.1)


def test_end_snap_restores_on_error():
    """_end_snap 中途出错也必须把其他区域的开关恢复回去。"""
    print("\n[_end_snap 出错也恢复]")

    class FakeEngine:
        def __init__(self):
            self.enabled = {"r1": False, "r2": True}
            self.applied = None
            self.removed = []
            self._playing_regions = set()
            self._para_state = {}

        def apply_region_enabled_map(self, m):
            self.applied = dict(m)
            self.enabled.update(m)

        def remove_region(self, rid):
            self.removed.append(rid)
            raise RuntimeError("模拟：删区域时炸了")

    class FakeStore:
        def region(self, rid):
            return None

    class FakeWin:
        engine = None
        _snap_rid = "temp1"
        _snap_enabled = {"r1": False, "r2": True}
        _snap_started = True
        status = type("S", (), {"showMessage": lambda *a, **k: None})()
        _snap_timer = type("T", (), {"stop": lambda *a, **k: None})()

        def _refresh_regions(self):
            pass

        def _refresh_cards(self):
            pass

    from screenreader.ui.main_window import MainWindow

    w = FakeWin()
    w.engine = FakeEngine()
    # 直接调用真的 _end_snap
    MainWindow._end_snap(w, silent=True)
    check("即使删区域失败，也恢复了开关", w.engine.applied == {"r1": False, "r2": True},
          str(w.engine.applied))
    check("确实尝试删了临时区域", w.engine.removed == ["temp1"], str(w.engine.removed))
    check("临时区域状态已清", w._snap_rid is None)


def main() -> int:
    for fn in (test_cancel_never_raises, test_hotkey_capture_mode, test_ocr_lock_timeout,
               test_ensure_threads, test_end_snap_restores_on_error):
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
